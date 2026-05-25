"""End-to-end RAG pipeline.

Phase 1 design: a LangChain LCEL chain that goes
    question -> retrieve -> format excerpts -> LLM -> answer with [chunk_id] citations.

Citation discipline is enforced two ways:
1. The system prompt requires inline [chunk_id] citations and a literal
   "INSUFFICIENT_CONTEXT: ..." refusal when retrieved chunks are inadequate.
2. We post-process the answer to extract cited chunk_ids and reject answers
   that claim facts without citing anything (configurable via min_chunks_for_answer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .ingestion import chunk_document, parse_document
from .llm import build_llm
from .observability import (
    estimate_cost,
    finalize_trace,
    get_langchain_callback,
    trace_ask,
)
from .prompts import PromptConfig, load_prompt
from .retrieval import retrieve as retrieve_chunks
from .vectorstore import get_vector_store, upsert_chunks


# ---------------------------------------------------------------------------
# Ingestion entry-point
# ---------------------------------------------------------------------------


def ingest_path(path: str | Path, chunk_tokens: int = 650, overlap_tokens: int = 100) -> int:
    """Parse + chunk + embed + upsert a file or directory. Returns chunk count."""
    path = Path(path)
    files: List[Path] = []
    if path.is_dir():
        for ext in ("*.pdf", "*.html", "*.htm", "*.txt", "*.md"):
            files.extend(path.rglob(ext))
    else:
        files = [path]

    total = 0
    for file_path in files:
        document = parse_document(file_path)
        if not document.paragraphs:
            continue
        chunks = chunk_document(document, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        total += upsert_chunks(chunks)
    return total


# ---------------------------------------------------------------------------
# Retrieval + answer generation
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Answer:
    question: str
    answer: str
    citations: List[str]
    retrieved: List[RetrievedChunk]
    insufficient_context: bool
    prompt_version: str


_CITATION_RE = re.compile(r"\[([^\[\]]+?)\]")
_INSUFFICIENT_RE = re.compile(r"^\s*INSUFFICIENT_CONTEXT\b", re.IGNORECASE)


def _format_excerpts(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        title = c.metadata.get("title") or c.doc_id
        page = c.metadata.get("page_number")
        loc = f"{title}" + (f", p.{page}" if page else "")
        parts.append(f"[{c.chunk_id}] ({loc})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _docs_to_retrieved(pairs) -> List[RetrievedChunk]:
    out: List[RetrievedChunk] = []
    for doc, score in pairs:
        if not isinstance(doc, Document):
            continue
        out.append(
            RetrievedChunk(
                chunk_id=str(doc.metadata.get("chunk_id", "?")),
                doc_id=str(doc.metadata.get("doc_id", "?")),
                text=doc.page_content,
                score=float(score),
                metadata=dict(doc.metadata),
            )
        )
    return out


class RagPipeline:
    def __init__(self, prompt_config: Optional[PromptConfig] = None, provider_override: Optional[str] = None):
        self.prompt_config = prompt_config or load_prompt()
        self.llm = build_llm(self.prompt_config.model, provider_override=provider_override)
        self.vector_store = get_vector_store()
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_config.templates.system),
                ("user", self.prompt_config.templates.user),
            ]
        )

    def retrieve(self, question: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        cfg = self.prompt_config.retrieval
        k = k or cfg.top_k
        pairs = retrieve_chunks(
            question,
            mode=cfg.mode,
            top_k=k,
            candidate_k=cfg.candidate_k,
        )
        chunks = _docs_to_retrieved(pairs)
        # min_score only makes sense for cosine (dense) mode. For BM25 / RRF / reranker
        # scores are unbounded, so we skip the threshold there.
        if cfg.mode == "dense":
            chunks = [c for c in chunks if c.score >= cfg.min_score]
        return chunks

    def ask(self, question: str) -> Answer:
        retrieval_mode = self.prompt_config.retrieval.mode
        model_label = f"{self.prompt_config.model.provider}:{self.prompt_config.model.name}"

        with trace_ask(
            question=question,
            prompt_version=self.prompt_config.version,
            retrieval_mode=retrieval_mode,
            model=model_label,
        ) as trace:
            retrieved = self.retrieve(question)

            if len(retrieved) < self.prompt_config.retrieval.min_chunks_for_answer:
                answer = Answer(
                    question=question,
                    answer="INSUFFICIENT_CONTEXT: no retrieved excerpts cleared the score threshold.",
                    citations=[],
                    retrieved=retrieved,
                    insufficient_context=True,
                    prompt_version=self.prompt_config.version,
                )
                finalize_trace(
                    trace,
                    answer=answer.answer,
                    citations=[],
                    insufficient_context=True,
                    retrieved_chunks=0,
                    extra_tags=["retrieval_empty"],
                )
                return answer

            chain = self.prompt_template | self.llm | StrOutputParser()
            callbacks = []
            handler = get_langchain_callback()
            if handler is not None:
                callbacks.append(handler)
            raw = chain.invoke(
                {
                    "question": question,
                    "context": _format_excerpts(retrieved),
                },
                config={"callbacks": callbacks} if callbacks else None,
            )

            cited = sorted({m.group(1) for m in _CITATION_RE.finditer(raw)})
            valid_ids = {c.chunk_id for c in retrieved}
            cited = [c for c in cited if c in valid_ids]

            insufficient = bool(_INSUFFICIENT_RE.match(raw))
            if not insufficient and not cited:
                raw = (
                    "INSUFFICIENT_CONTEXT: model produced an uncited answer. "
                    "Raw model output suppressed; see retrieved excerpts."
                )
                insufficient = True

            answer = Answer(
                question=question,
                answer=raw,
                citations=cited,
                retrieved=retrieved,
                insufficient_context=insufficient,
                prompt_version=self.prompt_config.version,
            )

            # Best-effort token count for cost estimation. Falls back to char/4.
            input_tokens = max(1, sum(len(c.text) for c in retrieved) // 4 + len(question) // 4)
            output_tokens = max(1, len(raw) // 4)
            cost = estimate_cost(input_tokens=input_tokens, output_tokens=output_tokens)

            finalize_trace(
                trace,
                answer=answer.answer,
                citations=answer.citations,
                insufficient_context=answer.insufficient_context,
                retrieved_chunks=len(retrieved),
                cost=cost,
            )
            return answer
