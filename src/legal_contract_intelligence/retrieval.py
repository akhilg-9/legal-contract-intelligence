"""Phase 2 retrieval: dense + BM25 + RRF + cross-encoder reranker.

Retrieval modes:
- dense      : vector cosine over Qdrant only
- sparse     : BM25 over the same corpus only
- hybrid     : reciprocal rank fusion (RRF) of dense + sparse
- reranked   : hybrid → cross-encoder rerank (bge-reranker-v2-m3)

The reranker is the heaviest stage and only fires when mode == "reranked".
All four modes return the same Document type, so the rest of the pipeline is
mode-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import settings
from .vectorstore import get_vector_store


RetrievalMode = Literal["dense", "sparse", "hybrid", "reranked"]


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# BM25 corpus
# ---------------------------------------------------------------------------


@dataclass
class _BM25Index:
    docs: List[Document]
    bm25: BM25Okapi
    corpus_signature: str  # invalidate cache if Qdrant grew


def _scroll_all_documents(collection_name: Optional[str] = None) -> List[Document]:
    """Read every chunk back from Qdrant. Fine at Phase-1/2 corpus sizes (low thousands).

    For larger corpora, replace with a persisted BM25 index that updates on upsert.
    """
    store = get_vector_store(collection_name)
    client = store.client
    name = collection_name or settings.qdrant_collection
    docs: List[Document] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            page = payload.get("page_content") or payload.get("text") or ""
            metadata = {k: v for k, v in payload.items() if k not in {"page_content", "text"}}
            if "metadata" in metadata and isinstance(metadata["metadata"], dict):
                metadata = {**metadata, **metadata.pop("metadata")}
            metadata.setdefault("chunk_id", str(p.id))
            docs.append(Document(page_content=page, metadata=metadata))
        if offset is None:
            break
    return docs


_bm25_cache: Optional[_BM25Index] = None


def _build_bm25(collection_name: Optional[str] = None) -> _BM25Index:
    docs = _scroll_all_documents(collection_name)
    tokenized = [_tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[""]])
    signature = f"{len(docs)}::{sum(len(t) for t in tokenized)}"
    return _BM25Index(docs=docs, bm25=bm25, corpus_signature=signature)


def get_bm25_index(force_refresh: bool = False, collection_name: Optional[str] = None) -> _BM25Index:
    global _bm25_cache
    if _bm25_cache is None or force_refresh:
        _bm25_cache = _build_bm25(collection_name)
    return _bm25_cache


def bm25_search(query: str, k: int = 10, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
    idx = get_bm25_index(collection_name=collection_name)
    if not idx.docs:
        return []
    scores = idx.bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(idx.docs, scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:k]]


def dense_search(query: str, k: int = 10, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
    store = get_vector_store(collection_name)
    return store.similarity_search_with_score(query, k=k)


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    *result_lists: List[Tuple[Document, float]],
    k_constant: int = 60,
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
    """RRF fusion. Score = sum_over_lists(1 / (k_constant + rank))."""
    scores: Dict[str, float] = {}
    doc_by_id: Dict[str, Document] = {}
    for results in result_lists:
        for rank, (doc, _) in enumerate(results, start=1):
            doc_id = str(doc.metadata.get("chunk_id") or id(doc))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_constant + rank)
            doc_by_id.setdefault(doc_id, doc)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_by_id[did], score) for did, score in fused]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_cross_encoder():
    # Lazy import — only loaded when reranked mode is used.
    from sentence_transformers import CrossEncoder

    device = settings.embedding_device.lower()
    if device not in {"mps", "cuda", "cpu"}:
        device = "cpu"
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)


def rerank(
    query: str,
    candidates: List[Tuple[Document, float]],
    top_k: int = 8,
) -> List[Tuple[Document, float]]:
    if not candidates:
        return []
    ce = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc, _ in candidates]
    scores = ce.predict(pairs)
    rescored = list(zip([d for d, _ in candidates], [float(s) for s in scores]))
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_k]


# ---------------------------------------------------------------------------
# Unified entrypoint
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    mode: RetrievalMode = "reranked",
    top_k: int = 8,
    candidate_k: int = 20,
    collection_name: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """Single retrieval entrypoint used by the pipeline.

    `candidate_k` is the per-source breadth before fusion / reranking;
    `top_k` is the final breadth returned to the LLM.
    """
    if mode == "dense":
        return dense_search(query, k=top_k, collection_name=collection_name)
    if mode == "sparse":
        return bm25_search(query, k=top_k, collection_name=collection_name)

    dense_hits = dense_search(query, k=candidate_k, collection_name=collection_name)
    sparse_hits = bm25_search(query, k=candidate_k, collection_name=collection_name)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, top_k=candidate_k)

    if mode == "hybrid":
        return fused[:top_k]
    if mode == "reranked":
        return rerank(query, fused, top_k=top_k)

    raise ValueError(f"unknown retrieval mode: {mode!r}")
