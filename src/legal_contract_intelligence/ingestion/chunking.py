"""Token-aware chunking.

Uses LangChain's RecursiveCharacterTextSplitter with a tiktoken length function
for true 500-800 token chunks with 100-token overlap. Paragraph boundaries are
preferred separators, so chunks rarely split mid-sentence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parsers import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


def _make_splitter(chunk_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator=True,
    )


def chunk_document(
    document: ParsedDocument,
    chunk_tokens: int = 650,
    overlap_tokens: int = 100,
) -> List[Chunk]:
    """Split a parsed document into token-bounded chunks with paragraph metadata."""
    splitter = _make_splitter(chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

    paragraph_offsets: List[int] = []
    full_text_parts: List[str] = []
    running = 0
    for paragraph in document.paragraphs:
        paragraph_offsets.append(running)
        full_text_parts.append(paragraph)
        running += len(paragraph) + 2  # account for the "\n\n" join
    full_text = "\n\n".join(full_text_parts)

    raw_chunks = splitter.split_text(full_text)
    chunks: List[Chunk] = []
    cursor = 0
    for i, raw in enumerate(raw_chunks):
        # Locate this chunk in the original text to recover paragraph indices.
        start = full_text.find(raw, cursor)
        if start == -1:
            start = cursor  # defensive: should not happen in normal runs
        end = start + len(raw)
        cursor = max(cursor, end - overlap_tokens * 4)  # heuristic advance

        first_paragraph = _paragraph_index_at(start, paragraph_offsets)
        last_paragraph = _paragraph_index_at(end, paragraph_offsets)

        page_number = _page_for_paragraph(first_paragraph, document.page_breaks)

        chunk_id = _stable_chunk_id(document.doc_id, i)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=raw,
                metadata={
                    "doc_id": document.doc_id,
                    "source_path": document.source_path,
                    "title": document.title,
                    "chunk_index": i,
                    "first_paragraph": first_paragraph,
                    "last_paragraph": last_paragraph,
                    "page_number": page_number,
                    "token_count": _token_len(raw),
                },
            )
        )
    return chunks


def _paragraph_index_at(char_offset: int, offsets: List[int]) -> int:
    lo, hi = 0, len(offsets) - 1
    if hi < 0:
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offsets[mid] <= char_offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _page_for_paragraph(paragraph_idx: int, page_breaks: List[int]) -> Optional[int]:
    if not page_breaks:
        return None
    # page_breaks[i] is the paragraph index at which page i begins.
    page = 0
    for i, start in enumerate(page_breaks):
        if start <= paragraph_idx:
            page = i
        else:
            break
    return page + 1  # 1-indexed pages


def _stable_chunk_id(doc_id: str, index: int) -> str:
    digest = hashlib.sha1(f"{doc_id}::{index}".encode()).hexdigest()[:10]
    return f"{doc_id}#{index:04d}-{digest}"
