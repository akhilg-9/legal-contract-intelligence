"""Embedding model wrapper.

Wraps sentence-transformers via LangChain's HuggingFaceEmbeddings so the same
object plugs directly into langchain_qdrant.QdrantVectorStore.

For the BGE family, queries must be prefixed with the retrieval instruction —
we do that manually for portability across LangChain versions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _device() -> str:
    requested = settings.embedding_device.lower()
    if requested in {"mps", "cuda", "cpu"}:
        return requested
    return "cpu"


def _needs_bge_prefix(model_name: str) -> bool:
    name = model_name.lower()
    # BGE-v1.5 family wants the retrieval instruction; v2-style models don't.
    return "bge" in name and "v1.5" in name


class BgePrefixedEmbeddings(Embeddings):
    """Wraps an Embeddings object to prepend the BGE query instruction at query time."""

    def __init__(self, inner: Embeddings, query_prefix: str = BGE_QUERY_PREFIX) -> None:
        self._inner = inner
        self._query_prefix = query_prefix

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(f"{self._query_prefix}{text}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    inner = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": _device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    if _needs_bge_prefix(settings.embedding_model):
        return BgePrefixedEmbeddings(inner)
    return inner


def embed_documents(texts: List[str]) -> List[List[float]]:
    return get_embeddings().embed_documents(texts)


def embed_query(text: str) -> List[float]:
    return get_embeddings().embed_query(text)
