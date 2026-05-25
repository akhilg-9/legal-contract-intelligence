"""Qdrant-backed vector store.

Thin wrapper around langchain_qdrant.QdrantVectorStore so the rest of the
pipeline stays LangChain-native (Document I/O, retrievers, LCEL).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings
from .embeddings import get_embeddings
from .ingestion.chunking import Chunk


# bge-large-en-v1.5 produces 1024-dim normalized embeddings.
EMBEDDING_DIM = 1024


def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(collection_name: Optional[str] = None) -> None:
    name = collection_name or settings.qdrant_collection
    client = _client()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=EMBEDDING_DIM,
            distance=qmodels.Distance.COSINE,
        ),
    )


def get_vector_store(collection_name: Optional[str] = None) -> QdrantVectorStore:
    name = collection_name or settings.qdrant_collection
    ensure_collection(name)
    return QdrantVectorStore(
        client=_client(),
        collection_name=name,
        embedding=get_embeddings(),
    )


def chunks_to_documents(chunks: Iterable[Chunk]) -> List[Document]:
    return [
        Document(
            page_content=c.text,
            metadata={"chunk_id": c.chunk_id, **c.metadata},
        )
        for c in chunks
    ]


def upsert_chunks(chunks: Iterable[Chunk], collection_name: Optional[str] = None) -> int:
    docs = chunks_to_documents(chunks)
    if not docs:
        return 0
    store = get_vector_store(collection_name)
    store.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    return len(docs)


def collection_info(collection_name: Optional[str] = None) -> dict:
    name = collection_name or settings.qdrant_collection
    client = _client()
    info = client.get_collection(collection_name=name)
    return {
        "name": name,
        "points_count": info.points_count,
        "vectors_count": getattr(info, "vectors_count", None),
        "status": str(info.status),
    }
