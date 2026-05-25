"""Runtime configuration loaded from environment + .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.environ.get("LCI_LLM_PROVIDER", "ollama")
    ollama_host: str = os.environ.get("LCI_OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.environ.get("LCI_OLLAMA_MODEL", "llama3.2:3b")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")

    embedding_model: str = os.environ.get("LCI_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_device: str = os.environ.get("LCI_EMBEDDING_DEVICE", "mps")

    qdrant_url: str = os.environ.get("LCI_QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.environ.get("LCI_QDRANT_COLLECTION", "contracts")

    edgar_user_agent: str = os.environ.get(
        "LCI_EDGAR_USER_AGENT",
        "Legal Contract Intelligence research@example.com",
    )


settings = Settings()
