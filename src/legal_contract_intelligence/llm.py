"""LLM backend selector.

Phase 1 supports two providers:
- ollama (default) — open-weights, local, no API cost
- openai          — frontier route for comparison

Returning a langchain_core.runnables.Runnable lets the pipeline pipe directly
into LCEL chains: `prompt | llm | parser`.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import settings
from .prompts import ModelConfig


def build_llm(model: ModelConfig, provider_override: Optional[str] = None) -> BaseChatModel:
    provider = (provider_override or model.provider).lower()
    if provider == "ollama":
        return ChatOllama(
            model=model.name,
            base_url=settings.ollama_host,
            temperature=model.temperature,
            num_predict=model.max_tokens,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env or switch the prompt config back to provider: ollama."
            )
        return ChatOpenAI(
            model=model.name,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            api_key=settings.openai_api_key,
        )
    raise ValueError(f"unknown LLM provider: {provider!r}")
