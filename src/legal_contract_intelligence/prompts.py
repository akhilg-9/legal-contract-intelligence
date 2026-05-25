"""Versioned prompt config loader.

Prompts live in `prompts/v<N>.yaml`. The version on disk is the unit of change
that the Phase 3 Ragas eval gates on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    name: str
    temperature: float = 0.1
    max_tokens: int = 1024


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid", "reranked"] = "dense"
    top_k: int = Field(default=6, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=200)
    min_score: float = Field(default=0.0)
    min_chunks_for_answer: int = Field(default=1, ge=0)


class Templates(BaseModel):
    system: str
    user: str

    @field_validator("user")
    @classmethod
    def must_have_placeholders(cls, value: str) -> str:
        for token in ("{question}", "{context}"):
            if token not in value:
                raise ValueError(f"user template missing {token!r} placeholder")
        return value


class PromptConfig(BaseModel):
    version: str
    description: str = ""
    model: ModelConfig
    retrieval: RetrievalConfig
    templates: Templates

    @classmethod
    def load(cls, path: str | Path) -> "PromptConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"prompt config not found: {path}")
        raw = yaml.safe_load(path.read_text())
        return cls.model_validate(raw)


def load_prompt(version: Optional[str] = None, prompts_dir: str | Path = "prompts") -> PromptConfig:
    """Load `prompts/<version>.yaml`. If version is None, picks the highest v* file."""
    prompts_dir = Path(prompts_dir)
    if version is None:
        candidates = sorted(prompts_dir.glob("v*.yaml"))
        if not candidates:
            raise FileNotFoundError(f"no prompt configs found in {prompts_dir}")
        path = candidates[-1]
    else:
        path = prompts_dir / f"{version}.yaml"
    return PromptConfig.load(path)
