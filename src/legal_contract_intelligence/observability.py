"""Phase 4 observability via Langfuse.

We wire Langfuse two ways:

1. **LangChain callback** — auto-traces every LCEL invocation (retrieval, LLM call,
   parser) inside the pipeline. Zero touch on the chain code.
2. **Explicit span around the whole ask()** — gives us one parent trace per user
   question with our own tags (retrieval_mode, prompt_version, citation_count,
   insufficient_context). Failures are tagged so Langfuse can filter on them.

Both are no-ops when LANGFUSE_PUBLIC_KEY is not set, so the pipeline runs
unchanged offline.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional

from .config import settings


@dataclass
class CostBreakdown:
    input_tokens: int
    output_tokens: int
    estimated_usd: float


def estimate_cost(input_tokens: int, output_tokens: int) -> CostBreakdown:
    usd = (
        settings.cost_per_request_overhead_usd
        + (input_tokens / 1_000_000.0) * settings.cost_per_m_input_tokens_usd
        + (output_tokens / 1_000_000.0) * settings.cost_per_m_output_tokens_usd
    )
    return CostBreakdown(input_tokens=input_tokens, output_tokens=output_tokens, estimated_usd=usd)


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Returns a Langfuse client if configured, else None. Cached."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def get_langchain_callback():
    """Returns a CallbackHandler if Langfuse is configured, else None."""
    if get_langfuse_client() is None:
        return None
    try:
        from langfuse.callback import CallbackHandler  # type: ignore[import-not-found]
    except ImportError:
        try:
            from langfuse.langchain import CallbackHandler  # type: ignore[import-not-found]
        except ImportError:
            return None
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


@contextmanager
def trace_ask(
    question: str,
    prompt_version: str,
    retrieval_mode: str,
    model: str,
) -> Iterator[Optional[Any]]:
    """Open a parent Langfuse trace for one ask() call. Yields the trace object.

    Tags are searchable in the Langfuse UI so the "root-cause walkthrough"
    workflow becomes: filter by tag, click into trace, see retrieval + LLM spans.
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    trace = client.trace(
        name="rag.ask",
        input={"question": question},
        metadata={
            "prompt_version": prompt_version,
            "retrieval_mode": retrieval_mode,
            "model": model,
        },
        tags=[f"prompt:{prompt_version}", f"retrieval:{retrieval_mode}", f"model:{model}"],
    )
    try:
        yield trace
    finally:
        try:
            client.flush()
        except Exception:
            pass


def finalize_trace(
    trace: Optional[Any],
    *,
    answer: str,
    citations: List[str],
    insufficient_context: bool,
    retrieved_chunks: int,
    cost: Optional[CostBreakdown] = None,
    extra_tags: Optional[List[str]] = None,
) -> None:
    """Attach outcome data + tags to the trace, then close it.

    Failure-rate dashboards in Langfuse filter on `tag:insufficient_context` and
    `tag:no_citations` (set when the answer lacks any valid citation).
    """
    if trace is None:
        return

    tags: List[str] = list(extra_tags or [])
    if insufficient_context:
        tags.append("insufficient_context")
    if not citations and not insufficient_context:
        tags.append("no_citations")

    update_kwargs: Dict[str, Any] = {
        "output": {
            "answer": answer,
            "citations": citations,
            "insufficient_context": insufficient_context,
            "retrieved_chunks": retrieved_chunks,
        },
    }
    if tags:
        update_kwargs["tags"] = tags
    if cost is not None:
        update_kwargs["metadata"] = {
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "estimated_usd": cost.estimated_usd,
        }
    try:
        trace.update(**update_kwargs)
    except Exception:
        pass
