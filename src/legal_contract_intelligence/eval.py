"""Phase 3 eval harness.

Runs the active RAG pipeline over the golden set, computes Ragas faithfulness/
relevancy/precision/recall, plus two structural metrics (expected_doc_recall@k
and refusal_accuracy) that don't need an LLM judge.

Designed so the same entrypoint serves local dev (`lci-eval`) and CI
(`lci-eval --smoke --format json`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .pipeline import RagPipeline
from .prompts import load_prompt


app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class GoldenItem:
    id: str
    question: str
    ground_truth: str
    expected_docs: List[str] = field(default_factory=list)
    clause_tags: List[str] = field(default_factory=list)


@dataclass
class ItemResult:
    id: str
    question: str
    answer: str
    insufficient_context: bool
    citations: List[str]
    retrieved_doc_ids: List[str]
    expected_doc_recall: bool  # at least one expected doc in retrieved
    refusal_correct: Optional[bool]  # only for refusal items
    elapsed_s: float


def load_golden(path: str | Path) -> List[GoldenItem]:
    items: List[GoldenItem] = []
    for raw in Path(path).read_text().splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        items.append(GoldenItem(**d))
    return items


def _structural_metrics(item: GoldenItem, result_doc_ids: List[str], answer: str) -> Dict[str, Any]:
    is_refusal = len(item.expected_docs) == 0
    refusal_correct: Optional[bool] = None
    if is_refusal:
        refusal_correct = answer.strip().upper().startswith("INSUFFICIENT_CONTEXT")
        expected_recall = True  # by definition not applicable
    else:
        expected_recall = any(d in result_doc_ids for d in item.expected_docs)
    return {
        "is_refusal": is_refusal,
        "expected_doc_recall": expected_recall,
        "refusal_correct": refusal_correct,
    }


# ---------------------------------------------------------------------------
# Ragas
# ---------------------------------------------------------------------------


def _ragas_metrics(items: List[GoldenItem], item_results: List[ItemResult], retrieved_texts: List[List[str]]):
    """Returns a dict of mean Ragas scores. Requires OPENAI_API_KEY (Ragas judge)."""
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[yellow]OPENAI_API_KEY not set — skipping Ragas (LLM-as-judge) metrics.[/yellow]")
        return {}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        console.print(f"[yellow]Ragas unavailable: {exc}; skipping.[/yellow]")
        return {}

    ds = Dataset.from_list(
        [
            {
                "question": item.question,
                "answer": r.answer,
                "contexts": ctx,
                "ground_truth": item.ground_truth,
            }
            for item, r, ctx in zip(items, item_results, retrieved_texts)
        ]
    )
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    # Flatten to plain floats
    out: Dict[str, float] = {}
    for k, v in result.to_pandas().mean(numeric_only=True).items():
        out[str(k)] = float(v)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval(
    golden_path: str = "evals/golden_set.jsonl",
    prompt_version: Optional[str] = None,
    smoke: bool = False,
    smoke_n: int = 3,
    skip_ragas: bool = False,
) -> Dict[str, Any]:
    prompt_config = load_prompt(version=prompt_version)
    pipeline = RagPipeline(prompt_config=prompt_config)

    items = load_golden(golden_path)
    if smoke:
        # Always include at least one refusal in the smoke set if available.
        refusals = [i for i in items if not i.expected_docs]
        positives = [i for i in items if i.expected_docs]
        picked: List[GoldenItem] = positives[: max(1, smoke_n - len(refusals[:1]))]
        if refusals:
            picked.append(refusals[0])
        items = picked

    results: List[ItemResult] = []
    retrieved_texts: List[List[str]] = []
    structural: List[Dict[str, Any]] = []

    for item in items:
        t0 = time.monotonic()
        answer = pipeline.ask(item.question)
        elapsed = time.monotonic() - t0
        retrieved_doc_ids = sorted({c.doc_id for c in answer.retrieved})
        retrieved_texts.append([c.text for c in answer.retrieved])
        st = _structural_metrics(item, retrieved_doc_ids, answer.answer)
        structural.append(st)
        results.append(
            ItemResult(
                id=item.id,
                question=item.question,
                answer=answer.answer,
                insufficient_context=answer.insufficient_context,
                citations=answer.citations,
                retrieved_doc_ids=retrieved_doc_ids,
                expected_doc_recall=st["expected_doc_recall"],
                refusal_correct=st["refusal_correct"],
                elapsed_s=elapsed,
            )
        )

    structural_summary = {
        "expected_doc_recall@k": _mean([s["expected_doc_recall"] for s in structural if not s["is_refusal"]]),
        "refusal_accuracy": _mean([s["refusal_correct"] for s in structural if s["is_refusal"]]),
        "avg_latency_s": _mean([r.elapsed_s for r in results]),
        "n_total": len(items),
        "n_refusal": sum(1 for s in structural if s["is_refusal"]),
    }

    ragas_summary = {} if skip_ragas else _ragas_metrics(items, results, retrieved_texts)

    return {
        "prompt_version": prompt_config.version,
        "retrieval_mode": prompt_config.retrieval.mode,
        "model": f"{prompt_config.model.provider}:{prompt_config.model.name}",
        "ragas": ragas_summary,
        "structural": structural_summary,
        "per_item": [asdict(r) for r in results],
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    golden: str = typer.Option("evals/golden_set.jsonl", help="Path to golden_set.jsonl"),
    prompt: Optional[str] = typer.Option(None, help="Prompt version, e.g. v1, v2. Default = highest."),
    smoke: bool = typer.Option(False, "--smoke", help="Run only a small subset (CI default)."),
    smoke_n: int = typer.Option(3, help="Number of items in --smoke mode."),
    skip_ragas: bool = typer.Option(False, "--skip-ragas", help="Skip LLM-as-judge metrics."),
    fmt: str = typer.Option("table", "--format", help="table | json"),
    out: Optional[str] = typer.Option(None, help="Write JSON results to this path."),
    fail_under_faithfulness: float = typer.Option(0.70, help="CI gate: fail if faithfulness < this."),
    fail_under_refusal: float = typer.Option(0.80, help="CI gate: fail if refusal_accuracy < this."),
) -> None:
    result = run_eval(
        golden_path=golden,
        prompt_version=prompt,
        smoke=smoke,
        smoke_n=smoke_n,
        skip_ragas=skip_ragas,
    )

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2))
        console.print(f"[green]wrote {out}[/green]")

    if fmt == "json":
        print(json.dumps(result, indent=2))
    else:
        _print_table(result)

    # CI gates
    failed = []
    faith = (result.get("ragas") or {}).get("faithfulness")
    if faith is not None and faith < fail_under_faithfulness:
        failed.append(f"faithfulness {faith:.3f} < {fail_under_faithfulness}")
    refusal = (result.get("structural") or {}).get("refusal_accuracy")
    if refusal is not None and refusal < fail_under_refusal:
        failed.append(f"refusal_accuracy {refusal:.3f} < {fail_under_refusal}")
    if failed:
        console.print("[red]EVAL FAILED:[/red] " + "; ".join(failed))
        sys.exit(2)


def _print_table(result: Dict[str, Any]) -> None:
    head = (
        f"prompt: {result['prompt_version']}    "
        f"retrieval: {result['retrieval_mode']}    model: {result['model']}"
    )
    console.print(head)

    structural = result.get("structural") or {}
    ragas = result.get("ragas") or {}

    table = Table(title="Summary", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    for k, v in {**structural, **ragas}.items():
        if v is None:
            disp = "n/a"
        elif isinstance(v, float):
            disp = f"{v:.3f}"
        else:
            disp = str(v)
        table.add_row(k, disp)
    console.print(table)


if __name__ == "__main__":
    app()
