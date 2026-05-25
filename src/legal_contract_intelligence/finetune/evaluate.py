"""Deterministic eval of the fine-tuned clause extractor.

Three LLM-free metrics:
1. json_validity        — fraction of outputs that parse as a valid ClauseLabel.
2. clause_type_exact    — fraction with the exact gold clause_type.
3. obligations_jaccard  — set Jaccard between predicted and gold obligation strings.
4. refusal_correctness  — fraction of out-of-scope clauses correctly labelled
                          clause_type="none".

The fine-tuned adapter (PEFT LoRA) is merged with the base model in-memory for
inference. If --adapter is unset, the base model is evaluated raw, which is a
useful "pre-tuning baseline" anchor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .schema import ClauseLabel, SYSTEM_PROMPT, is_valid_label_json, parse_label


app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class EvalSummary:
    n: int
    json_validity: float
    clause_type_exact: float
    obligations_jaccard: float
    refusal_correctness: float
    per_type_accuracy: Dict[str, float] = field(default_factory=dict)


def _jaccard(a: List[str], b: List[str]) -> float:
    set_a, set_b = {x.strip().lower() for x in a}, {x.strip().lower() for x in b}
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def _build_model(base_model: str, adapter: Optional[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter or base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return model, tokenizer


def _infer(model, tokenizer, clause_text: str, max_new_tokens: int = 512) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": clause_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


@app.command()
def main(
    base_model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct"),
    adapter: Optional[str] = typer.Option(None, help="Path to a PEFT adapter to merge"),
    eval_path: str = typer.Option("data/finetune/eval/heldout.jsonl", "--eval"),
    out: Optional[str] = typer.Option(None, help="Write summary JSON to this path"),
    n_limit: int = typer.Option(0, help="Cap evaluation at N examples (0 = all)"),
) -> None:
    rows = [json.loads(line) for line in Path(eval_path).read_text().splitlines() if line.strip()]
    if n_limit > 0:
        rows = rows[:n_limit]

    model, tokenizer = _build_model(base_model=base_model, adapter=adapter)
    console.print(f"evaluating on {len(rows)} examples  (adapter={adapter or 'NONE — base model'})")

    valid = 0
    type_correct = 0
    jaccards: List[float] = []
    refusal_total = 0
    refusal_correct = 0
    type_hits: Dict[str, int] = {}
    type_seen: Dict[str, int] = {}

    for row in rows:
        gold = row["label"]
        gold_type = gold.get("clause_type", "none")
        type_seen[gold_type] = type_seen.get(gold_type, 0) + 1

        raw = _infer(model, tokenizer, row["clause_text"])
        if not is_valid_label_json(raw):
            continue
        valid += 1
        pred: ClauseLabel = parse_label(raw)

        if pred.clause_type == gold_type:
            type_correct += 1
            type_hits[gold_type] = type_hits.get(gold_type, 0) + 1
        if gold_type == "none":
            refusal_total += 1
            if pred.clause_type == "none":
                refusal_correct += 1
        jaccards.append(_jaccard(pred.obligations, gold.get("obligations", [])))

    summary = EvalSummary(
        n=len(rows),
        json_validity=valid / len(rows) if rows else 0.0,
        clause_type_exact=type_correct / len(rows) if rows else 0.0,
        obligations_jaccard=sum(jaccards) / len(jaccards) if jaccards else 0.0,
        refusal_correctness=(refusal_correct / refusal_total) if refusal_total else 0.0,
        per_type_accuracy={
            t: (type_hits.get(t, 0) / count) for t, count in type_seen.items()
        },
    )

    table = Table(title="Clause Extractor Eval")
    table.add_column("metric"); table.add_column("value", justify="right")
    table.add_row("n", str(summary.n))
    table.add_row("json_validity", f"{summary.json_validity:.3f}")
    table.add_row("clause_type_exact", f"{summary.clause_type_exact:.3f}")
    table.add_row("obligations_jaccard", f"{summary.obligations_jaccard:.3f}")
    table.add_row("refusal_correctness", f"{summary.refusal_correctness:.3f}")
    console.print(table)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(
            json.dumps(
                {
                    "summary": summary.__dict__,
                    "adapter": adapter,
                    "base_model": base_model,
                },
                indent=2,
            )
        )
        console.print(f"[green]wrote {out}[/green]")


if __name__ == "__main__":
    app()
