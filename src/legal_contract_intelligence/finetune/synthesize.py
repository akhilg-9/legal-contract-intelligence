"""Scale seed examples to ~3k via OpenAI-as-labeler synthesis.

Strategy:
1. For each seed example, generate N paraphrases (clause text rephrased,
   parties renamed, jurisdictions varied) while preserving the label.
2. Validate every generated label against ClauseLabel; drop invalid ones.
3. De-duplicate by clause-text hash.
4. Stratify the final set by clause_type so rare types don't get drowned out.

OpenAI is the labeling oracle here, which is standard for bootstrapping a
fine-tuning corpus from a small seed. The fine-tuned student model then
runs entirely on open weights (Qwen-3 8B); the teacher dependency is one-time.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import track

from .schema import ClauseLabel, SYSTEM_PROMPT, is_valid_label_json


app = typer.Typer(add_completion=False)
console = Console()


PARAPHRASE_PROMPT = """You will be given an original legal clause and its structured label. Produce a paraphrased version of the clause that preserves the label exactly. Vary:

- Party names (use new fictional but realistic company / person names).
- Jurisdictions (substitute another US state where applicable).
- Numeric specifics (term length, percentages, hours, dollar amounts) — but keep them legally reasonable for the clause type.
- Sentence structure and word choice.

Do NOT add new obligations, remove obligations, or change clause_type. The label must still describe the paraphrased clause correctly.

Output JSON only:
{"clause_text": "<paraphrased clause>", "label": <unchanged label JSON>}"""


def _hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()


def _client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for synthesis.")
    return OpenAI(api_key=api_key)


def _generate_paraphrase(client, original: Dict, model: str, temperature: float) -> Optional[Dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PARAPHRASE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"clause_text": original["clause_text"], "label": original["label"]},
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "clause_text" not in parsed or "label" not in parsed:
        return None
    if not is_valid_label_json(json.dumps(parsed["label"])):
        return None
    # Preserve the original clause_type — the paraphrase must not change it.
    if parsed["label"].get("clause_type") != original["label"].get("clause_type"):
        return None
    return parsed


@app.command()
def main(
    seed: str = typer.Option("data/finetune/seed/seed_examples.jsonl", help="JSONL of seed examples"),
    out: str = typer.Option("data/finetune/synthesized/train.jsonl", help="JSONL output for training set"),
    out_eval: str = typer.Option("data/finetune/eval/heldout.jsonl", help="JSONL output for held-out eval set"),
    target: int = typer.Option(3000, help="Approx. total training examples after synthesis"),
    eval_holdout: int = typer.Option(200, help="Held-out eval examples (drawn from synthesized pool)"),
    paraphrases_per_seed: int = typer.Option(0, help="Overrides target if > 0"),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model to use as labeler"),
    temperature: float = typer.Option(0.8, help="Sampling temperature for paraphrasing"),
    seed_random: int = typer.Option(42),
) -> None:
    rng = random.Random(seed_random)
    seeds = [json.loads(line) for line in Path(seed).read_text().splitlines() if line.strip()]
    console.print(f"loaded {len(seeds)} seed examples")
    n_per_seed = paraphrases_per_seed if paraphrases_per_seed > 0 else max(1, (target - len(seeds)) // len(seeds))
    console.print(f"target ~{target} synthesized rows  →  {n_per_seed} paraphrases per seed")

    client = _client()
    rows: List[Dict] = list(seeds)  # start with the seed examples themselves
    seen: set = {_hash(r["clause_text"]) for r in rows}

    for original in track(seeds, description="paraphrasing"):
        for _ in range(n_per_seed):
            paraphrase = _generate_paraphrase(
                client, original, model=model, temperature=temperature
            )
            if paraphrase is None:
                continue
            h = _hash(paraphrase["clause_text"])
            if h in seen:
                continue
            seen.add(h)
            rows.append(paraphrase)

    rng.shuffle(rows)
    eval_rows = rows[:eval_holdout]
    train_rows = rows[eval_holdout:]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out_eval).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n")
    Path(out_eval).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n")

    by_type = Counter(r["label"]["clause_type"] for r in train_rows)
    console.print(f"wrote {len(train_rows)} train rows → {out}")
    console.print(f"wrote {len(eval_rows)} eval rows  → {out_eval}")
    console.print(f"top clause_types: {by_type.most_common(8)}")


if __name__ == "__main__":
    app()
