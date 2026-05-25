"""Render training curves from a TRL/Trainer `trainer_state.json` file.

Run after training:
    python -m legal_contract_intelligence.finetune.plot \\
        --state runs/qlora-clause-extractor/trainer_state.json \\
        --out   runs/qlora-clause-extractor/curves.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import typer


app = typer.Typer(add_completion=False)


@app.command()
def main(
    state: str = typer.Option(..., help="Path to trainer_state.json"),
    out: str = typer.Option("training_curves.png", help="Output PNG path"),
) -> None:
    import matplotlib.pyplot as plt

    data = json.loads(Path(state).read_text())
    history = data.get("log_history", [])
    train_steps: List[int] = []
    train_loss: List[float] = []
    eval_steps: List[int] = []
    eval_loss: List[float] = []

    for entry in history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_loss:
        ax.plot(train_steps, train_loss, label="train loss", alpha=0.8)
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="eval loss", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("QLoRA SFT — clause extractor")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
