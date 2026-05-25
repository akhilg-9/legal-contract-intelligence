"""QLoRA fine-tuning entrypoint for the clause extractor.

This is a TRL-native SFT loop using PEFT (LoRA) on top of 4-bit quantization
(bitsandbytes). Designed to run on a single A100 80GB or 2× A10 40GB. **Does
not run on Apple Silicon MPS** — bitsandbytes does not support MPS. For a Mac
local path, the README points at mlx-lm.

Typical invocation (RunPod / Lambda):
    python -m legal_contract_intelligence.finetune.train \\
        --train data/finetune/synthesized/train.jsonl \\
        --eval  data/finetune/eval/heldout.jsonl       \\
        --base-model Qwen/Qwen2.5-7B-Instruct          \\
        --output runs/qlora-clause-extractor           \\
        --epochs 2 --batch 4 --lr 2e-4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(add_completion=False)


def _load_dataset(path: str):
    from datasets import Dataset

    from .schema import format_training_example

    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(format_training_example(item["clause_text"], item["label"]))
    return Dataset.from_list(rows)


@app.command()
def main(
    train: str = typer.Option("data/finetune/synthesized/train.jsonl"),
    eval: str = typer.Option("data/finetune/eval/heldout.jsonl"),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model id. Swap to 'Qwen/Qwen3-8B-Instruct' once available.",
    ),
    output: str = typer.Option("runs/qlora-clause-extractor"),
    epochs: int = typer.Option(2),
    batch: int = typer.Option(4, help="Per-device batch size"),
    grad_accum: int = typer.Option(4, help="Gradient accumulation steps"),
    lr: float = typer.Option(2e-4),
    warmup_ratio: float = typer.Option(0.03),
    max_seq_len: int = typer.Option(2048),
    lora_r: int = typer.Option(16),
    lora_alpha: int = typer.Option(32),
    lora_dropout: float = typer.Option(0.05),
    save_steps: int = typer.Option(100),
    eval_steps: int = typer.Option(100),
    logging_steps: int = typer.Option(20),
) -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    Path(output).mkdir(parents=True, exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    train_ds = _load_dataset(train)
    eval_ds = _load_dataset(eval)

    cfg = SFTConfig(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        max_seq_length=max_seq_len,
        packing=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=3,
        report_to=["none"],
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    print(f"saved adapter + tokenizer to {output}")


if __name__ == "__main__":
    app()
