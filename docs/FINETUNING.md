# Fine-tuning (Phase 5)

Trains a clause-extractor LoRA adapter on top of Qwen-3 8B (substitute Qwen2.5-7B-Instruct until Qwen-3 ships under that exact HF id). The model takes a single clause and returns a structured `ClauseLabel` JSON:

```json
{
  "clause_type": "ip_assignment",
  "parties": ["Contractor", "Company"],
  "obligations": ["Work Product = work-for-hire to extent permitted", "..."],
  "risk_flags": ["broad_ip_assignment"]
}
```

This is the open-weights student. The teacher (OpenAI) is used **once** to scale the 25 hand-crafted seeds to a 3k training set; the runtime never touches an external API.

## End-to-end recipe

```bash
# 0. Need a CUDA GPU. M-series Mac doesn't work — bitsandbytes is CUDA-only.
#    Cheapest viable target: RunPod A100 80GB at ~$1.50–$2.00/h.

# 1. Synthesize 25 seeds → 3000 paraphrased train rows + 200 held-out eval rows.
#    Uses OpenAI (gpt-4o-mini) as the labeler. One-time cost ≈ a few dollars.
export OPENAI_API_KEY=sk-...
python -m legal_contract_intelligence.finetune.synthesize \
    --target 3000 --eval-holdout 200

# 2. Pre-tune baseline (anchors the lift).
python -m legal_contract_intelligence.finetune.evaluate \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --out runs/eval/pretune.json

# 3. QLoRA SFT — TRL path.
python -m legal_contract_intelligence.finetune.train \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --output runs/qlora-clause-extractor \
    --epochs 2 --batch 4 --lr 2e-4

# 3'. Alternative: Axolotl path (identical hyperparameters, different ergonomics).
accelerate launch -m axolotl.cli.train configs/axolotl_qwen_qlora.yaml

# 4. Post-tune eval.
python -m legal_contract_intelligence.finetune.evaluate \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter runs/qlora-clause-extractor \
    --out runs/eval/posttune.json

# 5. Plot training curves.
python -m legal_contract_intelligence.finetune.plot \
    --state runs/qlora-clause-extractor/trainer_state.json \
    --out runs/qlora-clause-extractor/curves.png
```

## What gets measured

Four LLM-free metrics on the 200-example held-out set:

| Metric | What it catches |
| --- | --- |
| `json_validity` | model outputs parseable JSON conforming to `ClauseLabel`. |
| `clause_type_exact` | exact match on the controlled-vocabulary clause type. |
| `obligations_jaccard` | set overlap on the obligation strings. |
| `refusal_correctness` | `clause_type="none"` on out-of-scope inputs. |

The pre-tune anchor matters: if base Qwen already nails `clause_type_exact > 0.85`, the LoRA mainly buys you `json_validity` improvements and reduced verbosity. If it doesn't (which is the realistic case for legal verbiage), the LoRA earns its keep on type accuracy directly. The pre-tune column in `BENCHMARKS.md` is the proof.

## Data pipeline

- **Seed (25):** `data/finetune/seed/seed_examples.jsonl`. Hand-crafted from the four sample contracts. Covers all 25 clause types (24 real + 1 `none` for refusals). This is the most important file — quality > quantity. A bad seed → bad paraphrases → bad model.
- **Synthesized (3k):** `data/finetune/synthesized/train.jsonl`. Generated on-demand by `synthesize.py`, not committed (too large + reproducible). Each seed gets ~120 paraphrases at temperature 0.8, varying party names, jurisdictions, numeric specifics, and sentence structure.
- **Held-out eval (200):** `data/finetune/eval/heldout.jsonl`. Drawn from the synthesized pool before the train split, so the model never sees them.

The teacher's `clause_type` is checked for stability — any paraphrase whose label drifts (e.g., reframes a non-solicit as a non-compete) is dropped. See `WHAT_BROKE.md` Phase 5 for the rationale and ~3% loss rate.

## Why QLoRA (and not full SFT)

- An 8B model in 4-bit + LoRA adapters fits on a single A100 80GB with room.
- Full SFT would require multi-GPU and ~10× the compute for marginal gain on this kind of structured-extraction task.
- LoRA adapters are tiny (~50–200 MB) and ship cleanly alongside the base model.

## Why Qwen-3 8B (planned) / Qwen2.5-7B-Instruct (current)

- Strong open-weights instruct model, top-tier at structured generation.
- Apache 2.0 licensed — no commercial gotchas.
- Available on Hugging Face Hub; integrates cleanly with TRL and Axolotl.

When Qwen-3 8B lands under a stable HF id, swap the `--base-model` flag — no other code changes.
