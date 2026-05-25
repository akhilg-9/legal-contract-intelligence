"""Phase 5: fine-tuning utilities for clause-extraction.

Public surface:
- finetune.synthesize : OpenAI-powered example synthesizer to scale 25 seed → 3k
- finetune.train      : TRL-based QLoRA training entrypoint (Qwen-3 8B by default)
- finetune.evaluate   : deterministic eval (JSON validity, EM, refusal correctness)
- finetune.plot       : training-curve plot generator from TRL logs
"""

__all__ = ["synthesize", "train", "evaluate", "plot"]
