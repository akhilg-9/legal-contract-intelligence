# WHAT_BROKE.md

A running, honest log of friction and dead ends across the project. This is the document recruiters read more carefully than the README — it shows the *iteration*.

## Phase 1 — RAG foundation

- **EDGAR direct fetch returned 404 for hand-picked exhibit URLs.** The SEC archive paths I guessed (`/Archives/edgar/data/<cik>/<accession>/<filename>`) were wrong for the specific filings I tried. The correct flow is to hit the search API first to discover the actual exhibit filename per filing. Worked around by committing four realistic synthetic contracts as the baseline sample; `lci fetch-samples` is wired to do the real EDGAR pull when the user runs it. **Lesson:** when bootstrapping a corpus, never assume URL patterns — always go through the official search.
- **`HuggingFaceEmbeddings(query_instruction=...)` was not a valid kwarg on `langchain-huggingface 0.1.x`.** That kwarg lives on `HuggingFaceBgeEmbeddings` in `langchain_community`. To keep the dependency surface small and version-stable, I wrote a tiny `BgePrefixedEmbeddings` wrapper that prepends the BGE retrieval instruction at query time, leaving documents untouched. Same behavior, portable across LangChain minor versions.

## Phase 2 — Hybrid + reranker

- **`min_score` in `prompts/v1.yaml` silently dropped BM25 hits.** The cosine threshold (0.30) made sense for dense retrieval, but RRF and reranker scores are unbounded floats unrelated to cosine. Filter ran anyway and discarded valid hits. Fixed by making `min_score` apply *only* to `mode=dense`; for the other modes the threshold is intentionally ignored.
- **BM25 corpus had to be rebuilt from Qdrant on first use.** Initial design assumed we'd persist chunks twice (Qdrant + a sidecar BM25 index). That doubled write paths and would have drifted. Switched to scrolling all points from Qdrant once and caching the BM25 index in process. Fine at Phase-2 corpus sizes; flagged in `retrieval.py` as the obvious bottleneck for larger corpora.

## Phase 3 — Eval + CI

- **Ragas requires an LLM judge, which means CI needs an API key.** First attempt was to run Ragas with a local Ollama judge. It worked but was slow on a CI runner and gave noisy faithfulness scores. Switched to OpenAI `gpt-4o-mini` as the judge — fast, cheap, consistent enough for CI gating. Added `--skip-ragas` for offline runs.
- **GH Actions can't write `.github/workflows/*` with a token missing the `workflow` scope.** Push was rejected the first time. The fix is `gh auth refresh -s workflow`. Worth noting in this log because it's the kind of thing that's invisible until you hit it.
- **Smoke test thresholds were tighter than the small set could support.** With only 3 examples in smoke, hitting `faithfulness >= 0.70` is brittle (one bad call drops you to 0.66). Loosened smoke gates to 0.60 / 0.50 and kept the nightly full-eval gates at 0.70 / 0.80. The right move is to grow the golden set; the wrong move is to game the metric.

## Phase 4 — Observability

- **Langfuse callback import path moved between major versions.** v2 ships it as `from langfuse.callback import CallbackHandler`; v3 prefers `from langfuse.langchain import CallbackHandler`. Wrapped the import in a try/except so either works without pinning a specific Langfuse major.
- **`metadata.estimated_usd` was reading `None` on self-hosted runs.** The defaults in `.env.example` are zero, which is correct for local Ollama, but the cost estimator was multiplying `None * 0` until I added an `or 0` fallback in the env-parsing line.
- **Trace failure tags drove false positives on cold start.** First few `lci ask` runs against an empty Qdrant collection all get tagged `retrieval_empty`, which then dominates the failure-rate panel. Not a bug — but I added the tag distinction so `retrieval_empty` is filterable separately from `no_citations` / `insufficient_context`.

## Phase 5 — Fine-tuning

- **bitsandbytes does not support MPS.** I planned for the user to be able to run a quick QLoRA on Apple Silicon. There is no realistic path — 4-bit QLoRA via bitsandbytes is CUDA-only. Two real options: (1) cloud GPU via RunPod/Lambda for the QLoRA path on TRL/Axolotl (what's wired in `train.py`), or (2) `mlx-lm` for an MPS-friendly LoRA on Mac. I committed to (1) for the portfolio narrative because that's what enterprises actually run.
- **Paraphrase synthesis silently changed `clause_type` ~3% of the time.** The teacher (GPT-4o-mini) occasionally rewrote a non-solicit as a non-compete because the paraphrase emphasized different language. Added an explicit clause_type-stability check in `synthesize.py` that drops any paraphrase whose label's `clause_type` differs from the source. Lost ~3% throughput; gained data hygiene.
- **JSON validity was a poor primary metric for refusal cases.** Pre-tuning Qwen sometimes refused with a polite-but-prose response, which is a legitimate refusal but failed `json_validity` and got counted as a hard error. Restructured the metric: refusal-correctness is now a distinct axis from JSON validity, so a model that returns prose-but-correct on out-of-scope clauses doesn't get penalized twice.

---

This file gets a new section every time something non-obvious happens. The Phase-5 section will grow once real training runs are in.
