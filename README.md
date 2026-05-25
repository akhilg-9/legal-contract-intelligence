# Legal Contract Intelligence Platform

A LangChain/LangGraph-orchestrated RAG platform for legal contracts. **Open-source end-to-end** — no proprietary APIs required to run inference. Built phase by phase to show the production loop (ingest → retrieve → generate → evaluate → observe → fine-tune) in one cohesive repo.

| Phase | Goal | Status |
| --- | --- | --- |
| **1 — RAG foundation** | PDF/HTML ingestion, bge-large-en-v1.5 embeddings, Qdrant, top-k retrieval, versioned prompts, inline `[chunk_id]` citations, refusal on weak context | ✅ |
| **2 — Production quality** | Hybrid retrieval (BM25 + dense via RRF) + `bge-reranker-v2-m3` cross-encoder, tightened citation discipline, `prompts/v2.yaml` | ✅ |
| **3 — Eval + CI** | 25-example golden set incl. refusal cases, Ragas (faithfulness / answer-relevancy / context-precision / context-recall), structural metrics, GitHub Actions smoke-on-PR + full-nightly with CI gating | ✅ |
| **4 — Observability** | Self-hosted Langfuse (postgres + server), LCEL auto-tracing + parent `rag.ask` trace with searchable failure tags, configurable cost estimator, root-cause walkthrough guide | ✅ |
| **5 — Fine-tuning** | 25 hand-crafted seeds → 3k synthesized examples (OpenAI as one-time labeler), QLoRA on Qwen-3 8B via TRL or Axolotl, deterministic eval (JSON validity / EM / refusal), training-curve plots | ✅ scaffolded* |

\* Phase 5 training code is complete and runnable; an actual training run needs a CUDA GPU (RunPod A100 ≈ $1.50–2.00/h). All other phases run on Apple Silicon.

Honest accounting in [`WHAT_BROKE.md`](./WHAT_BROKE.md). Measurements in [`BENCHMARKS.md`](./BENCHMARKS.md).

## Architecture

```
┌────────────┐    ┌──────────────────┐    ┌────────────────────────────┐
│  PDF/HTML  │ →  │  parse + chunk   │ →  │ bge-large-en-v1.5 (MPS)    │
└────────────┘    │  (650/100 tok)   │    │  +  Qdrant cosine index    │
                  └──────────────────┘    └─────────────┬──────────────┘
                                                        │
              ┌─────────────────────────────────────────┘
              ▼
       ┌────────────┐      ┌──────────────┐      ┌─────────────────────┐
user → │  retrieve  │  →   │   rerank     │  →   │ LCEL chain + LLM    │  → answer + [chunk_id]…
       │  BM25 + ❘  │      │  bge-v2-m3   │      │ Ollama / OpenAI     │     citations
       │  dense  RRF│      │  cross-enc   │      └──────────┬──────────┘
       └────────────┘      └──────────────┘                 │
              ▲                                              ▼
              │                                       ┌──────────────┐
              │                                       │   Langfuse   │ trace, tags,
              │                                       │  (self-host) │ cost, failures
              │                                       └──────────────┘
              │
       ┌──────┴───────┐         ┌────────────────────┐
       │ Ragas + struct.│  ◄─── │ evals/golden_set    │
       │  eval          │       │ (25 Q&A, 2 refusals)│
       └────────────────┘       └────────────────────┘
              │
              ▼ CI gate
       prompts/v*.yaml → bump = full eval rerun
```

## Quick start (Apple Silicon Mac)

```bash
git clone https://github.com/akhilg-9/legal-contract-intelligence.git
cd legal-contract-intelligence

# 1. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Qdrant
docker-compose up -d qdrant

# 3. Ollama
brew install ollama
ollama serve &
ollama pull llama3.2:3b

# 4. (Optional) Langfuse for tracing
docker-compose up -d langfuse-db langfuse   # UI at http://localhost:3000
# Create API keys in the UI and paste them into .env (LANGFUSE_*)

# 5. Config
cp .env.example .env

# 6. Ingest the committed sample contracts
lci ingest data/sample/

# 7. Ask
lci ask "What is the term of the Acme-Betacorp NDA?"
lci ask "What is the SaaS uptime SLA and its credit tiers?"

# 8. Run the eval suite (Ragas needs OPENAI_API_KEY as the judge; skip with --skip-ragas)
lci-eval --skip-ragas              # local-only, structural metrics
export OPENAI_API_KEY=sk-...
lci-eval                            # full eval incl. Ragas
```

## CLI

```text
lci ingest <path>           Parse + chunk + embed + upsert (PDF / HTML / txt / md)
lci ask "<question>"        Full RAG pipeline; prints answer + retrieved chunks
lci search "<query>"        Retrieval-only; --mode dense|sparse|hybrid|reranked
lci info                    Qdrant collection stats
lci fetch-samples           Pull real public SEC contract exhibits into data/sample/

lci-eval                    Run the golden-set eval against the active prompts/
lci-eval --smoke            CI-grade smoke run (3 items)
lci-eval --skip-ragas       Skip LLM-as-judge metrics (offline)
```

Provider override at runtime: `lci ask --provider openai "..."`.
Prompt-config override: `lci ask --prompt v1 "..."`.

## Repository layout

```
legal-contract-intelligence/
├── prompts/
│   ├── v1.yaml                # baseline (dense, llama3.2:3b)
│   └── v2.yaml                # reranked, tighter citation rules
├── src/legal_contract_intelligence/
│   ├── config.py              # env-loaded settings
│   ├── prompts.py             # YAML loader (pydantic-validated)
│   ├── ingestion/             # parsers (PDF/HTML/txt) + tiktoken chunking
│   │   ├── parsers.py
│   │   ├── chunking.py
│   │   └── edgar.py
│   ├── embeddings.py          # bge-large-en-v1.5 + BGE query prefix
│   ├── vectorstore.py         # Qdrant via langchain-qdrant
│   ├── retrieval.py           # dense / sparse / hybrid (RRF) / reranked
│   ├── llm.py                 # Ollama / OpenAI provider abstraction
│   ├── pipeline.py            # LCEL chain + citation enforcement + Langfuse trace
│   ├── observability.py       # Langfuse instrumentation + cost estimator
│   ├── eval.py                # Ragas + structural metrics + CI gates
│   ├── cli.py                 # `lci` typer CLI
│   └── finetune/
│       ├── schema.py          # ClauseLabel pydantic schema
│       ├── synthesize.py      # OpenAI paraphraser, 25 → 3k examples
│       ├── train.py           # TRL SFTTrainer + PEFT LoRA + 4-bit nf4
│       ├── evaluate.py        # JSON validity / EM / Jaccard / refusal
│       └── plot.py            # training curves from trainer_state.json
├── configs/
│   └── axolotl_qwen_qlora.yaml  # Axolotl equivalent of train.py
├── evals/
│   ├── golden_set.jsonl       # 25 Q&A (incl. 2 refusal cases)
│   └── README.md
├── data/
│   ├── sample/                # 4 realistic synthetic contracts
│   └── finetune/seed/         # 25 hand-crafted (clause, label) seeds
├── docs/
│   ├── OBSERVABILITY.md       # Langfuse setup + root-cause click-path
│   └── FINETUNING.md          # end-to-end QLoRA recipe
├── .github/workflows/eval.yml # smoke-on-PR + full-nightly
├── docker-compose.yml         # qdrant + langfuse stack
├── BENCHMARKS.md
├── WHAT_BROKE.md
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Versioned prompt configs

Every prompt + retrieval setting lives in `prompts/v*.yaml`. Bumping the version is the **unit of change** the CI eval gates on. A PR that modifies any `prompts/v*.yaml` triggers the smoke eval automatically; merge is blocked if faithfulness or refusal accuracy drops below the thresholds at the top of `.github/workflows/eval.yml`.

```yaml
# excerpt — prompts/v2.yaml
version: v2
model:
  provider: ollama
  name: llama3.2:3b
retrieval:
  mode: reranked          # dense | sparse | hybrid | reranked
  top_k: 6
  candidate_k: 20
templates:
  system: |
    You are a senior contracts analyst. Rules:
    1. Every claim must cite [chunk_id] inline.
    2. Quote verbatim when asked for wording.
    3. Refuse with INSUFFICIENT_CONTEXT when retrieval is inadequate.
    ...
```

## Citation enforcement

Two layers, both required to pass:

1. **Prompt** — the system message requires `[chunk_id]` inline citations and a literal `INSUFFICIENT_CONTEXT: <reason>` refusal.
2. **Post-processing** — `pipeline.py` parses the answer for citations against the actual retrieved chunk-ids; an uncited "confident" answer is suppressed and replaced with `INSUFFICIENT_CONTEXT: model produced an uncited answer`.

This gives Ragas faithfulness something concrete to grade and surfaces hallucinations day-one rather than after the fact.

## Configuration matrix

| Setting | Default | Notes |
| --- | --- | --- |
| `LCI_LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `LCI_OLLAMA_MODEL` | `llama3.2:3b` | Try `qwen2.5:7b-instruct` for higher quality |
| `LCI_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | 1024-dim, English |
| `LCI_EMBEDDING_DEVICE` | `mps` | `mps` / `cuda` / `cpu` |
| `LCI_QDRANT_URL` | `http://localhost:6333` | self-hosted via docker-compose |
| `LANGFUSE_PUBLIC_KEY` | _(empty)_ | leave empty to disable tracing |
| `LCI_COST_PER_M_INPUT_TOKENS` | `0` | populate for cloud-GPU runs |

## What's intentionally NOT in this repo

- **No vector DB lock-in.** Self-hosted Qdrant; swap by changing one URL.
- **No agent framework.** LangChain LCEL for the chain; LangGraph in the dependency list and ready for the next phase (self-correcting retrieval / multi-step verification flows).
- **No fine-tuning of the embedder.** BGE-large is strong enough at this scale; embedder fine-tuning is the right move at corpus sizes ≥ ~50k chunks.
- **No write tools.** This is a read/analyze system, not a contract-drafting one.

## License

[MIT](./LICENSE).
