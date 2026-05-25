# Legal Contract Intelligence Platform

[![CI](https://github.com/akhilg-9/legal-contract-intelligence/actions/workflows/eval.yml/badge.svg)](https://github.com/akhilg-9/legal-contract-intelligence/actions/workflows/eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1c3c3c.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Code style: open source](https://img.shields.io/badge/stack-open--source-brightgreen.svg)](#stack-at-a-glance)

> A RAG platform for legal contracts, built **open-source end-to-end** — no proprietary APIs required to run inference. Built phase by phase to show the production loop in one cohesive repo: **ingest → retrieve → rerank → generate → evaluate → observe → fine-tune**.

**Jump to:** [Phases](#phases) · [Architecture](#architecture) · [Stack](#stack-at-a-glance) · [Quick start](#quick-start-apple-silicon-mac) · [What it looks like](#what-it-looks-like-running) · [CLI](#cli) · [Eval](#evaluation) · [Observability](#observability)

---

## Phases

| # | Goal | Highlights | Status |
| :-: | :-- | :-- | :-: |
| 1 | **RAG foundation** | PDF/HTML ingestion · `bge-large-en-v1.5` · Qdrant · versioned prompts · `[chunk_id]` citations · refusal on weak context | ✅ |
| 2 | **Production retrieval** | BM25 + dense fused via RRF · `bge-reranker-v2-m3` cross-encoder · `prompts/v2.yaml` | ✅ |
| 3 | **Eval + CI** | 25-example golden set incl. refusals · Ragas (4 metrics) + structural metrics · GH Actions smoke-on-PR + full-nightly with CI gating | ✅ |
| 4 | **Observability** | Self-hosted Langfuse · LCEL auto-tracing + parent `rag.ask` trace · searchable failure tags · cost estimator | ✅ |
| 5 | **Fine-tuning** | 25 seeds → 3k synthesized · QLoRA on Qwen-3 8B via TRL **or** Axolotl · deterministic eval · training curves | ✅ scaffolded\* |

\* Phase 5 code is complete and runnable; an actual training run needs a CUDA GPU (RunPod A100 ≈ $1.50–2.00/h). Phases 1–4 run on Apple Silicon (MPS).

Honest accounting → [`WHAT_BROKE.md`](./WHAT_BROKE.md). Measurements → [`BENCHMARKS.md`](./BENCHMARKS.md).

---

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
user → │  retrieve  │  →   │   rerank     │  →   │ LCEL chain + LLM    │  → answer + [chunk_id]
       │ BM25+dense │      │  bge-v2-m3   │      │ Ollama / OpenAI     │     citations
       │  via RRF   │      │  cross-enc   │      └──────────┬──────────┘
       └────────────┘      └──────────────┘                 │
              ▲                                              ▼
              │                                       ┌──────────────┐
              │                                       │   Langfuse   │  trace, tags,
              │                                       │  (self-host) │  cost, failures
              │                                       └──────────────┘
              │
       ┌──────┴──────┐         ┌───────────────────┐
       │ Ragas +     │  ◄────  │ evals/golden_set  │
       │ structural  │         │ (25 Q&A · 2 refs) │
       └─────────────┘         └───────────────────┘
              │
              ▼ CI gate
       prompts/v*.yaml  →  bump = full eval rerun
```

---

## Stack at a glance

| Layer | Choice | Why |
| :-- | :-- | :-- |
| Orchestration | **LangChain** LCEL + **LangGraph** (queued for self-correcting flows) | Industry-standard glue without locking the chain into a single provider |
| Embeddings | **`BAAI/bge-large-en-v1.5`** on MPS via sentence-transformers | Top-tier open-weights retrieval; 1024-d; runs locally on Apple Silicon |
| Vector store | **Qdrant** self-hosted (docker-compose) | Open-source, fast cosine, no vendor lock-in |
| Lexical retrieval | **`rank_bm25`** | Catches keyword matches dense embeddings miss (e.g., specific dollar amounts) |
| Reranker | **`BAAI/bge-reranker-v2-m3`** cross-encoder | Eliminates "topical but wrong" hits — biggest precision lift in the stack |
| LLM (default) | **Ollama `llama3.2:3b`** | Open weights, runs on a laptop, no API cost |
| LLM (alt route) | OpenAI `gpt-4o-mini` via `--provider openai` | Comparable benchmarks, used as the Ragas judge in CI |
| Eval | **Ragas** (faithfulness, relevancy, precision, recall) + structural metrics | Pairs LLM-as-judge with deterministic checks for honest signal |
| CI | **GitHub Actions** with Qdrant service + auto-gating | Smoke on PRs, full nightly run |
| Observability | **Langfuse** self-hosted (postgres + server) | Trace every span, tag failures, derive cost — all locally |
| Fine-tuning | **TRL** SFTTrainer + **PEFT** LoRA + **bitsandbytes** 4-bit | Standard QLoRA recipe; Axolotl config provided as alternative |
| Base model | **Qwen-3 8B** (Qwen2.5-7B-Instruct today) | Apache 2.0, strong at structured generation |

---

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

# 4. Config
cp .env.example .env

# 5. Ingest the committed sample contracts
lci ingest data/sample/

# 6. Ask
lci ask "What is the term of the Acme-Betacorp NDA?"
lci ask "What is the SaaS uptime SLA and its credit tiers?"

# 7. (Optional) Langfuse for tracing — UI at http://localhost:3000
docker-compose up -d langfuse-db langfuse
# Create API keys in the UI → paste into .env (LANGFUSE_*)
```

> See [`docs/OBSERVABILITY.md`](./docs/OBSERVABILITY.md) for the Langfuse setup walkthrough and [`docs/FINETUNING.md`](./docs/FINETUNING.md) for the Phase-5 cloud-GPU recipe.

---

## What it looks like running

```text
$ lci ask "Who owns IP created by an independent contractor under the Zephyr agreement?"

╭─────────────────────────────────────── Answer  (prompts: v2) ───────────────────────────────────────╮
│ All Work Product created by Contractor in performing the Services is deemed a work made for hire   │
│ and is "the sole and exclusive property of Company" [independent_contractor_zephyr#0008-...].      │
│ To the extent any Work Product does not qualify as a work made for hire, the Contractor            │
│ "hereby irrevocably assigns to Company all right, title, and interest" including all IP rights     │
│ [independent_contractor_zephyr#0008-...]. The Contractor must also "execute all documents          │
│ reasonably necessary to perfect Company's ownership" [independent_contractor_zephyr#0008-...].     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯

                                Retrieved chunks (top 6)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ chunk_id                          ┃  score ┃ doc / page             ┃ preview                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ independent_contractor_zephyr#…   │  9.412 │ independent_contractor │ All deliverables, work …    │
│ independent_contractor_zephyr#…   │  4.891 │ independent_contractor │ Contractor shall indemnify… │
│ saas_subscription_orbit#0007-…    │  2.103 │ saas_subscription_orbit│ As between the Parties, …   │
│ …                                                                                                   │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
cited: independent_contractor_zephyr#0008-a91f3c2d8b
```

A correctly-refusing example:

```text
$ lci ask "What is the indemnification cap in the Acme-Betacorp NDA?"

╭─────────────────────────────────────── Answer  (prompts: v2) ───────────────────────────────────────╮
│ INSUFFICIENT_CONTEXT: the Acme-Betacorp NDA does not contain an indemnification clause or any cap  │
│ on indemnification.                                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯
flagged: insufficient context
```

---

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

Runtime overrides:

```bash
lci ask --provider openai "..."   # use OpenAI route instead of Ollama
lci ask --prompt v1 "..."         # use a specific prompt config version
```

---

## Versioned prompt configs

Every prompt + retrieval setting lives in `prompts/v*.yaml`. Bumping the version is the **unit of change** that the CI eval gates on. A PR that modifies any `prompts/v*.yaml` triggers the smoke eval automatically; merge is blocked if faithfulness or refusal accuracy drops below the thresholds.

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

---

## Citation enforcement

Two layers, both required to pass:

1. **Prompt** — the system message requires `[chunk_id]` inline citations and a literal `INSUFFICIENT_CONTEXT: <reason>` refusal.
2. **Post-processing** — `pipeline.py` parses the answer for citations against the actual retrieved chunk-ids; an uncited "confident" answer is suppressed and replaced with `INSUFFICIENT_CONTEXT: model produced an uncited answer`.

This gives Ragas faithfulness something concrete to grade and surfaces hallucinations day-one rather than after the fact.

---

## Evaluation

[`evals/golden_set.jsonl`](./evals/golden_set.jsonl) — 25 hand-crafted Q&A pairs over the four sample contracts, including 2 deliberate **refusal cases** that test whether the system correctly returns `INSUFFICIENT_CONTEXT` when no excerpt supports an answer.

**Metrics:**

| Metric | Source | What it catches |
| :-- | :-- | :-- |
| `faithfulness` | Ragas | claims in the answer that are entailed by retrieved chunks |
| `answer_relevancy` | Ragas | does the answer actually address the question |
| `context_precision` | Ragas | fraction of retrieved chunks that are relevant |
| `context_recall` | Ragas | fraction of the ground-truth covered by retrieved chunks |
| `expected_doc_recall@k` | structural | at least one chunk from each expected doc made it into top-k |
| `refusal_accuracy` | structural | for refusal cases, did the system output `INSUFFICIENT_CONTEXT` |

The full eval runs nightly on a cron in [`.github/workflows/eval.yml`](./.github/workflows/eval.yml); a smoke subset runs on every PR that touches prompts, code, or evals.

---

## Observability

`lci ask` invocations produce one Langfuse trace per question with auto-traced LCEL spans:

```
rag.ask  (parent trace, tags: prompt:v2 · retrieval:reranked · model:ollama:llama3.2:3b)
├── span: dense_search           ── top candidate_k by cosine
├── span: bm25_search            ── top candidate_k by BM25
├── span: reciprocal_rank_fusion ── fused candidate set
├── span: cross_encoder_rerank   ── final top_k after bge-reranker-v2-m3
├── span: ChatPromptTemplate     ── final messages sent
├── generation: ChatOllama       ── raw model response
└── output: { answer, citations, insufficient_context, retrieved_chunks }
```

Failure tags (`insufficient_context`, `no_citations`, `retrieval_empty`) are first-class so the failure-rate dashboard is one filter away. Full root-cause click-path in [`docs/OBSERVABILITY.md`](./docs/OBSERVABILITY.md).

When `LANGFUSE_PUBLIC_KEY` is empty in `.env`, instrumentation is a no-op — nothing changes in the codepath.

---

## Repository layout

```
legal-contract-intelligence/
├── prompts/
│   ├── v1.yaml                # baseline (dense, llama3.2:3b)
│   └── v2.yaml                # reranked, tighter citation rules
├── src/legal_contract_intelligence/
│   ├── config.py              # env-loaded settings
│   ├── prompts.py             # YAML loader (pydantic-validated)
│   ├── ingestion/             # parsers + tiktoken chunking + EDGAR
│   ├── embeddings.py          # bge-large-en-v1.5 + BGE query prefix
│   ├── vectorstore.py         # Qdrant via langchain-qdrant
│   ├── retrieval.py           # dense / sparse / hybrid (RRF) / reranked
│   ├── llm.py                 # Ollama / OpenAI provider abstraction
│   ├── pipeline.py            # LCEL chain + citation enforcement + tracing
│   ├── observability.py       # Langfuse + cost estimator
│   ├── eval.py                # Ragas + structural + CI gates
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

---

## Configuration

| Setting | Default | Notes |
| :-- | :-- | :-- |
| `LCI_LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `LCI_OLLAMA_MODEL` | `llama3.2:3b` | Try `qwen2.5:7b-instruct` for higher quality |
| `LCI_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | 1024-dim, English |
| `LCI_EMBEDDING_DEVICE` | `mps` | `mps` / `cuda` / `cpu` |
| `LCI_QDRANT_URL` | `http://localhost:6333` | self-hosted via docker-compose |
| `LANGFUSE_PUBLIC_KEY` | _(empty)_ | empty disables tracing entirely |
| `LCI_COST_PER_M_INPUT_TOKENS` | `0` | populate for cloud-GPU runs |

---

## What's intentionally NOT in this repo

- **No vector-DB lock-in.** Self-hosted Qdrant; swap by changing one URL.
- **No agent framework on the critical path.** LangChain LCEL handles the chain. LangGraph is in deps, ready for the next phase (self-correcting retrieval, multi-step verification).
- **No fine-tuning of the embedder.** BGE-large is strong enough at this scale; embedder tuning is the right move at corpus sizes ≥ ~50k chunks.
- **No write tools.** This is a read/analyze system, not a contract-drafting one.
- **No multi-tenancy / RBAC.** Single-tenant by design; production deployment would layer those on.

---

## Contributing & feedback

Issues and PRs welcome. The CI gate on `prompts/v*.yaml` and `src/**` is your safety net — open a PR and the smoke eval will tell you within ~5 minutes whether the change regressed faithfulness or refusal accuracy.

For bigger conversations (architecture pushback, alternative stack choices), open an issue and tag it `discussion`.

---

## License

[MIT](./LICENSE). Built by [@akhilg-9](https://github.com/akhilg-9).
