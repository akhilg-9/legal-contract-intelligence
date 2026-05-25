# Legal Contract Intelligence Platform

A LangChain/LangGraph-orchestrated RAG platform for legal contracts. Built **open-source end-to-end** — no proprietary APIs required to run it locally.

This repo is being built phase-by-phase. **Phase 1 (you are here)** is the RAG foundation: ingest contracts, embed with `BAAI/bge-large-en-v1.5`, store in self-hosted Qdrant, retrieve with citations, generate grounded answers, with strict citation enforcement.

## Roadmap

| Phase | Goal | Status |
| --- | --- | --- |
| **1 — RAG foundation** | PDF/HTML ingestion, BGE embeddings, Qdrant, top-k retrieval, versioned prompts in `prompts/v1.yaml`, inline `[chunk_id]` citations, refusal on weak context | ✅ this commit |
| 2 — Production quality | Hybrid retrieval (BM25 + vector, RRF fusion) + cross-encoder reranker (`bge-reranker-v2-m3`) | next |
| 3 — Eval + CI | Golden eval set (100 Q&A), Ragas faithfulness/relevance/precision, GitHub Actions gating prompt-config changes | |
| 4 — Observability | Self-hosted Langfuse, per-request traces, latency / cost / failure-rate dashboards | |
| 5 — Fine-tuning | QLoRA on Qwen-3 8B via TRL/Axolotl for clause → JSON extraction, training curves, WHAT_BROKE.md | |

## Phase 1 architecture

```
┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────┐
│  PDF / HTML │ →  │  parse + chunk  │ →  │ bge-large-en-v1.5 (MPS)  │
└─────────────┘    │  (token-aware,  │    │  +  Qdrant cosine index  │
                   │   ~650 tok,     │    └────────────┬─────────────┘
                   │   100 overlap)  │                 │
                   └─────────────────┘                 ▼
                                              ┌────────────────────┐
                user question  ─────────────► │ retrieve top-k     │
                                              │  (LangChain LCEL)  │
                                              └──────────┬─────────┘
                                                         ▼
                                              ┌────────────────────┐
                                              │ Ollama llama3.2:3b │
                                              │ (or OpenAI route)  │
                                              └──────────┬─────────┘
                                                         ▼
                            answer with inline [chunk_id] citations,
                            OR  "INSUFFICIENT_CONTEXT: ..." refusal
```

**Why this architecture:**
- `BAAI/bge-large-en-v1.5` is state-of-the-art open-weights retrieval; runs on Apple Silicon MPS in seconds per batch.
- Qdrant is self-hosted (Docker), no vendor lock-in, no API cost.
- LangChain provides the runtime glue (LCEL chains, Qdrant integration, Ollama/OpenAI providers) without locking us into any one provider.
- LangGraph is on the dependency list, waiting for Phase 3 — vanilla RAG doesn't need a graph.
- Prompts live in `prompts/v*.yaml`. Bumping the version is the **unit of change** that the Phase 3 Ragas eval gates on.

## Quick start (Apple Silicon Mac)

```bash
git clone https://github.com/akhilg-9/legal-contract-intelligence.git
cd legal-contract-intelligence

# 1. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Qdrant (self-hosted)
docker-compose up -d qdrant

# 3. Ollama (open-weights LLM)
brew install ollama        # or: https://ollama.com/download
ollama serve &              # in another tab, or as a service
ollama pull llama3.2:3b

# 4. Config
cp .env.example .env
# (defaults are fine; .env lets you swap models / providers without code changes)

# 5. Ingest the sample contracts (committed in data/sample/)
lci ingest data/sample/

# 6. Ask away
lci ask "What is the term of the Acme-Betacorp NDA, and how long do confidentiality obligations survive termination?"
lci ask "What is the SaaS service-level agreement uptime target and what are the credit tiers?"
lci ask "Who owns IP created by an independent contractor under the Zephyr agreement?"
```

You should see an answer with inline `[chunk_id]` citations and a table of the retrieved chunks with their cosine scores.

## CLI

```text
lci ingest <path>         Parse + chunk + embed + upsert (file or directory)
lci ask "<question>"      Full RAG pipeline; prints answer + retrieved chunks
lci search "<query>"      Retrieval-only; inspect what the retriever surfaces
lci info                  Qdrant collection stats
lci fetch-samples         Pull real public SEC contract exhibits into data/sample/
```

`lci ask` accepts `--provider openai` to switch to a frontier route at runtime, and `--prompt v2` to test an alternate prompt-config file once you've created one.

## Citation enforcement

The system refuses to answer when retrieval can't support it. Two enforcement layers:

1. **Prompt:** the system message requires inline `[chunk_id]` citations and a literal `INSUFFICIENT_CONTEXT: <reason>` refusal when retrieved excerpts are inadequate.
2. **Post-processing:** answers are parsed for citations against the actual retrieved chunk-ids. An uncited answer is treated as a citation failure and the raw model output is suppressed.

This gives Phase 3's Ragas faithfulness metric something concrete to grade against, and it surfaces hallucinations on day one rather than after the fact.

## Repository layout

```
legal-contract-intelligence/
├── prompts/
│   └── v1.yaml                # versioned prompt config — model, retrieval, templates
├── src/legal_contract_intelligence/
│   ├── config.py              # env-loaded settings
│   ├── prompts.py             # YAML loader (pydantic-validated)
│   ├── ingestion/
│   │   ├── parsers.py         # PDF + HTML + plain-text
│   │   ├── chunking.py        # tiktoken-based, paragraph-preserving
│   │   └── edgar.py           # SEC EDGAR fetcher (rate-limit-aware)
│   ├── embeddings.py          # bge-large-en-v1.5 on MPS w/ BGE query prefix
│   ├── vectorstore.py         # Qdrant via langchain-qdrant
│   ├── llm.py                 # Ollama / OpenAI provider abstraction
│   ├── pipeline.py            # LCEL chain + citation enforcement
│   └── cli.py                 # typer CLI
├── data/sample/               # committed sample contracts (realistic, synthetic)
├── docker-compose.yml         # qdrant (langfuse added in Phase 4)
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Configuration

Everything is configured by `.env` (see `.env.example`) and the active prompt-config YAML in `prompts/`. Swapping the LLM provider, model, retrieval `top_k`, or score threshold does **not** require a code change.

| Setting | Default | Notes |
| --- | --- | --- |
| `LCI_LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `LCI_OLLAMA_MODEL` | `llama3.2:3b` | Try `qwen2.5:7b-instruct` or `mistral:7b` for higher quality |
| `LCI_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | 1024-dim, English |
| `LCI_EMBEDDING_DEVICE` | `mps` | `mps` (Apple Silicon), `cuda`, or `cpu` |
| `LCI_QDRANT_URL` | `http://localhost:6333` | self-hosted via docker-compose |
| `LCI_QDRANT_COLLECTION` | `contracts` | one collection per dataset |

## What's intentionally NOT in Phase 1

- **Reranker** — `bge-reranker-v2-m3` arrives in Phase 2 along with hybrid retrieval.
- **Ragas / eval CI** — Phase 3.
- **Langfuse / observability** — Phase 4.
- **Fine-tuning** — Phase 5; clause-extraction QLoRA on Qwen-3 8B.

Each phase will ship a self-contained PR + a `BENCHMARKS.md` entry with real measurements, and a `WHAT_BROKE.md` entry documenting iteration.

## License

[MIT](./LICENSE).
