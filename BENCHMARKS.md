# BENCHMARKS.md

All measurements live here. Numbers come from real runs; placeholders are marked `TBD` and updated when the corresponding run happens.

## Phase 1 — RAG foundation

| Workload | Hardware | Number |
| --- | --- | --- |
| `bge-large-en-v1.5` encode (batch 32, 256 tok) | M3 MPS | TBD |
| Ingestion: 4 sample contracts → chunks | M3 MPS | TBD |
| Qdrant cold-start memory | docker | ~250 MB RSS |
| End-to-end `lci ask` (top_k=6, dense, llama3.2:3b) | M3 MPS + Ollama | TBD |

## Phase 2 — Hybrid + reranker

| Workload | Hardware | Number |
| --- | --- | --- |
| BM25 corpus build (from Qdrant scroll, 200 chunks) | M3 MPS | TBD |
| `bge-reranker-v2-m3` rerank, top-20 → top-6 | M3 MPS | TBD |
| End-to-end `lci ask` (mode=reranked) | M3 MPS + Ollama | TBD |
| End-to-end `lci ask` (mode=dense) | M3 MPS + Ollama | TBD |

The retrieval-mode delta is the headline Phase-2 result: BM25+dense+rerank vs dense-only on the same 25 golden questions.

## Phase 3 — Eval (golden set, 25 items)

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Expected Doc Recall@k | Refusal Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `prompts/v1.yaml` (dense only, llama3.2:3b) | TBD | TBD | TBD | TBD | TBD | TBD |
| `prompts/v2.yaml` (reranked, llama3.2:3b) | TBD | TBD | TBD | TBD | TBD | TBD |
| `prompts/v2.yaml` (reranked, gpt-4o-mini) | TBD | TBD | TBD | TBD | TBD | TBD |

CI gate thresholds (full eval): faithfulness ≥ 0.70, refusal_accuracy ≥ 0.80.

## Phase 4 — Observability

| Metric | Value |
| --- | --- |
| Tracing overhead per ask (Langfuse callback enabled) | TBD ms |
| Self-hosted Langfuse memory (server + postgres) | TBD MB |
| Cost per ask (Ollama llama3.2:3b on M3) | ~$0 |

## Phase 5 — Fine-tuning

### Data

| Stage | Count |
| --- | --- |
| Seed examples (hand-crafted) | 25 |
| Synthesized via OpenAI paraphraser | TBD |
| After clause_type-stability filter | TBD |
| Train set | TBD |
| Held-out eval | TBD |

### Training

| Setting | Value |
| --- | --- |
| Base model | Qwen2.5-7B-Instruct (substitute Qwen-3 8B when available) |
| Method | QLoRA (4-bit nf4 + LoRA r=16 α=32) |
| GPU | TBD (single A100 80GB target) |
| Epochs | 2 |
| LR | 2e-4 |
| Train tokens | TBD |
| Wall-clock | TBD |

### Eval (held-out, 200)

| Metric | Pre-tune (base) | Post-tune (LoRA merged) |
| --- | --- | --- |
| json_validity | TBD | TBD |
| clause_type_exact | TBD | TBD |
| obligations_jaccard | TBD | TBD |
| refusal_correctness | TBD | TBD |

Per-clause-type accuracy table will be expanded once the post-tune eval lands.
