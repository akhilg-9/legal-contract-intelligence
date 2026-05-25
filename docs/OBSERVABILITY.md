# Observability (Phase 4)

Every `lci ask` invocation produces one Langfuse **trace**, which you can drill into to see exactly what retrieval surfaced, what prompt was sent, and what the model said back.

## Bring up Langfuse

```bash
docker-compose up -d langfuse-db langfuse
open http://localhost:3000
```

First-time setup in the UI:
1. Create an account (it's local).
2. Create a project named `legal-contract-intelligence`.
3. Open **Settings → API Keys**, generate a key pair, copy them into `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_HOST=http://localhost:3000
```

Run `lci ask "..."` — within a few seconds the trace appears at http://localhost:3000/project/<id>/traces.

When the env keys are empty, instrumentation is a no-op. Nothing in the codepath changes; you don't need to touch the pipeline to disable tracing.

## What gets traced

```
rag.ask  (parent trace)
├── tags: prompt:v2  retrieval:reranked  model:ollama:llama3.2:3b
├── span: dense_search           ── top candidate_k by cosine
├── span: bm25_search            ── top candidate_k by BM25
├── span: reciprocal_rank_fusion ── fused candidate set
├── span: cross_encoder_rerank   ── final top_k after bge-reranker-v2-m3
├── span: ChatPromptTemplate     ── final messages sent
├── generation: ChatOllama       ── raw model response
└── output: { answer, citations, insufficient_context, retrieved_chunks }
```

LangChain spans (`dense_search` through `ChatOllama`) come for free via the auto-callback. The parent trace, output, and tags are emitted explicitly by `observability.trace_ask` / `finalize_trace`.

## Dashboards to build

The Langfuse UI ships with traces / generations / scores views out of the box. The ones worth pinning for this project:

1. **Latency** — P50/P95 of `rag.ask` over rolling 24h. Slice by `tag:retrieval:*` to compare modes.
2. **Cost per request** — derived from `metadata.estimated_usd` (set by the pipeline using the rates in `.env`). Defaults are zero because self-hosted Ollama on Apple Silicon is effectively free; for a cloud-GPU deployment, set `LCI_COST_PER_M_INPUT_TOKENS` and `LCI_COST_PER_M_OUTPUT_TOKENS` to your actual GPU-hour / token economics.
3. **Failure rate** — traces tagged with any of `insufficient_context`, `no_citations`, `retrieval_empty`. Express as a percentage of total `rag.ask` traces.
4. **Citation coverage** — distribution of `output.citations` length. Anything outside [1, 6] is interesting (0 = unguarded answer that the post-processor caught; 6+ = retriever returned redundant chunks).

## Root-cause walkthrough

The goal of this section is the literal click-path for going from a metric anomaly to the trace that caused it. Steps:

1. **Spot the anomaly.** Failure-rate dashboard ticks up from 4% to 12% over a 1-hour window.
2. **Drill into traces with failure tags.** In the Traces view, filter `tag:insufficient_context OR tag:no_citations`, narrow to the same hour.
3. **Sort by latency descending.** Failure cases with abnormally low latency usually mean retrieval returned nothing; failure cases with abnormally high latency mean the LLM hallucinated and got post-process-stripped after a full generation.
4. **Open a trace.** Look at the `cross_encoder_rerank` span output — were the top chunks even on-topic? If not, the failure is upstream of the LLM (retrieval / chunking).
5. **Compare to a baseline trace.** Open a known-good trace with the same prompt version and look at retrieved chunks side by side. The vocabulary mismatch between query and runbook-style language is the most common culprit; it shows up as a BM25 score gap.
6. **File the fix.** Three classes of fix:
   - **Add a synonym to the runbook** (queries like "data leak" don't surface a doc that says only "exfiltration").
   - **Bump the candidate_k** (the right chunk was hit by BM25 at rank 22 but our candidate window was 20).
   - **Tighten the system prompt** (the model is rephrasing instead of quoting).

Phase 5's fine-tuned model will reduce the third class structurally.

## Cost notes for self-hosted

For an honest portfolio metric, fill in `.env` with real numbers:

| Setup | Rough $/M input tokens | Rough $/M output tokens |
| --- | --- | --- |
| Apple Silicon M3 local (Ollama, 3B) | ~$0 | ~$0 |
| A10 cloud (RunPod, vLLM, 8B Q5) | ~$0.10–$0.15 | ~$0.15–$0.25 |
| H100 cloud (RunPod, vLLM, 70B AWQ) | ~$0.50–$1.00 | ~$1.50–$3.00 |

These reflect amortized GPU-hour cost ÷ tokens served. The estimator is intentionally simple — production-grade cost attribution is in Phase 5+.
