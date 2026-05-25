# Eval set

`golden_set.jsonl` — 25 hand-crafted question/ground-truth pairs over the four sample contracts, including two **refusal** cases that test whether the system correctly returns `INSUFFICIENT_CONTEXT` when no excerpt supports an answer.

**Honest scope note:** the plan targets 100 manually-verified Q&A pairs. 25 is the right number to start: each one is hand-written against the actual sample contracts and tagged with the expected source document plus clause tags. Scaling to 100 is the right move *after* you ingest a larger corpus (Phase 3 expansion).

## Schema

```jsonc
{
  "id": "g001",
  "question": "<natural-language question>",
  "ground_truth": "<the canonical answer, paraphrased from the actual contract text>",
  "expected_docs": ["<doc_id>", ...],   // ingested doc_ids that should be retrieved; [] for refusal cases
  "clause_tags": ["term", "survival"]   // freeform tags for slicing metrics later
}
```

## Metrics (Ragas)

The eval runner computes:

- `faithfulness` — claims in the answer that are entailed by retrieved chunks.
- `answer_relevancy` — does the answer actually address the question.
- `context_precision` — fraction of retrieved chunks that are relevant.
- `context_recall` — fraction of the ground-truth that is covered by retrieved chunks.

Plus two structural metrics computed without an LLM:

- `expected_doc_recall@k` — did at least one chunk from each `expected_docs` make it into the top-k?
- `refusal_accuracy` — for `expected_docs=[]` cases, did the system output `INSUFFICIENT_CONTEXT`?

## Running it

```bash
# Smoke (3 examples, fast; what CI runs on every PR):
lci-eval --smoke

# Full eval against the active prompt config:
lci-eval

# Specific prompt version:
lci-eval --prompt v1
lci-eval --prompt v2
```

Each run writes its scores to `evals/results/<prompt-version>__<timestamp>.json` and prints a comparison table.

## CI gating

`.github/workflows/eval.yml` runs the smoke eval on every PR that touches `prompts/`, `src/`, `evals/`, or `requirements.txt`. PRs fail if faithfulness or refusal_accuracy drops below the thresholds defined at the top of the workflow file. The full eval runs nightly.
