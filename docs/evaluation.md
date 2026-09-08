# RAG Evaluation

This document summarizes the repository's reproducible offline RAG evaluation. It intentionally omits machine-specific cache paths and internal development logs.

## Scope

The evaluator measures the existing dog-care RAG pipeline without calling a real LLM:

```text
question
→ risk-policy decision
→ BAAI/bge-small-zh-v1.5 embedding
→ local cosine retrieval
→ answer/fallback decision
→ source-display decision
→ labelled metrics
```

It does not change the embedding model, retrieval algorithm, default `top_k=3`, thresholds, chunking, prompts, or knowledge documents.

## Dataset schema

`tests/data/rag_eval_cases.jsonl` contains 22 fixed cases across `normal`, `training_behavior`, `boundary`, and `no_coverage` categories.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Stable case identifier |
| `category` | yes | Evaluation category |
| `question` | yes | Query sent through the RAG service |
| `should_show_sources` | yes | Expected source-presence policy |
| `expected_doc_ids` | no | Explicit retrieval ground truth |
| `expected_outcome` | no | Expected `answer` or `fallback` branch |
| `policy_provenance` | no | Human-review provenance for a policy clarification |
| `policy_note` | no | Concise explanation of that clarification |

Only 12 cases currently have unambiguous `expected_doc_ids`. The other 10 remain unresolved for retrieval metrics and are skipped rather than assigned guessed labels.

## Metrics

- **Expected source hit rate**: labelled cases where raw top-k contains at least one expected document.
- **Recall@k**: fraction of each labelled case's expected document IDs found in raw top-k, averaged over labelled cases.
- **Source display accuracy**: whether source presence matches `should_show_sources`.
- **Displayed source hit rate**: among cases with retrieval ground truth that expect sources, whether an expected document is actually displayed.
- **No-coverage fallback accuracy**: labelled no-coverage cases that avoid the context/LLM branch.
- **Boundary source-policy accuracy**: source-display accuracy restricted to boundary cases.
- **Latency**: in-process evaluator timing. It excludes HTTP, Redis and real-LLM latency.

No source-precision metric is reported because boundary/no-coverage cases do not yet have complete per-source relevance labels.

## Strict-offline run

The configured model must already be available locally. The repository does not bundle model weights and the evaluator does not require an LLM Provider API key.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python scripts/evaluate_rag.py
```

The command prints a summary and writes `reports/rag_eval_report.json`.

## Baseline and minimal policy fix

| Metric | Baseline | After fix |
| --- | ---: | ---: |
| Expected source hit rate | 1.0000 | 1.0000 |
| Recall@3 | 1.0000 | 1.0000 |
| Source display accuracy | 0.7727 | 0.9545 |
| Displayed source hit rate | 0.9167 | 0.9167 |
| No-coverage fallback accuracy | 0.6000 | 1.0000 |
| Boundary source-policy accuracy | 0.2000 | 1.0000 |

Failure analysis identified four categories:

1. missing high-risk intent coverage before retrieval;
2. source-display policy versus label mismatch;
3. retrieval relevance noise on cases without positive document labels;
4. ambiguous or over-narrow evaluation ground truth.

The reviewed minimal fix addressed only three clear safety-policy gaps:

- symptom plus medication decision;
- symptom plus disease inference;
- unsupported paw-licking/chewing causal inference.

These compound intents now use the existing cautious fallback before embedding, retrieval and LLM execution. Tests include positive paraphrases and safe negative controls. No threshold or retrieval tuning was used.

Three boundary labels were separately clarified by human review: cautious daily observation, adaptation/feeding, and routine/companionship questions may show directly relevant sources while avoiding diagnosis or unsupported causal claims. Consequently, the boundary metric change includes label clarification and must not be interpreted as pure algorithmic gain.

## Result boundaries

- `Recall@3=1.0000` applies only to the 12 retrieval-labelled cases.
- The fixed 22-case set is too small to establish general or production-grade quality.
- `training_004` remains a known source-display miss under the unchanged source threshold.
- `normal_002` and `normal_005` remain ground-truth design debt; their canonical document labels may be too narrow.
- The evaluator observes branch behaviour, retrieval and source policy, not natural-language answer quality.
- Cold-start timing is dominated by local model initialization and is not a performance claim.
