"""Deterministic, offline evaluation helpers for the existing RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.embedding_service import EmbeddingProvider
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.rag_service import RagService


class EvaluationDataError(ValueError):
    """Raised when a JSONL evaluation case cannot be safely interpreted."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    category: str
    question: str
    should_show_sources: bool
    expected_doc_ids: tuple[str, ...] | None
    expected_outcome: str | None
    policy_provenance: str | None
    policy_note: str | None


class RecordingEmbeddingProvider(EmbeddingProvider):
    """Delegates embeddings while retaining the vector used by RagService."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self.last_query_vector: list[float] | None = None

    async def embed_query(self, text: str) -> list[float]:
        vector = await self._provider.embed_query(text)
        self.last_query_vector = vector
        return vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.embed_documents(texts)


class DeterministicOfflineLLM:
    """Keeps evaluation offline while exposing whether RAG used retrieved context."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete_text(self, *, messages, model=None, temperature=0.2):
        _ = (messages, model, temperature)
        self.calls += 1
        return "离线评测占位回答：已走到基于检索片段的回答分支。", None


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            raw_case = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(raw_case, dict):
            raise EvaluationDataError(f"Line {line_number}: case must be a JSON object.")

        case_id = _required_string(raw_case, "id", line_number)
        category = _required_string(raw_case, "category", line_number)
        question = _required_string(raw_case, "question", line_number)
        if case_id in seen_ids:
            raise EvaluationDataError(f"Line {line_number}: duplicate case id `{case_id}`.")
        seen_ids.add(case_id)

        should_show_sources = raw_case.get("should_show_sources")
        if not isinstance(should_show_sources, bool):
            raise EvaluationDataError(
                f"Line {line_number}: `should_show_sources` must be a boolean."
            )

        expected_doc_ids = _optional_doc_ids(raw_case, line_number)
        expected_outcome = raw_case.get("expected_outcome")
        if expected_outcome is not None and expected_outcome not in {"answer", "fallback"}:
            raise EvaluationDataError(
                f"Line {line_number}: `expected_outcome` must be `answer` or `fallback`."
            )
        policy_provenance = _optional_string(raw_case, "policy_provenance", line_number)
        policy_note = _optional_string(raw_case, "policy_note", line_number)

        cases.append(
            EvaluationCase(
                case_id=case_id,
                category=category,
                question=question,
                should_show_sources=should_show_sources,
                expected_doc_ids=expected_doc_ids,
                expected_outcome=expected_outcome,
                policy_provenance=policy_provenance,
                policy_note=policy_note,
            )
        )
    return cases


async def evaluate_cases(
    *,
    cases: list[EvaluationCase],
    rag_service: RagService,
    knowledge_base_service: KnowledgeBaseService,
    embedding_provider: RecordingEmbeddingProvider,
    offline_llm: DeterministicOfflineLLM,
    top_k: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        offline_llm.calls = 0
        embedding_provider.last_query_vector = None
        started = time.perf_counter()
        _, sources = await rag_service.query(user_message=case.question, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000

        raw_hits = (
            knowledge_base_service.retrieve(
                query_vector=embedding_provider.last_query_vector,
                top_k=top_k,
            )
            if embedding_provider.last_query_vector is not None
            else []
        )
        retrieved_doc_ids = _unique_doc_ids(hit.chunk.doc_id for hit in raw_hits)
        displayed_doc_ids = _unique_doc_ids(source.doc_id for source in sources)
        source_display_correct = bool(sources) == case.should_show_sources
        expected_source_hit = _expected_source_hit(case.expected_doc_ids, retrieved_doc_ids)
        displayed_expected_source_hit = _displayed_expected_source_hit(
            case.expected_doc_ids,
            displayed_doc_ids,
            should_show_sources=case.should_show_sources,
        )
        recall_at_k = _recall_at_k(case.expected_doc_ids, retrieved_doc_ids)
        used_retrieved_context = offline_llm.calls > 0
        outcome_correct = _outcome_correct(case.expected_outcome, used_retrieved_context)

        results.append(
            {
                "id": case.case_id,
                "category": case.category,
                "question": case.question,
                "expected_doc_ids": list(case.expected_doc_ids)
                if case.expected_doc_ids is not None
                else None,
                "raw_top_k": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "doc_id": hit.chunk.doc_id,
                        "score": round(hit.score, 4),
                    }
                    for hit in raw_hits
                ],
                "retrieved_doc_ids": retrieved_doc_ids,
                "displayed_sources": [
                    {
                        "chunk_id": source.chunk_id,
                        "doc_id": source.doc_id,
                        "score": source.score,
                    }
                    for source in sources
                ],
                "displayed_doc_ids": displayed_doc_ids,
                "should_show_sources": case.should_show_sources,
                "source_display_correct": source_display_correct,
                "expected_source_hit": expected_source_hit,
                "displayed_expected_source_hit": displayed_expected_source_hit,
                "recall_at_k": recall_at_k,
                "expected_outcome": case.expected_outcome,
                "policy_provenance": case.policy_provenance,
                "policy_note": case.policy_note,
                "used_retrieved_context": used_retrieved_context,
                "outcome_correct": outcome_correct,
                "latency_ms": round(latency_ms, 3),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_deterministic",
        "status": "completed",
        "top_k": top_k,
        "case_count": len(results),
        "summary": _summarize(results),
        "cases": results,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def blocked_report(*, top_k: int, reason: str, dataset_case_count: int) -> dict[str, Any]:
    """Record an unavailable local dependency without fabricating metric values."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_deterministic",
        "status": "blocked",
        "top_k": top_k,
        "dataset_case_count": dataset_case_count,
        "case_count": 0,
        "blocker": reason,
        "summary": {
            "expected_source_hit_rate": None,
            "recall_at_k": None,
            "source_display_accuracy": None,
            "displayed_source_hit_rate": None,
            "no_coverage_fallback_accuracy": None,
            "boundary_source_policy_accuracy": None,
            "latency_ms": {"count": 0, "mean": None, "p50": None, "p95": None, "max": None},
            "labelled_retrieval_cases": 0,
            "unresolved_retrieval_cases": 0,
        },
        "cases": [],
    }


def format_console_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "=== Offline RAG Evaluation ===",
        f"cases: {report['case_count']}",
        f"top_k: {report['top_k']}",
        _format_metric("expected_source_hit_rate", summary["expected_source_hit_rate"]),
        _format_metric("recall_at_k", summary["recall_at_k"]),
        _format_metric("source_display_accuracy", summary["source_display_accuracy"]),
        _format_metric("displayed_source_hit_rate", summary["displayed_source_hit_rate"]),
        _format_metric("no_coverage_fallback_accuracy", summary["no_coverage_fallback_accuracy"]),
        _format_metric("boundary_source_policy_accuracy", summary["boundary_source_policy_accuracy"]),
        _format_latency(summary),
        f"unresolved_retrieval_cases: {summary['unresolved_retrieval_cases']}",
    ]
    return "\n".join(lines)


def run_evaluation_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(evaluate_cases(**kwargs))


def _required_string(raw_case: dict[str, Any], field: str, line_number: int) -> str:
    value = raw_case.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDataError(f"Line {line_number}: `{field}` must be a non-empty string.")
    return value.strip()


def _optional_doc_ids(raw_case: dict[str, Any], line_number: int) -> tuple[str, ...] | None:
    if "expected_doc_ids" not in raw_case:
        return None
    value = raw_case["expected_doc_ids"]
    if not isinstance(value, list) or not value:
        raise EvaluationDataError(
            f"Line {line_number}: `expected_doc_ids` must be a non-empty list of strings."
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise EvaluationDataError(
            f"Line {line_number}: `expected_doc_ids` must contain non-empty strings only."
        )
    return tuple(item.strip() for item in value)


def _optional_string(
    raw_case: dict[str, Any], field: str, line_number: int
) -> str | None:
    value = raw_case.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDataError(
            f"Line {line_number}: `{field}` must be a non-empty string when provided."
        )
    return value.strip()


def _unique_doc_ids(doc_ids) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def _expected_source_hit(expected_doc_ids, retrieved_doc_ids) -> bool | None:
    if expected_doc_ids is None:
        return None
    return bool(set(expected_doc_ids).intersection(retrieved_doc_ids))


def _displayed_expected_source_hit(
    expected_doc_ids,
    displayed_doc_ids,
    *,
    should_show_sources: bool,
) -> bool | None:
    if expected_doc_ids is None or not should_show_sources:
        return None
    return bool(set(expected_doc_ids).intersection(displayed_doc_ids))


def _recall_at_k(expected_doc_ids, retrieved_doc_ids) -> float | None:
    if expected_doc_ids is None:
        return None
    return len(set(expected_doc_ids).intersection(retrieved_doc_ids)) / len(expected_doc_ids)


def _outcome_correct(expected_outcome: str | None, used_retrieved_context: bool) -> bool | None:
    if expected_outcome is None:
        return None
    return used_retrieved_context if expected_outcome == "answer" else not used_retrieved_context


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    labelled_retrieval = [item for item in results if item["expected_source_hit"] is not None]
    labelled_displayed_sources = [
        item for item in results if item["displayed_expected_source_hit"] is not None
    ]
    no_coverage = [item for item in results if item["category"] == "no_coverage"]
    boundary = [item for item in results if item["category"] == "boundary"]
    latencies = [item["latency_ms"] for item in results]
    return {
        "expected_source_hit_rate": _mean_or_none(
            [float(item["expected_source_hit"]) for item in labelled_retrieval]
        ),
        "recall_at_k": _mean_or_none([item["recall_at_k"] for item in labelled_retrieval]),
        "source_display_accuracy": _mean_or_none(
            [float(item["source_display_correct"]) for item in results]
        ),
        "displayed_source_hit_rate": _mean_or_none(
            [float(item["displayed_expected_source_hit"]) for item in labelled_displayed_sources]
        ),
        "no_coverage_fallback_accuracy": _mean_or_none(
            [float(item["outcome_correct"]) for item in no_coverage if item["outcome_correct"] is not None]
        ),
        "boundary_source_policy_accuracy": _mean_or_none(
            [float(item["source_display_correct"]) for item in boundary]
        ),
        "latency_ms": {
            "count": len(latencies),
            "mean": _mean_or_none(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "labelled_retrieval_cases": len(labelled_retrieval),
        "unresolved_retrieval_cases": len(results) - len(labelled_retrieval),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _format_metric(name: str, value: float | None) -> str:
    return f"{name}: {value:.4f}" if value is not None else f"{name}: n/a"


def _format_latency(summary: dict[str, Any]) -> str:
    latency = summary["latency_ms"]
    if latency["count"] == 0:
        return "latency_ms: n/a"
    return (
        "latency_ms: "
        f"mean={latency['mean']:.3f}, p50={latency['p50']:.3f}, "
        f"p95={latency['p95']:.3f}, max={latency['max']:.3f}"
    )
