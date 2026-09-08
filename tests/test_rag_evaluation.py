import asyncio
import json

import pytest

from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.rag_evaluation import (
    DeterministicOfflineLLM,
    EvaluationDataError,
    RecordingEmbeddingProvider,
    blocked_report,
    evaluate_cases,
    format_console_summary,
    load_evaluation_cases,
    write_report,
)
from app.services.rag_service import RagService


class FakeEmbeddingProvider:
    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "社会化" in text else [0.0, 1.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _write_cases(path, lines: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines),
        encoding="utf-8",
    )


def _make_runtime(tmp_path):
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "doc_id": "socialization",
                        "title": "社会化",
                        "source": "socialization.md",
                        "chunk_id": "socialization#1",
                        "text": "社会化训练",
                        "vector": [1.0, 0.0],
                    },
                    {
                        "doc_id": "other",
                        "title": "其他",
                        "source": "other.md",
                        "chunk_id": "other#1",
                        "text": "其他",
                        "vector": [0.0, 1.0],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    knowledge_base_service = KnowledgeBaseService(
        index_path=str(index_path), score_threshold=0.3
    )
    embedding_provider = RecordingEmbeddingProvider(FakeEmbeddingProvider())
    offline_llm = DeterministicOfflineLLM()
    rag_service = RagService(
        embedding_provider=embedding_provider,
        knowledge_base_service=knowledge_base_service,
        llm_service=offline_llm,
        answer_score_threshold=0.42,
        source_score_threshold=0.54,
        symptom_answer_score_threshold=0.52,
        symptom_source_score_threshold=0.66,
    )
    return rag_service, knowledge_base_service, embedding_provider, offline_llm


def test_case_loader_supports_backward_compatible_and_labelled_cases(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(
        cases_path,
        [
            {
                "id": "legacy",
                "category": "normal",
                "question": "问题",
                "should_show_sources": True,
            },
            {
                "id": "labelled",
                "category": "normal",
                "question": "另一个问题",
                "should_show_sources": True,
                "expected_doc_ids": ["doc-a"],
                "expected_outcome": "answer",
                "policy_provenance": "human_review_approved_2026-09-08",
                "policy_note": "人工批准的策略澄清。",
            },
        ],
    )

    cases = load_evaluation_cases(cases_path)

    assert cases[0].expected_doc_ids is None
    assert cases[1].expected_doc_ids == ("doc-a",)
    assert cases[1].expected_outcome == "answer"
    assert cases[1].policy_provenance == "human_review_approved_2026-09-08"
    assert cases[1].policy_note == "人工批准的策略澄清。"


def test_case_loader_rejects_malformed_case(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text('{"id":"bad"}\n', encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="category"):
        load_evaluation_cases(cases_path)


def test_empty_cases_produce_an_empty_report(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("\n", encoding="utf-8")
    rag_service, knowledge_base_service, embedding_provider, offline_llm = _make_runtime(tmp_path)

    report = asyncio.run(
        evaluate_cases(
            cases=load_evaluation_cases(cases_path),
            rag_service=rag_service,
            knowledge_base_service=knowledge_base_service,
            embedding_provider=embedding_provider,
            offline_llm=offline_llm,
            top_k=3,
        )
    )

    assert report["case_count"] == 0
    assert report["summary"]["recall_at_k"] is None


def test_blocked_report_never_fabricates_metrics():
    report = blocked_report(
        top_k=3,
        reason="embedding unavailable",
        dataset_case_count=22,
    )

    assert report["status"] == "blocked"
    assert report["dataset_case_count"] == 22
    assert report["summary"]["expected_source_hit_rate"] is None
    assert report["summary"]["displayed_source_hit_rate"] is None
    assert report["cases"] == []


def test_evaluation_computes_metrics_skips_unresolved_and_writes_report(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(
        cases_path,
        [
            {
                "id": "retrieval",
                "category": "normal",
                "question": "社会化怎么开始？",
                "should_show_sources": True,
                "expected_doc_ids": ["socialization"],
                "expected_outcome": "answer",
            },
            {
                "id": "boundary",
                "category": "boundary",
                "question": "另一个日常问题",
                "should_show_sources": True,
                "policy_provenance": "human_review_approved_2026-09-08",
                "policy_note": "人工批准展示来源。",
            },
            {
                "id": "no-coverage",
                "category": "no_coverage",
                "question": "狗狗发烧是什么问题？",
                "should_show_sources": False,
                "expected_outcome": "fallback",
            },
        ],
    )
    rag_service, knowledge_base_service, embedding_provider, offline_llm = _make_runtime(tmp_path)

    report = asyncio.run(
        evaluate_cases(
            cases=load_evaluation_cases(cases_path),
            rag_service=rag_service,
            knowledge_base_service=knowledge_base_service,
            embedding_provider=embedding_provider,
            offline_llm=offline_llm,
            top_k=3,
        )
    )
    report_path = tmp_path / "report.json"
    write_report(report, report_path)

    assert report["summary"]["expected_source_hit_rate"] == 1.0
    assert report["summary"]["recall_at_k"] == 1.0
    assert report["summary"]["displayed_source_hit_rate"] == 1.0
    assert report["summary"]["unresolved_retrieval_cases"] == 2
    assert report["summary"]["no_coverage_fallback_accuracy"] == 1.0
    assert json.loads(report_path.read_text(encoding="utf-8"))["case_count"] == 3
    assert report["cases"][0]["raw_top_k"][0] == {
        "chunk_id": "socialization#1",
        "doc_id": "socialization",
        "score": 1.0,
    }
    assert report["cases"][0]["displayed_expected_source_hit"] is True
    assert report["cases"][0]["displayed_sources"] == [
        {
            "chunk_id": "socialization#1",
            "doc_id": "socialization",
            "score": 1.0,
        }
    ]
    assert report["cases"][1]["displayed_expected_source_hit"] is None
    assert report["cases"][1]["policy_provenance"] == "human_review_approved_2026-09-08"
    assert report["cases"][1]["policy_note"] == "人工批准展示来源。"
    assert "unresolved_retrieval_cases: 2" in format_console_summary(report)
