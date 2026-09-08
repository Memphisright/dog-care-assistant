import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.rag import RagQueryRequest
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.rag_service import RagService
from tests.conftest import FakeLLMService
from tests.fake_redis import FakeRedis


class FakeEmbeddingProvider:
    def __init__(self, *, query_vector: list[float]) -> None:
        self._query_vector = query_vector
        self.query_calls: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._query_vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._query_vector for _ in texts]


class FakeRagLLMService(FakeLLMService):
    def __init__(
        self,
        answer: str = "可以先从日常作息和环境安排角度做温和观察，但当前依据不足以支持更明确的判断。",
    ) -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    async def complete_text(self, *, messages, model=None, temperature=0.2):
        _ = (model, temperature)
        self.calls.append(messages)
        return self.answer, None


def _write_index(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _valid_index_payload() -> dict:
    return {
        "chunks": [
            {
                "doc_id": "dog_socialization_basics",
                "title": "幼犬训练与社会化基础",
                "source": "knowledge/dogs/puppy_training_and_socialization.md",
                "chunk_id": "dog_socialization_basics#chunk-001",
                "text": "幼犬训练应尽量短时、重复、正向反馈，社会化要在安全可控的环境中循序渐进地增加经验。",
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "doc_id": "feeding_basics",
                "title": "喂养与营养基础",
                "source": "knowledge/dogs/feeding_and_nutrition_basics.md",
                "chunk_id": "feeding_basics#chunk-001",
                "text": "换粮更适合多天渐进过渡，并持续观察食欲、饮水、排便和精神状态是否稳定。",
                "vector": [0.0, 1.0, 0.0],
            },
        ]
    }


def _build_settings(index_path: Path, **overrides) -> Settings:
    params = {
        "DEEPSEEK_API_KEY": "test-key",
        "APP_API_KEYS": "test-key",
        "DEV_BYPASS_AUTH": False,
        "RATE_LIMIT_PER_MINUTE": 30,
        "MAX_CONCURRENT_PER_KEY": 2,
        "RAG_INDEX_PATH": str(index_path),
        "RAG_SCORE_THRESHOLD": 0.3,
        "RAG_ANSWER_SCORE_THRESHOLD": 0.42,
        "RAG_SOURCE_SCORE_THRESHOLD": 0.54,
        "RAG_SYMPTOM_ANSWER_SCORE_THRESHOLD": 0.52,
        "RAG_SYMPTOM_SOURCE_SCORE_THRESHOLD": 0.66,
    }
    params.update(overrides)
    return Settings(**params)


def _build_rag_service(
    *,
    index_path: Path,
    query_vector: list[float],
    answer: str = "可以先从日常作息和环境安排角度做温和观察，但当前依据不足以支持更明确的判断。",
    settings_overrides: dict | None = None,
):
    resolved_settings = _build_settings(index_path, **(settings_overrides or {}))
    llm_service = FakeRagLLMService(answer=answer)
    knowledge_base_service = KnowledgeBaseService(
        index_path=str(index_path),
        score_threshold=resolved_settings.rag_score_threshold,
    )
    rag_service = RagService(
        embedding_provider=FakeEmbeddingProvider(query_vector=query_vector),
        knowledge_base_service=knowledge_base_service,
        llm_service=llm_service,
        answer_score_threshold=resolved_settings.rag_answer_score_threshold,
        source_score_threshold=resolved_settings.rag_source_score_threshold,
        symptom_answer_score_threshold=resolved_settings.rag_symptom_answer_score_threshold,
        symptom_source_score_threshold=resolved_settings.rag_symptom_source_score_threshold,
    )
    return rag_service, llm_service, resolved_settings


def test_rag_query_returns_answer_and_sources(tmp_path):
    index_path = _write_index(tmp_path / "dog_basic_index.json", _valid_index_payload())
    settings = _build_settings(index_path)
    app = create_app(settings=settings, redis_client=FakeRedis())

    with TestClient(app) as client:
        rag_service, llm_service, _ = _build_rag_service(
            index_path=index_path,
            query_vector=[1.0, 0.0, 0.0],
        )
        client.app.state.llm_service = llm_service
        client.app.state.rag_service = rag_service

        response = client.post(
            "/api/v1/rag/query",
            json={"user_message": "幼犬什么时候适合开始做社会化训练？", "top_k": 2},
            headers={"x-api-key": "test-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["request_id"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["title"] == "幼犬训练与社会化基础"
    assert llm_service.calls, "LLM should be called when retrieval returns strong hits."


def test_rag_query_top_k_validation_returns_422(tmp_path):
    index_path = tmp_path / "missing.json"
    settings = _build_settings(index_path)
    app = create_app(settings=settings, redis_client=FakeRedis())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rag/query",
            json={"user_message": "测试", "top_k": 0},
            headers={"x-api-key": "test-key"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_rag_query_returns_503_when_index_missing(tmp_path):
    index_path = tmp_path / "missing_index.json"
    settings = _build_settings(index_path)
    app = create_app(settings=settings, redis_client=FakeRedis())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rag/query",
            json={"user_message": "幼犬训练怎么开始？", "top_k": 4},
            headers={"x-api-key": "test-key"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "RAG_INDEX_NOT_READY"


def test_rag_query_returns_503_for_invalid_indexes(tmp_path):
    invalid_payloads = [
        ("empty", {"chunks": []}),
        ("broken_json", "{not-json"),
        (
            "vector_mismatch",
            {
                "chunks": [
                    {
                        "doc_id": "a",
                        "title": "A",
                        "source": "a.md",
                        "chunk_id": "a#chunk-001",
                        "text": "A",
                        "vector": [1.0, 0.0],
                    },
                    {
                        "doc_id": "b",
                        "title": "B",
                        "source": "b.md",
                        "chunk_id": "b#chunk-001",
                        "text": "B",
                        "vector": [1.0, 0.0, 0.0],
                    },
                ]
            },
        ),
    ]

    for name, payload in invalid_payloads:
        index_path = tmp_path / f"{name}.json"
        if name == "broken_json":
            index_path.write_text(payload, encoding="utf-8")
        else:
            _write_index(index_path, payload)

        settings = _build_settings(index_path)
        app = create_app(settings=settings, redis_client=FakeRedis())
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/rag/query",
                json={"user_message": "测试", "top_k": 2},
                headers={"x-api-key": "test-key"},
            )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "RAG_INDEX_NOT_READY"


def test_chat_endpoints_still_work_when_rag_index_invalid(tmp_path):
    index_path = tmp_path / "missing_index.json"
    settings = _build_settings(index_path)
    app = create_app(settings=settings, redis_client=FakeRedis())

    with TestClient(app) as client:
        client.app.state.llm_service = FakeLLMService()

        stream_response = client.post(
            "/api/v1/chat/stream",
            json={"session_id": "rag-missing-chat", "user_message": "你好"},
            headers={"x-api-key": "test-key"},
        )
        structured_response = client.post(
            "/api/v1/chat/structured",
            json={
                "session_id": "rag-missing-structured",
                "user_message": "今天状态还不错",
                "task_type": "daily_card",
            },
            headers={"x-api-key": "test-key"},
        )

    assert stream_response.status_code == 200
    assert structured_response.status_code == 200


def test_rag_query_returns_empty_sources_for_low_relevance(tmp_path):
    index_path = _write_index(tmp_path / "dog_basic_index.json", _valid_index_payload())
    settings = _build_settings(index_path, RAG_SCORE_THRESHOLD=0.95)
    app = create_app(settings=settings, redis_client=FakeRedis())

    with TestClient(app) as client:
        rag_service, llm_service, _ = _build_rag_service(
            index_path=index_path,
            query_vector=[0.4, 0.4, 0.4],
            settings_overrides={"RAG_SCORE_THRESHOLD": 0.95},
        )
        client.app.state.llm_service = llm_service
        client.app.state.rag_service = rag_service

        response = client.post(
            "/api/v1/rag/query",
            json={"user_message": "怎么安排更合适？", "top_k": 3},
            headers={"x-api-key": "test-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["answer"].strip() != ""
    assert any(
        keyword in body["answer"]
        for keyword in ("知识库", "依据", "未覆盖", "无法确认")
    )
    assert not llm_service.calls, "LLM should not be called when retrieval never reaches answer threshold."


def test_rag_query_returns_clean_fallback_for_symptom_questions(tmp_path):
    index_path = _write_index(tmp_path / "dog_basic_index.json", _valid_index_payload())
    settings = _build_settings(index_path)
    app = create_app(settings=settings, redis_client=FakeRedis())

    with TestClient(app) as client:
        rag_service, llm_service, _ = _build_rag_service(
            index_path=index_path,
            query_vector=[1.0, 0.0, 0.0],
        )
        client.app.state.llm_service = llm_service
        client.app.state.rag_service = rag_service

        response = client.post(
            "/api/v1/rag/query",
            json={"user_message": "狗狗发烧是什么问题？", "top_k": 3},
            headers={"x-api-key": "test-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["answer"].strip() != ""
    assert any(
        keyword in body["answer"]
        for keyword in ("知识库", "依据", "不要只依赖", "症状判断")
    )
    assert not llm_service.calls, "LLM should not be called when symptom-like questions stay below answer threshold."


def test_rag_query_can_answer_without_showing_sources_for_boundary_symptom_questions(tmp_path):
    index_path = _write_index(tmp_path / "dog_basic_index.json", _valid_index_payload())
    settings = _build_settings(
        index_path,
        RAG_SYMPTOM_ANSWER_SCORE_THRESHOLD=0.52,
        RAG_SYMPTOM_SOURCE_SCORE_THRESHOLD=0.7,
    )
    app = create_app(settings=settings, redis_client=FakeRedis())

    with TestClient(app) as client:
        rag_service, llm_service, _ = _build_rag_service(
            index_path=index_path,
            query_vector=[0.45, 0.6, 0.65],
            answer="可以先从饮食节奏和作息观察角度做温和记录，但当前证据不足以支持更明确的原因判断。",
            settings_overrides={
                "RAG_SYMPTOM_ANSWER_SCORE_THRESHOLD": 0.52,
                "RAG_SYMPTOM_SOURCE_SCORE_THRESHOLD": 0.7,
            },
        )
        client.app.state.llm_service = llm_service
        client.app.state.rag_service = rag_service

        response = client.post(
            "/api/v1/rag/query",
            json={"user_message": "狗狗最近精神不太好，而且睡觉不太规律，平时可以先怎么观察？", "top_k": 3},
            headers={"x-api-key": "test-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].strip() != ""
    assert body["sources"] == []
    assert llm_service.calls, "LLM should still be called when hits pass answer threshold."


def test_rag_query_request_defaults_top_k_to_three():
    payload = RagQueryRequest(user_message="幼犬适合多久散步一次？")
    assert payload.top_k == 3


@pytest.mark.parametrize(
    "question",
    [
        "狗狗呕吐后要不要吃药？",
        "狗狗吐了以后能不能自己喂药？",
        "狗狗呕吐后该吃什么药？",
        "狗狗拉稀可以先用止泻药吗？",
        "狗狗咳嗽是不是感冒了？",
        "狗狗咳嗽是不是生病了？",
        "狗狗咳嗽是不是呼吸道有问题？",
        "狗狗一直咳是什么病？",
        "狗狗老舔爪子是不是焦虑？",
        "狗狗反复舔爪是不是焦虑导致的？",
        "狗狗一直舔脚是不是皮肤有问题？",
        "狗狗啃自己爪子是什么原因？",
    ],
)
def test_high_risk_intent_combinations_short_circuit_before_embedding(question, tmp_path):
    index_path = _write_index(tmp_path / "dog_basic_index.json", _valid_index_payload())
    rag_service, llm_service, _ = _build_rag_service(
        index_path=index_path,
        query_vector=[1.0, 0.0, 0.0],
    )

    answer, sources = asyncio.run(rag_service.query(user_message=question, top_k=3))

    assert sources == []
    assert answer.strip()
    assert rag_service._embedding_provider.query_calls == []
    assert not llm_service.calls


@pytest.mark.parametrize(
    "question",
    [
        "训练奖励和日常主食应该怎么安排？",
        "天气冷时散步时间怎么安排？",
        "狗狗出门不愿意走是不是路线太复杂？",
        "怎么训练狗狗配合擦爪子？",
        "外出回来怎么做基础脚部清洁？",
        "狗狗咬玩具时怎么做替代训练？",
    ],
)
def test_high_risk_intent_combinations_do_not_capture_safe_controls(question):
    assert not RagService._is_strict_diagnostic_query(question)
