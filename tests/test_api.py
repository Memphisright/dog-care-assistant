import json

from app.services.memory_service import MemoryService
from tests.fake_redis import FakeRedis


def test_task_type_invalid_returns_422(test_client):
    payload = {
        "session_id": "s1",
        "user_message": "hello",
        "task_type": "unknown_type",
    }
    response = test_client.post(
        "/api/v1/chat/structured",
        json=payload,
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_stream_sse_events_format(test_client):
    payload = {"session_id": "s-stream", "user_message": "你好"}
    response = test_client.post(
        "/api/v1/chat/stream",
        json=payload,
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 200
    text = response.text
    assert "event: token" in text
    assert "event: done" in text
    assert "event: error" not in text


def test_legacy_endpoint_plain_text_chunks(test_client):
    payload = {"session_id": "legacy", "user_message": "测试兼容"}
    response = test_client.post(
        "/chat_with_memory",
        json=payload,
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" not in response.text
    assert response.text.strip() != ""
    assert "[ERROR]" not in response.text


def test_lock_conflict_returns_409(test_client):
    redis_client = test_client.app.state.redis
    session_id = "lock-session"
    test_client.app.state.lock_manager._ttl_seconds = 120
    # Manually hold the lock to trigger session conflict.
    test_client.app.state.lock_manager._redis._values[f"lock:chat:{session_id}"] = "held-token"
    payload = {"session_id": session_id, "user_message": "hello"}
    response = test_client.post(
        "/api/v1/chat/stream",
        json=payload,
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SESSION_LOCKED"
    assert redis_client is not None


def test_rate_limit_returns_429():
    from app.core.config import Settings
    from app.main import create_app
    from fastapi.testclient import TestClient
    from tests.conftest import FakeLLMService

    settings = Settings(
        DEEPSEEK_API_KEY="test-key",
        APP_API_KEYS="test-key",
        DEV_BYPASS_AUTH=False,
        RATE_LIMIT_PER_MINUTE=1,
        MAX_CONCURRENT_PER_KEY=2,
    )
    app = create_app(settings=settings, redis_client=FakeRedis())
    with TestClient(app) as client:
        client.app.state.llm_service = FakeLLMService()
        payload = {"session_id": "rl", "user_message": "hello"}
        first = client.post(
            "/api/v1/chat/stream",
            json=payload,
            headers={"x-api-key": "test-key"},
        )
        assert first.status_code == 200
        second = client.post(
            "/api/v1/chat/stream",
            json=payload,
            headers={"x-api-key": "test-key"},
        )
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "RATE_LIMITED"


def test_memory_sliding_window_and_ttl():
    import asyncio

    async def _run():
        redis_client = FakeRedis()
        memory = MemoryService(redis_client, ttl_seconds=3600, max_turns=2)
        for i in range(4):
            await memory.append_exchange(
                "m1",
                user_message=f"user-{i}",
                assistant_message=f"assistant-{i}",
            )
        raw = await redis_client.get("chat_history:m1")
        data = json.loads(raw)
        # system + last 2 turns (4 messages)
        assert len(data) == 5
        assert data[1]["content"] == "user-2"
        assert data[-1]["content"] == "assistant-3"
        assert redis_client._expires["chat_history:m1"] == 3600

    asyncio.run(_run())
