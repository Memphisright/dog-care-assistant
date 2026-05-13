import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.structured import DailyCardOutput, RiskAnalysisOutput, RiskItem, TaskType
from tests.fake_redis import FakeRedis


class FakeLLMService:
    async def aclose(self):
        return None

    async def stream_completion(self, *, messages, model=None, temperature=0.7):
        _ = (messages, model, temperature)
        for token in ["?", "?"]:
            yield token

    async def complete_text(self, *, messages, model=None, temperature=0.2):
        _ = (messages, model, temperature)
        return "????", None

    async def complete_structured(self, *, task_type, messages, tool_context, model=None):
        _ = (messages, tool_context, model)
        if task_type is TaskType.daily_card:
            return (
                "????????",
                DailyCardOutput(
                    date="2026-04-02",
                    summary="??????",
                    focus="?????",
                    priorities=["??????", "????????"],
                    risks=["???????"],
                    action_items=["???????", "????????"],
                ),
                None,
            )
        return (
            "??????????",
            RiskAnalysisOutput(
                overall_level="medium",
                summary="?????????????",
                key_risks=[
                    RiskItem(
                        risk="????",
                        level="medium",
                        impact="???????",
                        mitigation="????????????",
                    )
                ],
                recommendations=["??????????"],
            ),
            None,
        )


@pytest.fixture
def test_client():
    settings = Settings(
        DEEPSEEK_API_KEY="test-key",
        APP_API_KEYS="test-key",
        DEV_BYPASS_AUTH=False,
        RATE_LIMIT_PER_MINUTE=30,
        MAX_CONCURRENT_PER_KEY=2,
    )
    fake_redis = FakeRedis()
    app = create_app(settings=settings, redis_client=fake_redis)
    with TestClient(app) as client:
        client.app.state.llm_service = FakeLLMService()
        yield client
