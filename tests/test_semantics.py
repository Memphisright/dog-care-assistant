import asyncio
import json
from datetime import date

from app.schemas.structured import TaskType
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService, PET_OBSERVER_SYSTEM_PROMPT
from tests.fake_redis import FakeRedis


class StubLLMService(LLMService):
    def __init__(self) -> None:
        pass

    async def complete_text(self, *, messages, model=None, temperature=0.2):
        _ = (messages, model, temperature)
        payload = {
            "answer": "The pet looked calm overall.",
            "structured_output": {
                "date": "2024-01-01",
                "summary": "Stable appetite and activity.",
                "focus": "Hydration and rest",
                "priorities": ["Observe water intake"],
                "risks": ["Mild evening restlessness"],
                "action_items": ["Keep tonight's routine consistent"],
            },
        }
        return json.dumps(payload), None


def test_daily_card_date_is_injected_by_server():
    async def _run():
        service = StubLLMService()
        answer, structured_output, usage = await service.complete_structured(
            task_type=TaskType.daily_card,
            messages=[{"role": "system", "content": "test"}],
            tool_context={"memory_lookup": {"hits": [], "count": 0}},
            model=None,
        )
        assert answer == "The pet looked calm overall."
        assert structured_output.date == date.today().isoformat()
        assert usage is None

    asyncio.run(_run())


def test_memory_history_normalizes_to_pet_persona():
    async def _run():
        redis_client = FakeRedis()
        await redis_client.setex(
            "chat_history:s1",
            3600,
            json.dumps(
                [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "hello"},
                ]
            ),
        )
        memory = MemoryService(redis_client, ttl_seconds=3600, max_turns=4)
        history = await memory.load_history("s1")
        assert history[0]["role"] == "system"
        assert history[0]["content"] == PET_OBSERVER_SYSTEM_PROMPT
        assert history[1]["content"] == "hello"

    asyncio.run(_run())


def test_pet_persona_prompt_discourages_ai_self_identification():
    prompt = PET_OBSERVER_SYSTEM_PROMPT.lower()
    assert "pet speaking to your human" in prompt
    assert "do not describe yourself as an ai assistant" in prompt


class RepairingLLMService(LLMService):
    def __init__(self) -> None:
        self._calls = 0

    async def complete_text(self, *, messages, model=None, temperature=0.2):
        _ = (messages, model, temperature)
        self._calls += 1
        if self._calls == 1:
            payload = {
                "answer": "The pet looked steady overall.",
                "structured_output": {
                    "summary": "Calm appetite and activity.",
                    "priorities": "Observe appetite, hydration",
                    "risks": "Evening restlessness",
                    "action_items": "Keep bedtime routine stable",
                },
            }
            return json.dumps(payload), None

        repaired = {
            "answer": "The pet looked steady overall.",
            "structured_output": {
                "summary": "Calm appetite and activity.",
                "focus": "Routine observation",
                "priorities": ["Observe appetite", "Hydration"],
                "risks": ["Evening restlessness"],
                "action_items": ["Keep bedtime routine stable"],
            },
        }
        return json.dumps(repaired), None


def test_daily_card_string_fields_are_normalized_without_retry():
    async def _run():
        normalized = LLMService._normalize_structured_output(
            task_type=TaskType.daily_card,
            raw_structured={
                "summary": "Stable day.",
                "focus": "Comfort",
                "priorities": "food, water\nrest",
                "risks": "none",
                "action_items": "keep routine、observe litter box",
            },
            current_date="2026-04-09",
        )
        assert normalized["date"] == "2026-04-09"
        assert normalized["priorities"] == ["food", "water", "rest"]
        assert normalized["risks"] == ["none"]
        assert normalized["action_items"] == ["keep routine", "observe litter box"]

    asyncio.run(_run())


class RiskRepairingLLMService(LLMService):
    def __init__(self) -> None:
        self._calls = 0

    async def complete_text(self, *, messages, model=None, temperature=0.2):
        _ = (messages, model, temperature)
        self._calls += 1
        if self._calls == 1:
            return "This may simply reflect extra energy and appetite today.", None

        repaired = {
            "answer": "This may reflect a higher-energy day, but it is worth observing the pattern.",
            "structured_output": {
                "overall_level": "low",
                "summary": "Current observations are notable but not enough for a strong conclusion.",
                "key_risks": [
                    {
                        "risk": "Night activity increased",
                        "level": "low",
                        "impact": "May disturb routine or sleep rhythm",
                        "mitigation": "Observe for another one to two days and compare with appetite and litter habits",
                    }
                ],
                "recommendations": "Track appetite and energy over the next two days",
            },
        }
        return json.dumps(repaired), None


def test_daily_card_retries_once_after_validation_failure():
    async def _run():
        service = RepairingLLMService()
        answer, structured_output, usage = await service.complete_structured(
            task_type=TaskType.daily_card,
            messages=[{"role": "system", "content": "test"}],
            tool_context={"memory_lookup": {"hits": [], "count": 0}},
            model=None,
        )
        assert answer == "The pet looked steady overall."
        assert structured_output.date == date.today().isoformat()
        assert service._calls == 2
        assert usage is None

    asyncio.run(_run())


def test_risk_analysis_retries_when_model_returns_plain_text():
    async def _run():
        service = RiskRepairingLLMService()
        answer, structured_output, usage = await service.complete_structured(
            task_type=TaskType.risk_analysis,
            messages=[{"role": "system", "content": "test"}],
            tool_context={"memory_lookup": {"hits": [], "count": 0}},
            model=None,
        )
        assert "higher-energy day" in answer
        assert structured_output.overall_level == "low"
        assert structured_output.recommendations == [
            "Track appetite and energy over the next two days"
        ]
        assert service._calls == 2
        assert usage is None

    asyncio.run(_run())
