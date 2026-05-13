import json
import re
from datetime import date
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.core.exceptions import AppError
from app.schemas.structured import (
    DailyCardOutput,
    RiskAnalysisOutput,
    TaskType,
    UsageInfo,
)


class LLMService:
    def __init__(self, *, api_key: str, base_url: str, default_model: str) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def aclose(self) -> None:
        await self._client.close()

    def _ensure_configured(self) -> None:
        if not self._api_key:
            raise AppError(
                status_code=500,
                code="CONFIG_ERROR",
                message="DEEPSEEK_API_KEY is missing.",
            )

    async def stream_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        self._ensure_configured()
        try:
            response_stream = await self._client.chat.completions.create(
                model=model or self._default_model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                status_code=502,
                code="UPSTREAM_ERROR",
                message=f"LLM stream request failed: {exc}",
                retryable=True,
            ) from exc

    async def complete_text(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> tuple[str, UsageInfo | None]:
        self._ensure_configured()
        try:
            response = await self._client.chat.completions.create(
                model=model or self._default_model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
        except Exception as exc:
            raise AppError(
                status_code=502,
                code="UPSTREAM_ERROR",
                message=f"LLM completion request failed: {exc}",
                retryable=True,
            ) from exc

        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return content, usage

    async def complete_structured(
        self,
        *,
        task_type: TaskType,
        messages: list[dict[str, str]],
        tool_context: dict[str, Any],
        model: str | None = None,
    ):
        schema_model = (
            DailyCardOutput
            if task_type is TaskType.daily_card
            else RiskAnalysisOutput
        )
        current_date = date.today().isoformat()
        instruction = self._build_structured_instruction(
            task_type=task_type,
            schema_model=schema_model,
            tool_context=tool_context,
            current_date=current_date,
        )
        completion_messages = [
            *messages,
            {"role": "system", "content": instruction},
        ]
        text, usage = await self.complete_text(
            messages=completion_messages,
            model=model,
            temperature=0.2,
        )
        payload: dict[str, Any] | None = None
        try:
            payload = self._extract_json_payload(text)
            answer, structured_output = self._validate_structured_payload(
                task_type=task_type,
                schema_model=schema_model,
                payload=payload,
                current_date=current_date,
            )
        except AppError as exc:
            repaired_payload = await self._repair_structured_payload(
                task_type=task_type,
                original_response_text=text,
                original_payload=payload,
                repair_error=exc,
                schema_model=schema_model,
                tool_context=tool_context,
                current_date=current_date,
                model=model,
            )
            answer, structured_output = self._validate_structured_payload(
                task_type=task_type,
                schema_model=schema_model,
                payload=repaired_payload,
                current_date=current_date,
                error_prefix="Structured output validation failed after repair retry",
            )
        return answer, structured_output, usage

    @staticmethod
    def _build_structured_instruction(
        *,
        task_type: TaskType,
        schema_model,
        tool_context: dict[str, Any],
        current_date: str,
    ) -> str:
        shared_prefix = (
            "Return JSON only. The top-level response must be an object with keys "
            "'answer' and 'structured_output'. "
            "Do not use markdown, code fences, or extra commentary outside JSON. "
            f"Tool context: {json.dumps(tool_context, ensure_ascii=False)}. "
        )

        if task_type is TaskType.daily_card:
            return (
                shared_prefix
                + "This task is a pet daily status card and a summary of the pet's day, not a human work summary. "
                + "Focus on the pet's appetite, sleep, activity, mood, stool or litter changes, play energy, behavior cues, comfort, and caregiving suggestions. "
                + "The answer field may sound gently companion-like and pet-centered, but it must stay grounded in the provided observations. "
                + "The server will inject today's date as "
                + current_date
                + ", so do not invent or rely on any other date. "
                + "`structured_output` must contain: summary, focus, priorities, risks, action_items. "
                + "Each field should describe the pet's daily state and care priorities. "
                + "Write it like a pet daily observation card for an owner, or a tidy summary of the pet's day. "
                + "Do not turn it into a human diary, work log, or productivity report. "
                + "Keep list fields concrete and short."
            )

        return (
            shared_prefix
            + "This task is a pet behavior observation and risk hint summary. "
            + "Use cautious, non-diagnostic wording. "
            + "Do not present conclusions as medical facts or professional diagnosis. "
            + "Frame results as observations, possible concerns, and gentle next-step suggestions. "
            + "All user-facing text fields must be written in Simplified Chinese, including answer, summary, key_risks, impact, mitigation, and recommendations. "
            + "Do not mix in English words or short English phrases unless the user explicitly requests English. "
            + "If the input is brief, still return a complete JSON object rather than plain prose. "
            + f"`structured_output` must satisfy this JSON schema: {schema_model.model_json_schema()}."
        )

    @staticmethod
    def _extract_json_payload(content: str) -> dict[str, Any]:
        content = content.strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise AppError(
                status_code=502,
                code="STRUCTURED_OUTPUT_INVALID",
                message="No JSON object found in model response.",
            )

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise AppError(
                status_code=502,
                code="STRUCTURED_OUTPUT_INVALID",
                message=f"Invalid JSON in model response: {exc}",
            ) from exc

        if not isinstance(parsed, dict):
            raise AppError(
                status_code=502,
                code="STRUCTURED_OUTPUT_INVALID",
                message="Model response JSON must be an object.",
            )
        return parsed

    @staticmethod
    def _normalize_structured_output(
        *,
        task_type: TaskType,
        raw_structured: Any,
        current_date: str,
    ) -> Any:
        if not isinstance(raw_structured, dict):
            return raw_structured

        normalized = dict(raw_structured)
        if task_type is TaskType.daily_card:
            normalized["date"] = current_date
            for field_name in ("priorities", "risks", "action_items"):
                normalized[field_name] = LLMService._normalize_list_field(
                    normalized.get(field_name)
                )
            return normalized

        normalized["recommendations"] = LLMService._normalize_list_field(
            normalized.get("recommendations")
        )
        key_risks = normalized.get("key_risks")
        if isinstance(key_risks, dict):
            normalized["key_risks"] = [key_risks]
        return normalized

    @staticmethod
    def _normalize_list_field(value: Any) -> list[str]:
        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]
        if isinstance(value, str):
            parts = re.split(r"[\n,，、;；]+", value)
            return [part.strip(" -\t") for part in parts if part.strip(" -\t")]
        if value is None:
            return []
        text = str(value).strip()
        return [text] if text else []

    def _validate_structured_payload(
        self,
        *,
        task_type: TaskType,
        schema_model,
        payload: dict[str, Any],
        current_date: str,
        error_prefix: str = "Structured output validation failed",
    ):
        answer = str(payload.get("answer", "")).strip()
        if not answer:
            raise AppError(
                status_code=502,
                code="STRUCTURED_OUTPUT_INVALID",
                message="Structured response is missing `answer`.",
            )

        raw_structured = self._normalize_structured_output(
            task_type=task_type,
            raw_structured=payload.get("structured_output"),
            current_date=current_date,
        )
        try:
            structured_output = schema_model.model_validate(raw_structured)
        except ValidationError as exc:
            raise AppError(
                status_code=502,
                code="STRUCTURED_OUTPUT_INVALID",
                message=f"{error_prefix}: {exc}",
                details=exc.errors(),
            ) from exc
        return answer, structured_output

    async def _repair_structured_payload(
        self,
        *,
        task_type: TaskType,
        original_response_text: str,
        original_payload: dict[str, Any] | None,
        repair_error: AppError,
        schema_model,
        tool_context: dict[str, Any],
        current_date: str,
        model: str | None,
    ) -> dict[str, Any]:
        task_label = (
            "pet daily card response"
            if task_type is TaskType.daily_card
            else "pet behavior observation and risk hint response"
        )
        extra_notes = {
            "priorities_risks_action_items_must_be_lists": task_type is TaskType.daily_card,
            "recommendations_must_be_list": task_type is TaskType.risk_analysis,
            "date_is_server_injected": current_date if task_type is TaskType.daily_card else None,
        }
        repair_instruction = (
            f"Fix the output so it becomes a valid {task_label}. "
            "Do not rewrite the task meaning. Keep the existing answer semantics as much as possible. "
            "Only repair structure, field types, missing fields, or non-JSON formatting. "
            "If the original response was plain prose, convert it into the required JSON shape without changing the core meaning. "
            "All user-facing text fields must be in Simplified Chinese. "
            "Do not leave English words or mixed Chinese-English phrasing in answer or structured_output fields. "
            "Return JSON only with keys 'answer' and 'structured_output'. "
            f"Target schema: {schema_model.model_json_schema()}. "
            f"Tool context: {json.dumps(tool_context, ensure_ascii=False)}."
        )
        repair_input = {
            "task_type": task_type.value,
            "original_response_text": original_response_text,
            "original_payload": original_payload,
            "repair_error": {
                "code": repair_error.code,
                "message": repair_error.message,
                "details": repair_error.details,
            },
            "target_schema": schema_model.model_json_schema(),
            "notes": extra_notes,
        }
        repaired_text, _ = await self.complete_text(
            messages=[
                {"role": "system", "content": repair_instruction},
                {
                    "role": "user",
                    "content": json.dumps(repair_input, ensure_ascii=False),
                },
            ],
            model=model,
            temperature=0.0,
        )
        return self._extract_json_payload(repaired_text)
