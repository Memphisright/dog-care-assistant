from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    daily_card = "daily_card"
    risk_analysis = "risk_analysis"


class ChatStructuredRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    user_message: str = Field(min_length=1, max_length=8000)
    task_type: TaskType
    model: str | None = None


class DailyCardOutput(BaseModel):
    date: str
    summary: str
    focus: str
    priorities: list[str]
    risks: list[str]
    action_items: list[str]


class RiskItem(BaseModel):
    risk: str
    level: Literal["low", "medium", "high"]
    impact: str
    mitigation: str


class RiskAnalysisOutput(BaseModel):
    overall_level: Literal["low", "medium", "high"]
    summary: str
    key_risks: list[RiskItem]
    recommendations: list[str]


class ToolCallRecord(BaseModel):
    tool_name: str
    input: dict[str, Any]
    output: dict[str, Any]


class UsageInfo(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatStructuredResponse(BaseModel):
    answer: str
    structured_output: DailyCardOutput | RiskAnalysisOutput
    tool_calls: list[ToolCallRecord]
    usage: UsageInfo | None = None
    request_id: str

