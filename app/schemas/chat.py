from pydantic import BaseModel, Field

from app.schemas.structured import UsageInfo


class ChatStreamRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    user_message: str = Field(min_length=1, max_length=8000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    model: str | None = None


class TokenEventPayload(BaseModel):
    request_id: str
    session_id: str
    index: int
    token: str


class DoneEventPayload(BaseModel):
    request_id: str
    session_id: str
    finish_reason: str
    usage: UsageInfo | None = None


class ErrorEventPayload(BaseModel):
    request_id: str
    code: str
    message: str
    retryable: bool = False

