from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    user_message: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=8)


class RagSource(BaseModel):
    doc_id: str
    title: str
    snippet: str
    source: str
    chunk_id: str
    score: float


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[RagSource]
    request_id: str
