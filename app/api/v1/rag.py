from fastapi import APIRouter, Depends, Request

from app.core.security import verify_api_key
from app.schemas.rag import RagQueryRequest, RagQueryResponse

router = APIRouter()


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post("/api/v1/rag/query", response_model=RagQueryResponse)
async def rag_query(
    body: RagQueryRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
) -> RagQueryResponse:
    request_id = request.state.request_id
    rate_limiter = request.app.state.rate_limiter
    rag_service = request.app.state.rag_service

    lease = await rate_limiter.acquire(api_key, _client_ip(request))
    try:
        answer, sources = await rag_service.query(
            user_message=body.user_message,
            top_k=body.top_k,
        )
        return RagQueryResponse(
            answer=answer,
            sources=sources,
            request_id=request_id,
        )
    finally:
        await lease.release()
