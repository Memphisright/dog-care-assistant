import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.exceptions import AppError
from app.core.security import verify_api_key
from app.schemas.chat import (
    ChatStreamRequest,
    DoneEventPayload,
    ErrorEventPayload,
    TokenEventPayload,
)
from app.schemas.structured import (
    ChatStructuredRequest,
    ChatStructuredResponse,
)

router = APIRouter()


def _to_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post("/api/v1/chat/stream")
async def chat_stream(
    body: ChatStreamRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    request_id = request.state.request_id
    settings = request.app.state.settings

    rate_limiter = request.app.state.rate_limiter
    lock_manager = request.app.state.lock_manager
    memory_service = request.app.state.memory_service
    llm_service = request.app.state.llm_service

    lease = await rate_limiter.acquire(api_key, _client_ip(request))
    lock_token = await lock_manager.acquire(body.session_id)
    if not lock_token:
        await lease.release()
        raise AppError(
            status_code=409,
            code="SESSION_LOCKED",
            message="Another request is already running for this session.",
            retryable=True,
        )

    try:
        messages = await memory_service.build_messages(body.session_id, body.user_message)
    except Exception:
        await lock_manager.release(body.session_id, lock_token)
        await lease.release()
        raise

    temperature = body.temperature if body.temperature is not None else 0.7

    async def event_generator() -> AsyncIterator[str]:
        full_response = ""
        token_index = 0
        try:
            async for token in llm_service.stream_completion(
                messages=messages,
                model=body.model,
                temperature=temperature,
            ):
                full_response += token
                payload = TokenEventPayload(
                    request_id=request_id,
                    session_id=body.session_id,
                    index=token_index,
                    token=token,
                ).model_dump()
                token_index += 1
                yield _to_sse("token", payload)

            await memory_service.append_exchange(
                body.session_id,
                user_message=body.user_message,
                assistant_message=full_response,
            )
            done_payload = DoneEventPayload(
                request_id=request_id,
                session_id=body.session_id,
                finish_reason="stop",
                usage=None,
            ).model_dump()
            yield _to_sse("done", done_payload)
        except AppError as exc:
            error_payload = ErrorEventPayload(
                request_id=request_id,
                code=exc.code,
                message=exc.message,
                retryable=exc.retryable,
            ).model_dump()
            yield _to_sse("error", error_payload)
        except Exception:
            error_payload = ErrorEventPayload(
                request_id=request_id,
                code="INTERNAL_ERROR",
                message="Unexpected server error.",
                retryable=False,
            ).model_dump()
            yield _to_sse("error", error_payload)
        finally:
            await lock_manager.release(body.session_id, lock_token)
            await lease.release()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/v1/chat/structured", response_model=ChatStructuredResponse)
async def chat_structured(
    body: ChatStructuredRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
) -> ChatStructuredResponse:
    _ = request.app.state.settings
    request_id = request.state.request_id

    rate_limiter = request.app.state.rate_limiter
    lock_manager = request.app.state.lock_manager
    memory_service = request.app.state.memory_service
    tool_service = request.app.state.tool_service
    llm_service = request.app.state.llm_service

    lease = await rate_limiter.acquire(api_key, _client_ip(request))
    lock_token = await lock_manager.acquire(body.session_id)
    if not lock_token:
        await lease.release()
        raise AppError(
            status_code=409,
            code="SESSION_LOCKED",
            message="Another request is already running for this session.",
            retryable=True,
        )

    try:
        messages = await memory_service.build_messages(body.session_id, body.user_message)
        memory_tool_call = await tool_service.memory_lookup(
            session_id=body.session_id,
            query=body.user_message,
            top_k=3,
        )
        tool_calls = [memory_tool_call]
        tool_context = {"memory_lookup": memory_tool_call.output}

        answer, structured_output, usage = await llm_service.complete_structured(
            task_type=body.task_type,
            messages=messages,
            tool_context=tool_context,
            model=body.model,
        )
        await memory_service.append_exchange(
            body.session_id,
            user_message=body.user_message,
            assistant_message=answer,
        )
        return ChatStructuredResponse(
            answer=answer,
            structured_output=structured_output,
            tool_calls=tool_calls,
            usage=usage,
            request_id=request_id,
        )
    finally:
        await lock_manager.release(body.session_id, lock_token)
        await lease.release()


@router.post("/chat_with_memory", deprecated=True)
async def chat_with_memory(
    body: ChatStreamRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    rate_limiter = request.app.state.rate_limiter
    lock_manager = request.app.state.lock_manager
    memory_service = request.app.state.memory_service
    llm_service = request.app.state.llm_service

    lease = await rate_limiter.acquire(api_key, _client_ip(request))
    lock_token = await lock_manager.acquire(body.session_id)
    if not lock_token:
        await lease.release()
        raise AppError(
            status_code=409,
            code="SESSION_LOCKED",
            message="Another request is already running for this session.",
            retryable=True,
        )

    try:
        messages = await memory_service.build_messages(body.session_id, body.user_message)
    except Exception:
        await lock_manager.release(body.session_id, lock_token)
        await lease.release()
        raise

    temperature = body.temperature if body.temperature is not None else 0.7

    async def legacy_generator() -> AsyncIterator[str]:
        full_response = ""
        try:
            async for token in llm_service.stream_completion(
                messages=messages,
                model=body.model,
                temperature=temperature,
            ):
                full_response += token
                yield token
            await memory_service.append_exchange(
                body.session_id,
                user_message=body.user_message,
                assistant_message=full_response,
            )
        except AppError as exc:
            yield f"\n[ERROR] {exc.message}"
        except Exception:
            yield "\n[ERROR] Unexpected server error."
        finally:
            await lock_manager.release(body.session_id, lock_token)
            await lease.release()

    return StreamingResponse(legacy_generator(), media_type="text/event-stream")

