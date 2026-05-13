from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1.chat import router as chat_router
from app.api.v1.rag import router as rag_router
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, error_payload
from app.infra.locks import ChatSessionLockManager
from app.infra.rate_limit import RateLimiter
from app.infra.redis_client import create_redis_client
from app.services.embedding_service import create_embedding_provider
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.rag_service import RagService
from app.services.tool_service import ToolService


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return rid or str(uuid4())


def create_app(
    *,
    settings: Settings | None = None,
    redis_client=None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        active_redis = redis_client or create_redis_client(resolved_settings.redis_url)
        app.state.redis = active_redis
        app.state.memory_service = MemoryService(
            active_redis,
            ttl_seconds=resolved_settings.memory_ttl_seconds,
            max_turns=resolved_settings.memory_max_turns,
        )
        app.state.lock_manager = ChatSessionLockManager(
            active_redis,
            ttl_seconds=resolved_settings.lock_ttl_seconds,
        )
        app.state.rate_limiter = RateLimiter(
            active_redis,
            per_minute=resolved_settings.rate_limit_per_minute,
            max_concurrent=resolved_settings.max_concurrent_per_key,
        )
        app.state.tool_service = ToolService(app.state.memory_service)
        app.state.llm_service = LLMService(
            api_key=resolved_settings.deepseek_api_key,
            base_url=resolved_settings.deepseek_base_url,
            default_model=resolved_settings.default_model,
        )
        app.state.embedding_provider = create_embedding_provider(
            mode=resolved_settings.rag_embedding_mode,
            model_name=resolved_settings.rag_embedding_model,
        )
        app.state.knowledge_base_service = KnowledgeBaseService(
            index_path=resolved_settings.rag_index_path,
            score_threshold=resolved_settings.rag_score_threshold,
        )
        app.state.rag_service = RagService(
            embedding_provider=app.state.embedding_provider,
            knowledge_base_service=app.state.knowledge_base_service,
            llm_service=app.state.llm_service,
            answer_score_threshold=resolved_settings.rag_answer_score_threshold,
            source_score_threshold=resolved_settings.rag_source_score_threshold,
            symptom_answer_score_threshold=resolved_settings.rag_symptom_answer_score_threshold,
            symptom_source_score_threshold=resolved_settings.rag_symptom_source_score_threshold,
        )
        yield
        await app.state.llm_service.aclose()
        if redis_client is None:
            await active_redis.aclose()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id", str(uuid4()))
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(
                code=exc.code,
                message=exc.message,
                request_id=_request_id(request),
                details=exc.details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=error_payload(
                code="VALIDATION_ERROR",
                message="Request validation failed.",
                request_id=_request_id(request),
                details=exc.errors(),
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        message = str(exc.detail) if exc.detail else "HTTP error"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(
                code=f"HTTP_{exc.status_code}",
                message=message,
                request_id=_request_id(request),
            ),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=error_payload(
                code="INTERNAL_ERROR",
                message="Unexpected server error.",
                request_id=_request_id(request),
            ),
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(chat_router)
    app.include_router(rag_router)
    return app


app = create_app()
