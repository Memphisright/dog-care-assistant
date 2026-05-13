from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from app.core.exceptions import AppError

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str:
    settings = request.app.state.settings
    if settings.dev_bypass_auth:
        return "dev-bypass"

    allowed_keys = settings.api_keys_set
    if not allowed_keys:
        raise AppError(
            status_code=403,
            code="AUTH_FORBIDDEN",
            message="API key auth is enabled but APP_API_KEYS is empty.",
        )

    if not api_key or api_key not in allowed_keys:
        raise AppError(
            status_code=403,
            code="AUTH_FORBIDDEN",
            message="Invalid or missing x-api-key.",
        )
    return api_key

