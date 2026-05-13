from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    status_code: int
    code: str
    message: str
    retryable: bool = False
    details: Any | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def error_payload(
    *,
    code: str,
    message: str,
    request_id: str,
    details: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if details is not None:
        payload["error"]["details"] = details
    return payload

