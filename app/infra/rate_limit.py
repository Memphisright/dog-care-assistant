from datetime import datetime, timezone

from app.core.exceptions import AppError


class RateLimitLease:
    def __init__(self, redis_client, concurrent_key: str) -> None:
        self._redis = redis_client
        self._concurrent_key = concurrent_key
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        current = await self._redis.decr(self._concurrent_key)
        if current <= 0:
            await self._redis.delete(self._concurrent_key)


class RateLimiter:
    def __init__(
        self,
        redis_client,
        *,
        per_minute: int,
        max_concurrent: int,
    ) -> None:
        self._redis = redis_client
        self._per_minute = per_minute
        self._max_concurrent = max_concurrent

    async def acquire(self, api_key: str, client_ip: str) -> RateLimitLease:
        identity = f"{api_key}:{client_ip or 'unknown'}"
        bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        count_key = f"ratelimit:count:{identity}:{bucket}"
        concurrent_key = f"ratelimit:concurrent:{identity}"

        count = await self._redis.incr(count_key)
        if count == 1:
            await self._redis.expire(count_key, 70)
        if count > self._per_minute:
            await self._redis.decr(count_key)
            raise AppError(
                status_code=429,
                code="RATE_LIMITED",
                message="Too many requests",
                retryable=True,
            )

        concurrent = await self._redis.incr(concurrent_key)
        if concurrent == 1:
            await self._redis.expire(concurrent_key, 120)
        if concurrent > self._max_concurrent:
            await self._redis.decr(concurrent_key)
            raise AppError(
                status_code=429,
                code="RATE_LIMITED",
                message="Too many concurrent requests",
                retryable=True,
            )

        return RateLimitLease(self._redis, concurrent_key)

