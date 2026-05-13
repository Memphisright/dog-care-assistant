from uuid import uuid4


UNLOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class ChatSessionLockManager:
    def __init__(self, redis_client, ttl_seconds: int = 30) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"lock:chat:{session_id}"

    async def acquire(self, session_id: str) -> str | None:
        token = str(uuid4())
        ok = await self._redis.set(
            self._key(session_id),
            token,
            nx=True,
            ex=self._ttl_seconds,
        )
        return token if ok else None

    async def release(self, session_id: str, token: str | None) -> None:
        if not token:
            return
        await self._redis.eval(UNLOCK_SCRIPT, 1, self._key(session_id), token)

