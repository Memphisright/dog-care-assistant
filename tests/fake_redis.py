class FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._expires: dict[str, int] = {}

    async def get(self, key: str):
        return self._values.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self._values[key] = value
        self._expires[key] = ttl
        return True

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        if nx and key in self._values:
            return False
        self._values[key] = value
        if ex is not None:
            self._expires[key] = ex
        return True

    async def incr(self, key: str):
        current = int(self._values.get(key, "0"))
        current += 1
        self._values[key] = str(current)
        return current

    async def decr(self, key: str):
        current = int(self._values.get(key, "0"))
        current -= 1
        self._values[key] = str(current)
        return current

    async def expire(self, key: str, ttl: int):
        self._expires[key] = ttl
        return True

    async def delete(self, key: str):
        existed = 1 if key in self._values else 0
        self._values.pop(key, None)
        self._expires.pop(key, None)
        return existed

    async def eval(self, script: str, numkeys: int, key: str, token: str):
        if self._values.get(key) == token:
            self._values.pop(key, None)
            self._expires.pop(key, None)
            return 1
        return 0

    async def aclose(self):
        return None

