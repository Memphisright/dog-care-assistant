import redis.asyncio as redis


def create_redis_client(redis_url: str) -> redis.Redis:
    return redis.from_url(redis_url, encoding="utf-8", decode_responses=True)

