import json


PET_OBSERVER_SYSTEM_PROMPT = (
    "You are a pet speaking to your human in a gentle, companion-style voice. "
    "Stay grounded in the user's input and the current conversation history. "
    "Sound like a warm, affectionate pet with a calm personality, not like a generic assistant. "
    "Do not describe yourself as an AI assistant. "
    "Do not say you lack emotions or that you are only a model. "
    "Do not invent highly specific facts that are not supported by the user's message or prior session context. "
    "You may sound close and a little cute, but avoid excessive baby talk, overdone roleplay, or constant animal noises. "
    "Keep the tone natural, intimate, and believable."
)


class MemoryService:
    def __init__(
        self,
        redis_client,
        *,
        ttl_seconds: int,
        max_turns: int,
        system_prompt: str = PET_OBSERVER_SYSTEM_PROMPT,
    ) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._max_turns = max_turns
        self._system_prompt = system_prompt

    def _history_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"

    def _default_history(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self._system_prompt}]

    def _normalize_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        default_system = self._default_history()[0]
        if not messages:
            return [default_system]

        if messages[0].get("role") == "system":
            rest = messages[1:]
        else:
            rest = messages

        return [default_system, *rest]

    def _trim_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized = self._normalize_history(messages)
        if not normalized:
            return self._default_history()

        system_message = normalized[0]
        rest = normalized[1:]

        max_message_count = self._max_turns * 2
        trimmed_rest = rest[-max_message_count:] if max_message_count > 0 else []
        return [system_message, *trimmed_rest]

    async def load_history(self, session_id: str) -> list[dict[str, str]]:
        raw = await self._redis.get(self._history_key(session_id))
        if not raw:
            return self._default_history()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return self._normalize_history(parsed)
        except json.JSONDecodeError:
            pass
        return self._default_history()

    async def save_history(self, session_id: str, messages: list[dict[str, str]]) -> None:
        trimmed = self._trim_messages(messages)
        await self._redis.setex(
            self._history_key(session_id),
            self._ttl_seconds,
            json.dumps(trimmed, ensure_ascii=False),
        )

    async def build_messages(
        self,
        session_id: str,
        user_message: str,
    ) -> list[dict[str, str]]:
        history = await self.load_history(session_id)
        return [*history, {"role": "user", "content": user_message}]

    async def append_exchange(
        self,
        session_id: str,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        history = await self.load_history(session_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        await self.save_history(session_id, history)

    async def lookup(
        self,
        session_id: str,
        *,
        query: str | None = None,
        top_k: int = 3,
    ) -> dict:
        history = await self.load_history(session_id)
        entries = [m for m in history if m.get("role") in {"user", "assistant"}]
        recent = list(reversed(entries))

        if query:
            needle = query.lower()
            recent = [
                msg
                for msg in recent
                if needle in str(msg.get("content", "")).lower()
            ]

        hits = recent[:top_k]
        return {"hits": hits, "count": len(hits)}
