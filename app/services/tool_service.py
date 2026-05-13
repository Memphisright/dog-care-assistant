from app.schemas.structured import ToolCallRecord
from app.services.memory_service import MemoryService


class ToolService:
    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service

    async def memory_lookup(
        self,
        *,
        session_id: str,
        query: str | None,
        top_k: int = 3,
    ) -> ToolCallRecord:
        output = await self._memory_service.lookup(
            session_id,
            query=query,
            top_k=top_k,
        )
        return ToolCallRecord(
            tool_name="memory_lookup",
            input={"session_id": session_id, "query": query, "top_k": top_k},
            output=output,
        )

