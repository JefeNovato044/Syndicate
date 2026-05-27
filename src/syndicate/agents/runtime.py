"""Runtime-only facade for safe web/service consumption."""

from collections.abc import AsyncGenerator

from ..communication_models import StreamChunk
from ..protocols import AgentInterface


class AgentRuntime(AgentInterface):
    """Facade that exposes only operational methods from an agent."""

    def __init__(self, agent: AgentInterface):
        self._agent = agent

    async def invoke(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
    ) -> str:
        return await self._agent.invoke(user_input, owner_id=owner_id, chat_id=chat_id)

    async def stream(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
        include_thinking: bool = False,
    ) -> AsyncGenerator[StreamChunk, None]:
        async for chunk in self._agent.stream(
            user_input,
            owner_id=owner_id,
            chat_id=chat_id,
            include_thinking=include_thinking,
        ):
            yield chunk

    def invoke_sync(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
    ) -> str:
        return self._agent.invoke_sync(user_input, owner_id=owner_id, chat_id=chat_id)
    
    async def get_full_history(
        self,
        owner_id: str = "default",
        chat_id: str = "default",
        include_closed_buckets: bool = True,
        include_deleted: bool = False,
    ) -> list:
        return await self._agent.get_full_history(
            owner_id=owner_id,
            chat_id=chat_id,
            include_closed_buckets=include_closed_buckets,
            include_deleted=include_deleted,
        )