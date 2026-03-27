"""Runtime-interface usage pattern for web/service integrations.

This example shows how to expose only Syndicate's operational methods
(invoke/stream/invoke_sync) to request handlers by using AgentInterface.
"""

from collections.abc import AsyncGenerator

from syndicate import AgentInterface
from syndicate.agents import GenericAgent


class ChatService:
    """Service layer that depends on the runtime contract only."""

    def __init__(self, runtime: AgentInterface):
        self._runtime = runtime

    async def ask(self, message: str, owner_id: str, chat_id: str) -> str:
        return await self._runtime.invoke(message, owner_id=owner_id, chat_id=chat_id)

    async def stream(self, message: str, owner_id: str, chat_id: str) -> AsyncGenerator[str, None]:
        async for chunk in self._runtime.stream(message, owner_id=owner_id, chat_id=chat_id):
            if chunk.content:
                yield chunk.content


def build_chat_service(agent: GenericAgent) -> ChatService:
    """Create a service boundary that hides mutable setup methods."""
    runtime = agent.as_runtime()
    return ChatService(runtime)


# Test-friendly mock shape
class MockRuntime:
    async def invoke(self, user_input: str, owner_id: str = "default", chat_id: str = "default") -> str:
        return f"mock:{owner_id}:{chat_id}:{user_input}"

    async def stream(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
        include_thinking: bool = False,
    ):
        yield type("Chunk", (), {"content": f"mock-stream:{user_input}"})

    def invoke_sync(self, user_input: str, owner_id: str = "default", chat_id: str = "default") -> str:
        return f"mock-sync:{owner_id}:{chat_id}:{user_input}"
