"""Public protocols for Syndicate consumers."""

from typing import Protocol, runtime_checkable
from collections.abc import AsyncGenerator

from .communication_models import StreamChunk


@runtime_checkable
class AgentInterface(Protocol):
    """Minimal operational contract for a Syndicate agent runtime."""

    async def invoke(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
    ) -> str:
        ...

    async def stream(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
        include_thinking: bool = False,
    ) -> AsyncGenerator[StreamChunk, None]:
        ...

    def invoke_sync(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
    ) -> str:
        ...
