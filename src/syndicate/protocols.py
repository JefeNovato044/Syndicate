"""Public protocols for Syndicate consumers."""

from typing import Any, Dict, Protocol, runtime_checkable
from collections.abc import AsyncGenerator
from .communication_models import Message

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

    async def get_full_history(
        self,
        owner_id: str = "default",
        chat_id: str = "default",
        include_closed_buckets: bool = True,
        include_deleted: bool = False,
    ) -> list[Message]:
        """Retrieve the full conversation history for a chat session."""
        ...


ObserverEvent = Dict[str, Any]


@runtime_checkable
class Observer(Protocol):
    """Lifecycle observer contract for request/model/tool execution hooks."""

    async def on_request_start(self, event: ObserverEvent) -> None:
        ...

    async def on_request_end(self, event: ObserverEvent) -> None:
        ...

    async def on_model_call_start(self, event: ObserverEvent) -> None:
        ...

    async def on_model_call_end(self, event: ObserverEvent) -> None:
        ...

    async def on_tool_call_start(self, event: ObserverEvent) -> None:
        ...

    async def on_tool_call_end(self, event: ObserverEvent) -> None:
        ...

    async def on_error(self, event: ObserverEvent) -> None:
        ...
