# clients/base.py

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
from collections.abc import AsyncGenerator

from ..communication_models import Message, ChatResponse, StreamChunk

if TYPE_CHECKING:
    from . import Client


class Client(ABC):
    """Base class for all LLM clients.
    
    All LLM client implementations should inherit from this class to enable
    type-safe declarations like:
        llm_client: Client  # Base type
        llm_client: GeminiClient  # Specific type
    
    The engine is fully async. Subclasses must implement:
        - chat_completion_async() — used by _orchestrate_invoke()
        - chat_completion_stream() — used by _orchestrate_stream()
    """
    
    provider_type: str = "base"
    
    @abstractmethod
    async def chat_completion_async(
        self,
        messages: List["Message"],
        system_message: "Message",
        tools=None,
        **kwargs
    ) -> "ChatResponse":
        """Send chat completion request to the LLM (async).
        
        This is the primary method called by the agent's orchestration engine.
        
        Args:
            messages: List of message objects
            system_message: System prompt message
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ChatResponse object with the LLM's response
        """
        pass
    
    @abstractmethod
    async def chat_completion_stream(
        self,
        messages: List["Message"],
        system_message: "Message",
        tools=None,
        **kwargs
    ) -> AsyncGenerator["StreamChunk", None]:
        """Stream content chunks as they're generated from the LLM.
        
        Args:
            messages: List of message objects
            system_message: System prompt message
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific parameters
            
        Yields:
            StreamChunk objects with content chunks progressively
            
        Example:
            async for chunk in client.chat_completion_stream(messages, system_message):
                print(chunk.content, end="", flush=True)
        """
        pass
