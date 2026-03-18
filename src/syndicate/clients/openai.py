"""OpenAI-compatible client for local models (Ollama, LM Studio, vLLM, etc.)."""

import asyncio
import json
import logging
from typing import List, Optional, Dict, Any
from collections.abc import AsyncGenerator
import httpx

from ..communication_models import Message, ChatResponse, ToolCall, StreamChunk
from .base import Client

logger = logging.getLogger(__name__)


class OpenAIClient(Client):
    """OpenAI-compatible client for local and cloud LLMs.
    
    Supports:
    - OpenAI API (GPT-4, GPT-3.5, etc.)
    - Ollama (http://localhost:11434/v1)
    - LM Studio (http://localhost:1234/v1)
    - vLLM (http://localhost:8000/v1)
    - Any OpenAI-compatible API endpoint
    
    Example:
        # Using Ollama
        client = OpenAIClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama doesn't require a real key
            model_name="llama3"
        )
        
        # Using OpenAI
        client = OpenAIClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model_name="gpt-4"
        )
    """
    
    provider_type = "openai"
    
    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy",
        model_name: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120
    ):
        """
        Initialize OpenAI-compatible client.
        
        Args:
            base_url: Base URL of the API endpoint (e.g., "http://localhost:11434/v1")
            api_key: API key (for OpenAI, Ollama uses "ollama" or any string)
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Persistent async HTTP client (lazy-initialized)
        self._async_client: Optional[httpx.AsyncClient] = None
    
    def _decode_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Convert our Message objects to OpenAI format.
        
        Args:
            messages: List of our Message objects
            
        Returns:
            List of OpenAI-format message dicts
        """
        role_mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool"
        }
        
        openai_messages = []
        for msg in messages:
            role = role_mapping.get(msg.role, "user")
            msg_dict = {"role": role, "content": msg.content or ""}

            # Assistant messages with tool calls: include tool_calls array
            if role == "assistant" and msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Tool result messages: include tool_call_id
            if role == "tool" and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id

            openai_messages.append(msg_dict)
        
        return openai_messages
    
    def _encode_response(self, raw_response: Dict[str, Any]) -> ChatResponse:
        """
        Convert OpenAI response to our ChatResponse format.
        
        Args:
            raw_response: Raw response from OpenAI API
            
        Returns:
            ChatResponse object
        """
        # Extract content
        content = ""
        if "choices" in raw_response and len(raw_response["choices"]) > 0:
            choice = raw_response["choices"][0]
            if "message" in choice:
                content = choice["message"].get("content", "")
        
        # Extract tool calls
        tool_calls = None
        if "choices" in raw_response and len(raw_response["choices"]) > 0:
            choice = raw_response["choices"][0]
            if "message" in choice and "tool_calls" in choice["message"]:
                tool_calls = []
                for tool_call in choice["message"]["tool_calls"]:
                    raw_args = tool_call.get("function", {}).get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            raw_args = {}
                    tool_calls.append(ToolCall(
                        id=tool_call.get("id", ""),
                        name=tool_call.get("function", {}).get("name", ""),
                        arguments=raw_args
                    ))
        
        # Extract finish reason
        finish_reason = None
        if "choices" in raw_response and len(raw_response["choices"]) > 0:
            finish_reason = raw_response["choices"][0].get("finish_reason")
        
        # Extract usage metadata
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if "usage" in raw_response:
            usage = raw_response["usage"]
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        
        return ChatResponse(
            content=content,
            role="ai",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=raw_response
        )
    
    def _format_tools(self, tools: List[Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Format tools for OpenAI API.
        
        Args:
            tools: List of tool objects (BaseTool instances or dicts)
            
        Returns:
            List of OpenAI-format tool definitions
        """
        if not tools:
            return None
        
        openai_tools = []
        for tool in tools:
            # Handle BaseTool instances
            if hasattr(tool, 'to_format'):
                tool_dict = tool.to_format(self.provider_type)
                if isinstance(tool_dict, dict):
                    openai_tools.append(tool_dict)
            # Handle dict directly
            elif isinstance(tool, dict):
                openai_tools.append(tool)
        
        return openai_tools if openai_tools else None
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the persistent async HTTP client."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._async_client

    async def chat_completion_async(
        self,
        messages: List[Message],
        system_message: Message,
        tools=None,
        **kwargs
    ) -> ChatResponse:
        """
        Async version of chat completion.
        
        Args:
            messages: List of Message objects
            system_message: System prompt message
            tools: Optional list of tool definitions
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object with standardized response
        """
        # Decode our Messages to OpenAI format
        openai_messages = self._decode_messages(messages)
        
        # Add system message if provided
        if system_message:
            openai_messages.insert(0, {
                "role": "system",
                "content": system_message.content
            })
        
        # Format tools if provided
        formatted_tools = self._format_tools(tools)
        
        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        # Add tools if provided
        if formatted_tools:
            payload["tools"] = formatted_tools
            payload["tool_choice"] = "auto"
        
        # Make async request using persistent client
        client = self._get_async_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        
        # Encode to our ChatResponse format
        return self._encode_response(response.json())
    
    async def chat_completion_stream(
        self,
        messages: List[Message],
        system_message: Message,
        tools=None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream content chunks as they're generated from OpenAI-compatible API.
        
        Args:
            messages: List of Message objects
            system_message: System prompt message
            tools: Optional list of tool definitions
            **kwargs: Additional parameters
            
        Yields:
            StreamChunk objects with content chunks progressively
        """
        # Decode our Messages to OpenAI format
        openai_messages = self._decode_messages(messages)
        
        # Add system message if provided
        if system_message:
            openai_messages.insert(0, {
                "role": "system",
                "content": system_message.content
            })
        
        # Format tools if provided
        formatted_tools = self._format_tools(tools)
        
        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True  # Enable streaming
        }
        
        # Add tools if provided
        if formatted_tools:
            payload["tools"] = formatted_tools
            payload["tool_choice"] = "auto"
        
        # Make async request with streaming using persistent client
        client = self._get_async_client()

        # Accumulator for OpenAI's incremental tool_call deltas.
        # OpenAI streams tool calls as partial fragments:
        #   {index, id?, function: {name?, arguments: "partial..."}}
        # We must concatenate argument strings by index and emit
        # complete ToolCall objects only after the stream ends.
        _tool_call_acc: Dict[int, Dict[str, Any]] = {}
        last_finish_reason: Optional[str] = None

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            
            # Parse SSE (Server-Sent Events) format
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                # SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    # Handle [DONE] marker
                    if data_str.strip() == "[DONE]":
                        # Emit accumulated tool calls as complete ToolCall objects
                        final_tool_calls = None
                        if _tool_call_acc:
                            final_tool_calls = []
                            for _idx in sorted(_tool_call_acc):
                                tc = _tool_call_acc[_idx]
                                raw_args = tc.get("arguments", "{}")
                                try:
                                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                                except json.JSONDecodeError:
                                    parsed_args = {}
                                final_tool_calls.append(
                                    ToolCall(
                                        id=tc.get("id", ""),
                                        name=tc.get("name", ""),
                                        arguments=parsed_args,
                                    )
                                )
                        yield StreamChunk(
                            content="",
                            is_finished=True,
                            finish_reason=last_finish_reason or ("tool_calls" if final_tool_calls else "stop"),
                            tool_calls=final_tool_calls,
                        )
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Extract content from delta
                        content = ""
                        thinking = ""
                        finish_reason = None

                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")
                            if finish_reason is not None:
                                last_finish_reason = finish_reason

                            if delta:
                                content = delta.get("content", "") or ""
                                thinking = delta.get("reasoning_content", "") or ""

                                # Accumulate tool call deltas by index
                                for tc_delta in delta.get("tool_calls", []):
                                    idx = tc_delta.get("index", 0)
                                    if idx not in _tool_call_acc:
                                        _tool_call_acc[idx] = {"id": "", "name": "", "arguments": ""}
                                    if tc_delta.get("id"):
                                        _tool_call_acc[idx]["id"] = tc_delta["id"]
                                    func = tc_delta.get("function", {})
                                    if func.get("name"):
                                        _tool_call_acc[idx]["name"] = func["name"]
                                    if func.get("arguments"):
                                        _tool_call_acc[idx]["arguments"] += func["arguments"]

                                role = delta.get("role", "")
                                if not content and not thinking and role and not delta.get("tool_calls"):
                                    continue

                        # Yield content/thinking chunks (tool calls emitted on [DONE])
                        if content or thinking or finish_reason:
                            yield StreamChunk(
                                content=content,
                                thinking=thinking,
                                is_finished=(finish_reason is not None and finish_reason != "tool_calls"),
                                finish_reason=finish_reason,
                            )

                    except json.JSONDecodeError:
                        # Skip invalid JSON
                        continue
    
    # -- Async context manager for proper lifecycle management --

    async def __aenter__(self) -> "OpenAIClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self):
        """Close the async HTTP client and release connections."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None
