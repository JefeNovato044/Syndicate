"""Gemini LLM client for Google's Gemini API."""

import json
import logging
from typing import List, Optional, Tuple, Dict, Any
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types
from google.oauth2 import service_account

from ..communication_models import Message, ChatResponse, ToolCall, StreamChunk
from .base import Client

logger = logging.getLogger(__name__)


class GeminiClient(Client):
    """Gemini LLM client with support for both API key and service account authentication.
    
    Supports:
    - Gemini API (via API key)
    - Vertex AI (via service account credentials)
    - Application Default Credentials (ADC)
    
    Example:
        # Using API key
        client = GeminiClient(
            model_name="gemini-1.5-pro",
            api_key="your-api-key"
        )
        
        # Using Vertex AI service account
        client = GeminiClient(
            model_name="gemini-1.5-pro",
            service_account_credentials="/path/to/service-account.json",
            project="your-project-id",
            location="us-central1"
        )
        
        # Using default credentials (ADC)
        client = GeminiClient(model_name="gemini-1.5-pro")
    """
    
    provider_type = "gemini"
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        service_account_credentials: Optional[str] = None,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: str = "us-central1"
    ):
        """
        Initialize Gemini client.
        
        Args:
            model_name: Name of the Gemini model to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
            temperature: Sampling temperature (0.0 to 2.0)
            service_account_credentials: Path to service account JSON file (for Vertex AI)
            api_key: Gemini API key (for direct API access)
            project: Google Cloud project ID (required for Vertex AI)
            location: Google Cloud region (default: "us-central1")
            
        Raises:
            ValueError: If no valid credentials are provided
        """
        self.model_name = model_name
        self.temperature = temperature
        self.project = project
        self.location = location
        
        # Initialize client with appropriate credentials
        if service_account_credentials:
            # Vertex AI with service account
            credentials = service_account.Credentials.from_service_account_file(
                service_account_credentials,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.client = genai.Client(
                vertexai=True,
                credentials=credentials,
                project=credentials.project_id,
                location=location
            )
            logger.info(f"Initialized Gemini client with Vertex AI credentials from {service_account_credentials}")
            
        elif api_key:
            # Direct API with API key
            self.client = genai.Client(api_key=api_key)
            logger.info("Initialized Gemini client with API key")
            
        else:
            # Use Application Default Credentials (ADC)
            try:
                self.client = genai.Client()
                logger.info("Initialized Gemini client with Application Default Credentials")
            except Exception as e:
                raise ValueError(
                    "No credentials provided. Either:\n"
                    "1. Pass 'service_account_credentials' (path to JSON file)\n"
                    "2. Pass 'api_key' (Gemini API key)\n"
                    "3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                    "4. Set up Application Default Credentials (ADC)\n"
                    f"Error: {str(e)}"
                ) from e
    
    def _decode_messages(self, messages: List[Message]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Convert our Message objects to Gemini format.
        
        Handles:
        - Text messages (user/model)
        - AI messages with tool_calls (function calls with thought_signatures)
        - Tool messages (function responses)
        
        Args:
            messages: List of Message objects
            
        Returns:
            Tuple of (gemini_messages, system_instruction)
        """
        role_mapping = {"human": "user", "ai": "model", "system": "user", "tool": "user"}
        
        chat_messages = []
        system_instruction = None
        
        for msg in messages:
            # Extract system prompt separately (Gemini uses system_instruction)
            if msg.role == "system":
                system_instruction = msg.content
                continue
            
            # Map role
            gemini_role = role_mapping.get(msg.role, "user")
            
            if not gemini_role:
                logger.warning(f"Couldn't decode message with invalid role: {msg.role}")
                continue
            
            # Handle tool (function response) messages
            if msg.role == "tool":
                # Gemini expects functionResponse in parts with the function NAME
                try:
                    # Try to parse content as JSON for structured response
                    response_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": msg.content}
                
                # tool_call_id should be the function name for Gemini compatibility
                # Format: either "function_name" or kept as-is if it's just the name
                func_name = msg.tool_call_id or "unknown"
                
                chat_messages.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": func_name,
                            "response": response_data
                        }
                    }]
                })
                continue
            
            # Handle AI messages with tool calls (function calls)
            if msg.role == "ai" and msg.tool_calls:
                parts = []
                for tc in msg.tool_calls:
                    fc_part = {
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments
                        }
                    }
                    # Preserve thought_signature for Gemini 3+ (REQUIRED for multi-turn)
                    if tc.thought_signature:
                        fc_part["thoughtSignature"] = tc.thought_signature
                    parts.append(fc_part)
                
                chat_messages.append({
                    "role": "model",
                    "parts": parts
                })
                continue
            
            # Standard text message
            chat_messages.append({
                "role": gemini_role,
                "parts": [{"text": msg.content}]
            })
        
        return chat_messages, system_instruction
    
    def _encode_response(self, raw_response) -> ChatResponse:
        """
        Convert Gemini response to our ChatResponse format.
        
        Args:
            raw_response: Raw response from Gemini API (GenerateContentResponse)
            
        Returns:
            ChatResponse object
        """
        # Initialize defaults
        content = ""
        thinking = None
        tool_calls = None
        finish_reason = None
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        thinking_tokens = None
        
        # Extract text content and thinking from candidates
        # Always iterate through parts to capture both text and thinking
        if hasattr(raw_response, 'candidates') and raw_response.candidates:
            candidate = raw_response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                text_parts = []
                thinking_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    # Extract thinking/reasoning content (for thinking models)
                    if hasattr(part, 'thought') and part.thought:
                        thinking_parts.append(str(part.thought))
                content = ''.join(text_parts)
                if thinking_parts:
                    thinking = ''.join(thinking_parts)
        
        # Extract usage metadata
        if hasattr(raw_response, 'usage_metadata'):
            usage = raw_response.usage_metadata
            prompt_tokens = getattr(usage, 'prompt_token_count', None)
            completion_tokens = getattr(usage, 'candidates_token_count', None)
            total_tokens = getattr(usage, 'total_token_count', None)
            # Extract thinking tokens if available (for thinking models)
            thinking_tokens = getattr(usage, 'thoughts_token_count', None)
        
        # Extract finish reason
        if hasattr(raw_response, 'candidates') and raw_response.candidates:
            candidate = raw_response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = str(candidate.finish_reason)
        
        # Extract tool calls (function calls) with thought signatures
        if hasattr(raw_response, 'candidates') and raw_response.candidates:
            candidate = raw_response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    # Check if this part is a function call
                    if hasattr(part, 'function_call') and part.function_call:
                        if tool_calls is None:
                            tool_calls = []
                        
                        fc = part.function_call
                        # Extract function call details
                        func_name = getattr(fc, 'name', None)
                        if func_name:
                            # Get arguments - might be dict or object
                            func_args = {}
                            if hasattr(fc, 'args') and fc.args:
                                # fc.args might be a Struct or dict
                                if isinstance(fc.args, dict):
                                    func_args = fc.args
                                else:
                                    # Convert Struct to dict
                                    try:
                                        func_args = dict(fc.args)
                                    except (TypeError, ValueError):
                                        logger.warning(f"Failed to convert function arguments to dict: {fc.args}")
                                        func_args = {}
                            
                            # Extract thought_signature (required for Gemini 3+ multi-turn function calling)
                            thought_sig = None
                            if hasattr(part, 'thought_signature') and part.thought_signature:
                                thought_sig = part.thought_signature
                            
                            # Use function name as ID for Gemini compatibility
                            # (Gemini requires name in functionResponse, not arbitrary ID)
                            tool_calls.append(ToolCall(
                                id=func_name,  # Use name as ID for Gemini
                                name=func_name,
                                arguments=func_args,
                                thought_signature=thought_sig
                            ))
        
        return ChatResponse(
            content=content,
            role="ai",
            thinking=thinking,
            thinking_tokens=thinking_tokens,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=raw_response
        )
    
    def _format_tools(self, tools: List[Any]) -> Optional[List[Any]]:
        """
        Format tools for Gemini API.
        
        Args:
            tools: List of tool objects (BaseTool instances or dicts)
            
        Returns:
            List of Gemini-format tool definitions
        """
        if not tools:
            return None

        function_declarations: List[Any] = []
        native_tools: List[Any] = []

        for tool in tools:
            if hasattr(tool, 'to_format'):
                formatted = tool.to_format(self.provider_type)
                if formatted is not None:
                    if isinstance(formatted, dict) and "name" in formatted:
                        function_declarations.append(
                            types.FunctionDeclaration(
                                name=formatted["name"],
                                description=formatted.get("description", ""),
                                parameters=formatted.get("parameters"),
                            )
                        )
                    elif isinstance(formatted, types.Tool) or callable(formatted):
                        native_tools.append(formatted)
                    else:
                        logger.warning(
                            "Skipping unsupported Gemini tool format from %s: %s",
                            getattr(tool, "name", tool.__class__.__name__),
                            type(formatted).__name__,
                        )
            elif isinstance(tool, dict) and "name" in tool:
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        parameters=tool.get("parameters"),
                    )
                )
            elif isinstance(tool, types.Tool) or callable(tool):
                native_tools.append(tool)
            elif tool is not None:
                logger.warning(
                    "Skipping unsupported Gemini tool object: %s",
                    type(tool).__name__,
                )

        result = list(native_tools)
        if function_declarations:
            result.append(types.Tool(function_declarations=function_declarations))

        return result if result else None
    
    async def chat_completion_async(
        self,
        messages: List[Message],
        system_message: Optional[Message] = None,
        image: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Async version of chat completion.
        
        Args:
            messages: List of Message objects
            system_message: System prompt message
            image: Optional image data (base64 encoded string or bytes)
            tools: Optional list of tool definitions (BaseTool instances or dicts)
            thinking_level: Optional thinking level for extended thinking ("none", "low", "medium", "high")
            **kwargs: Additional parameters for Gemini API
            
        Returns:
            ChatResponse object with standardized response
        """
        # Decode our Messages to Gemini format
        chat_messages, system_instruction = self._decode_messages(messages)
        
        # Add system message if provided
        if system_message:
            system_instruction = system_message.content
        
        # Add image to last user message if provided
        if image and chat_messages:
            # Find last user message
            for msg in reversed(chat_messages):
                if msg["role"] == "user":
                    msg["parts"].append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image
                        }
                    })
                    break
        
        # Format tools if provided
        formatted_tools = self._format_tools(tools)
        
        # Build config with conditional tools and thinking
        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": kwargs.get("temperature", self.temperature),
            "tools": formatted_tools if formatted_tools else None,
        }
        
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        
        config = types.GenerateContentConfig(**config_kwargs)
        
        # Generate response (async — native coroutine via google.genai.aio)
        raw_response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=chat_messages,
            config=config
        )
        
        # Encode to our ChatResponse format
        return self._encode_response(raw_response)
    
    async def chat_completion_stream(
        self,
        messages: List[Message],
        system_message: Optional[Message] = None,
        image: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream content chunks as they're generated from Gemini.
        
        Args:
            messages: List of Message objects
            system_message: System prompt message
            image: Optional image data (base64 encoded string or bytes)
            tools: Optional list of tool definitions (BaseTool instances or dicts)
            thinking_level: Optional thinking level for extended thinking ("none", "low", "medium", "high")
            **kwargs: Additional parameters
            
        Yields:
            StreamChunk objects with content chunks progressively
        """
        # Decode our Messages to Gemini format
        chat_messages, system_instruction = self._decode_messages(messages)
        
        # Add system message if provided
        if system_message:
            system_instruction = system_message.content
        
        # Add image to last user message if provided
        if image and chat_messages:
            # Find last user message
            for msg in reversed(chat_messages):
                if msg["role"] == "user":
                    msg["parts"].append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image
                        }
                    })
                    break
        
        # Format tools if provided
        formatted_tools = self._format_tools(tools)
        
        # Build config with conditional tools and thinking
        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": kwargs.get("temperature", self.temperature),
            "tools": formatted_tools if formatted_tools else None,
        }
        
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        
        config = types.GenerateContentConfig(**config_kwargs)
        
        # Generate response with native async streaming via google.genai.aio
        async for chunk in self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=chat_messages,
            config=config
        ):
            # Extract text content, thinking, and tool calls from chunk
            content = ""
            thinking = None
            tool_calls = None
            
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    thinking_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                        # Extract thinking/reasoning content (for thinking models)
                        if hasattr(part, 'thought') and part.thought:
                            thinking_parts.append(str(part.thought))
                        # Check for function calls in streaming
                        if hasattr(part, 'function_call') and part.function_call:
                            if tool_calls is None:
                                tool_calls = []
                            fc = part.function_call
                            func_name = getattr(fc, 'name', None)
                            if func_name:
                                func_args = {}
                                if hasattr(fc, 'args') and fc.args:
                                    if isinstance(fc.args, dict):
                                        func_args = fc.args
                                    else:
                                        try:
                                            func_args = dict(fc.args)
                                        except (TypeError, ValueError):
                                            func_args = {}
                                
                                # Extract thought_signature for multi-turn function calling
                                thought_sig = None
                                if hasattr(part, 'thought_signature') and part.thought_signature:
                                    thought_sig = part.thought_signature
                                
                                # Use function name as ID for Gemini compatibility
                                tool_calls.append(ToolCall(
                                    id=func_name,
                                    name=func_name,
                                    arguments=func_args,
                                    thought_signature=thought_sig
                                ))
                    content = ''.join(text_parts)
                    if thinking_parts:
                        thinking = ''.join(thinking_parts)
            
            # Extract finish reason
            finish_reason = None
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
            
            # Yield chunk
            yield StreamChunk(
                content=content,
                thinking=thinking,
                is_finished=(finish_reason is not None),
                finish_reason=finish_reason,
                tool_calls=tool_calls
            )
    
    def close(self):
        """Close the Gemini client resources."""
        # The genai.Client doesn't have explicit close method,
        # but we can log for clarity
        logger.debug("Gemini client closed")
    
    async def aclose(self):
        """Close the async Gemini client resources."""
        # No-op since genai.Client doesn't have explicit async close
        logger.debug("Async Gemini client closed")
