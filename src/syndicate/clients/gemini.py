"""Gemini LLM client for Google's Gemini API."""

import io
import json
import logging
from typing import List, Optional, Tuple, Dict, Any, Union
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types
from google.oauth2 import service_account

from ..communication_models import Message, ChatResponse, ToolCall, StreamChunk, File
from .base import Client

logger = logging.getLogger(__name__)


class UploadedFile:
    """A reference to a file stored server-side via the Gemini File API.

    Do not construct this directly — it is returned by
    :meth:`GeminiClient.upload_file` after a successful upload.
    The ``uri`` is valid for 48 hours after upload.

    Attributes:
        uri: Gemini-assigned file URI (e.g. ``"files/abc123"``).
        mime_type: MIME type of the uploaded file.
        name: Gemini-assigned resource name.
    """

    __slots__ = ("uri", "mime_type", "name")

    def __init__(self, uri: str, mime_type: str, name: str) -> None:
        self.uri = uri
        self.mime_type = mime_type
        self.name = name

    def __repr__(self) -> str:
        return f"UploadedFile(uri={self.uri!r}, mime_type={self.mime_type!r}, name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UploadedFile):
            return NotImplemented
        return self.uri == other.uri and self.mime_type == other.mime_type and self.name == other.name


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
        
        Orphan-safe: when a bucket rollover splits a function-call/response pair
        across buckets, the orphaned half is silently dropped so Gemini never sees
        a functionResponse without a preceding functionCall (or vice-versa).
        
        Args:
            messages: List of Message objects
            
        Returns:
            Tuple of (gemini_messages, system_instruction)
        """
        role_mapping = {"human": "user", "ai": "model", "system": "user", "tool": "user"}
        
        chat_messages = []
        system_instruction = None
        tool_call_id_to_name: Dict[str, str] = {}

        # Pre-scan: collect every tool_call_id that actually has a response in this
        # history window.  Used below to drop dangling functionCall parts whose
        # matching functionResponse was rolled into a closed/summarised bucket.
        tool_response_ids: set = {
            msg.tool_call_id
            for msg in messages
            if msg.role == "tool" and msg.tool_call_id
        }

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
                tool_call_id = msg.tool_call_id or "unknown"
                func_name = tool_call_id_to_name.get(tool_call_id)

                if func_name is None:
                    # Orphaned tool response: the matching ai(tool_calls) message is
                    # not in this history window (it was in a now-closed/summarised
                    # bucket).  Sending this to Gemini would produce a
                    # functionResponse turn with no preceding functionCall turn,
                    # triggering a 400 INVALID_ARGUMENT.  Drop it instead.
                    logger.warning(
                        "_decode_messages: dropping orphaned functionResponse "
                        "(tool_call_id=%s not found in current history window)",
                        tool_call_id,
                    )
                    continue

                try:
                    response_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": msg.content}
                
                chat_messages.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "id": tool_call_id,
                            "name": func_name,
                            "response": response_data
                        }
                    }]
                })
                continue
            
            # Handle AI messages with tool calls (function calls)
            if msg.role == "ai" and msg.tool_calls:
                # Only include tool calls whose response is present in this history
                # window.  If a rollover stored the matching tool response in a
                # different (now-closed) bucket, including the functionCall here
                # without a following functionResponse would also break Gemini's
                # strict turn-ordering rules.
                matched_calls = [tc for tc in msg.tool_calls if tc.id in tool_response_ids]

                if not matched_calls:
                    # Every tool call in this turn is dangling — no responses in
                    # the visible history.  Skip the entire model turn.
                    logger.warning(
                        "_decode_messages: dropping ai message with %d dangling "
                        "tool_call(s) (no matching responses in history window)",
                        len(msg.tool_calls),
                    )
                    continue

                parts = []
                for tc in matched_calls:
                    fc_part = {
                        "functionCall": {
                            "id": tc.id,
                            "name": tc.name,
                            "args": tc.arguments
                        }
                    }
                    tool_call_id_to_name[tc.id] = tc.name
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
                    part_text = getattr(part, 'text', None)
                    raw_thought = getattr(part, 'thought', None)

                    # Gemini thought parts are flagged with thought=True and place
                    # the actual reasoning text in part.text.
                    is_thought_part = isinstance(raw_thought, bool) and raw_thought
                    if part_text:
                        if is_thought_part:
                            thinking_parts.append(part_text)
                        else:
                            text_parts.append(part_text)

                    # Compatibility path for mocked/non-standard payloads where
                    # the thought string itself is stored directly in `thought`.
                    if isinstance(raw_thought, str) and raw_thought:
                        thinking_parts.append(raw_thought)
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
                            
                            # Preserve provider tool call id for Gemini 3 multi-turn mapping.
                            tool_call_id = getattr(fc, 'id', None) or func_name
                            tool_calls.append(ToolCall(
                                id=tool_call_id,
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

    def _resolve_file_part(self, f: "File | UploadedFile") -> Any:
        """
        Convert a :class:`File` or :class:`UploadedFile` to a ``types.Part``.

        - :class:`File`         → ``types.Part.from_bytes()`` (inline, re-sent every request)
        - :class:`UploadedFile` → ``types.Part.from_uri()``   (server-side reference)

        Args:
            f: A :class:`File` or :class:`UploadedFile` instance.

        Returns:
            A ``types.Part`` ready for inclusion in a Gemini contents list.

        Raises:
            TypeError: If the input is not one of the expected types.
        """
        if isinstance(f, File):
            return types.Part.from_bytes(data=f.data, mime_type=f.mime_type)

        if isinstance(f, UploadedFile):
            return types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type)

        raise TypeError(
            f"Expected File or UploadedFile, got {type(f).__name__}."
        )

    async def upload_file(self, file: File) -> UploadedFile:
        """
        Upload a :class:`File` to the Gemini File API (up to 2 GB).

        The file is stored server-side for 48 hours.  The returned
        :class:`UploadedFile` can be passed in ``files=`` across multiple
        requests without re-uploading the bytes each time.

        Args:
            file: A :class:`File` instance containing the raw bytes,
                MIME type, and optional display name.

        Returns:
            An :class:`UploadedFile` with the provider-assigned ``uri``,
            ``mime_type``, and ``name``.

        Example::

            f = File(data=pdf_bytes, mime_type="application/pdf")
            uploaded = await client.upload_file(f)
            response = await client.chat_completion_async(messages, files=[uploaded])
        """
        config = types.UploadFileConfig(mime_type=file.mime_type, display_name=file.name)
        raw = await self.client.aio.files.upload(
            file=io.BytesIO(file.data), config=config
        )
        return UploadedFile(uri=raw.uri, mime_type=raw.mime_type, name=raw.name)

    async def chat_completion_async(
        self,
        messages: List[Message],
        system_message: Optional[Message] = None,
        image: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[str] = None,
        include_thoughts: Optional[bool] = None,
        files: Optional[List[Union[File, UploadedFile]]] = None,
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
            include_thoughts: Optional flag to request thought text in the response
            files: Optional list of :class:`File` or :class:`UploadedFile` instances
                to attach to the last user message. Use :class:`File` for inline data
                (≤100 MB, ≤50 MB PDFs) or :class:`UploadedFile` returned by
                :meth:`upload_file` for server-side references (up to 2 GB, 48h TTL).
            **kwargs: Additional parameters for Gemini API
            
        Returns:
            ChatResponse object with standardized response
        """
        # Decode our Messages to Gemini format
        chat_messages, system_instruction = self._decode_messages(messages)
        
        # Add system message if provided
        if system_message:
            system_instruction = system_message.content
        
        # Collect extra parts (legacy image + new files) and inject into the last user message
        _extra_parts: List[Any] = []
        if image:
            _extra_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image}})
        if files:
            for f in files:
                _extra_parts.append(self._resolve_file_part(f))
        if _extra_parts and chat_messages:
            for msg in reversed(chat_messages):
                if msg["role"] == "user":
                    msg["parts"].extend(_extra_parts)
                    break
        
        # Format tools if provided
        formatted_tools = self._format_tools(tools)
        
        # Build config with conditional tools and thinking
        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": kwargs.get("temperature", self.temperature),
            "tools": formatted_tools if formatted_tools else None,
        }
        
        thinking_config_kwargs = {}
        if thinking_level is not None:
            thinking_config_kwargs["thinking_level"] = thinking_level
        if include_thoughts is not None:
            thinking_config_kwargs["include_thoughts"] = include_thoughts
        if thinking_config_kwargs:
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_config_kwargs)
        
        # Forward any extra provider-specific kwargs (e.g. top_p, top_k, max_output_tokens)
        _known_kwargs = {"temperature"}
        for key, value in kwargs.items():
            if key not in _known_kwargs:
                config_kwargs[key] = value
        
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
        include_thoughts: Optional[bool] = None,
        files: Optional[List[Union[File, UploadedFile]]] = None,
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
            include_thoughts: Optional flag to request thought text in the response
            files: Optional list of :class:`File` or :class:`UploadedFile` instances
                to attach to the last user message. Use :class:`File` for inline data
                (≤100 MB, ≤50 MB PDFs) or :class:`UploadedFile` returned by
                :meth:`upload_file` for server-side references (up to 2 GB, 48h TTL).
            **kwargs: Additional parameters
            
        Yields:
            StreamChunk objects with content chunks progressively
        """
        # Decode our Messages to Gemini format
        chat_messages, system_instruction = self._decode_messages(messages)
        
        # Add system message if provided
        if system_message:
            system_instruction = system_message.content
        
        # Collect extra parts (legacy image + new files) and inject into the last user message
        _extra_parts: List[Any] = []
        if image:
            _extra_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image}})
        if files:
            for f in files:
                _extra_parts.append(self._resolve_file_part(f))
        if _extra_parts and chat_messages:
            for msg in reversed(chat_messages):
                if msg["role"] == "user":
                    msg["parts"].extend(_extra_parts)
                    break
        
        # Format tools if provided
        formatted_tools = self._format_tools(tools)
        
        # Build config with conditional tools and thinking
        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": kwargs.get("temperature", self.temperature),
            "tools": formatted_tools if formatted_tools else None,
        }
        
        thinking_config_kwargs = {}
        if thinking_level is not None:
            thinking_config_kwargs["thinking_level"] = thinking_level
        if include_thoughts is not None:
            thinking_config_kwargs["include_thoughts"] = include_thoughts
        if thinking_config_kwargs:
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_config_kwargs)
        
        # Forward any extra provider-specific kwargs (e.g. top_p, top_k, max_output_tokens)
        _known_kwargs = {"temperature"}
        for key, value in kwargs.items():
            if key not in _known_kwargs:
                config_kwargs[key] = value
        
        config = types.GenerateContentConfig(**config_kwargs)
        
        # Generate response with native async streaming via google.genai.aio.
        # generate_content_stream() is an async method returning an async iterable
        # and must be awaited before iteration.
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=chat_messages,
            config=config
        ):
            # Extract text content, thinking, and tool calls from chunk
            content = ""
            thinking = None
            thinking_tokens = None
            tool_calls = None
            
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    thinking_parts = []
                    for part in candidate.content.parts:
                        part_text = getattr(part, 'text', None)
                        raw_thought = getattr(part, 'thought', None)

                        # Gemini thought parts are flagged with thought=True and place
                        # the actual reasoning text in part.text.
                        is_thought_part = isinstance(raw_thought, bool) and raw_thought
                        if part_text:
                            if is_thought_part:
                                thinking_parts.append(part_text)
                            else:
                                text_parts.append(part_text)

                        # Compatibility path for mocked/non-standard payloads where
                        # the thought string itself is stored directly in `thought`.
                        if isinstance(raw_thought, str) and raw_thought:
                            thinking_parts.append(raw_thought)
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
                                tool_call_id = getattr(fc, 'id', None) or func_name
                                tool_calls.append(ToolCall(
                                    id=tool_call_id,
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

            # Extract chunk-level usage metadata when available.
            usage = None
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata is not None:
                usage_meta = chunk.usage_metadata
                usage = {}

                prompt_tokens = getattr(usage_meta, 'prompt_token_count', None)
                if isinstance(prompt_tokens, (int, float)) and not isinstance(prompt_tokens, bool):
                    usage["prompt_tokens"] = int(prompt_tokens)

                completion_tokens = getattr(usage_meta, 'candidates_token_count', None)
                if isinstance(completion_tokens, (int, float)) and not isinstance(completion_tokens, bool):
                    usage["completion_tokens"] = int(completion_tokens)

                total_tokens = getattr(usage_meta, 'total_token_count', None)
                if isinstance(total_tokens, (int, float)) and not isinstance(total_tokens, bool):
                    usage["total_tokens"] = int(total_tokens)

                raw_thinking_tokens = getattr(usage_meta, 'thoughts_token_count', None)
                if isinstance(raw_thinking_tokens, (int, float)) and not isinstance(raw_thinking_tokens, bool):
                    thinking_tokens = int(raw_thinking_tokens)
                    usage["thinking_tokens"] = thinking_tokens

                if not usage:
                    usage = None
            
            # Yield chunk
            yield StreamChunk(
                content=content,
                thinking=thinking,
                thinking_tokens=thinking_tokens,
                is_finished=(finish_reason is not None),
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                usage=usage,
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
