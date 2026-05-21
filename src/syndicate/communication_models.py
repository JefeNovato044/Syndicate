import json

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional, List, Any, Dict
from datetime import datetime, timezone

class ToolCall(BaseModel):
    """Represents a tool/function call from the LLM."""
    id: str = Field(..., description="Unique identifier for this tool call")
    name: str = Field(..., description="Name of the tool/function to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool")
    thought_signature: Optional[bytes | str] = Field(
        None, 
        description="Encrypted thinking state signature (Gemini 3+). Must be preserved and sent back for multi-turn function calling."
    )

class Message(BaseModel):
    """
    Models that represents a message in a conversation.
    Role is normalized to 'human', 'ai', 'system', and 'tool'.
    Stores thinking tokens for advanced models
    """

    role: Literal['human', 'ai', 'system', 'tool'] = Field(..., description="Role of the message sender: 'human', 'ai', 'system', or 'tool'.")
    content: str = Field(..., description="Content of the message.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the message in UTC.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the message.")
    
    
    # AI-Specific: When the model initiates an action
    tool_calls: Optional[List[ToolCall]] = Field(
        None, 
        description="List of ToolCall objects (only for role='ai')."
    )
    thinking: Optional[str] = Field(
        None, 
        description="Internal reasoning/thinking process (for models that support it)."
    )
    thinking_tokens: Optional[int] = Field(
        None, 
        description="Number of tokens used in thinking process."
        )
    
    # Tool-Specific: When the framework provides an answer
    tool_call_id: Optional[str] = Field(
        None, 
        description="ID matching the tool result back to the AI (only for role='tool')."
    )

    @field_validator('role', mode='before')
    @classmethod
    def normalize_role(cls, role: str) -> str:
        """Normalizes synonyms (user, assistant, etc.) to core Syndicate roles."""
        if isinstance(role, str):
            r = role.lower().strip()
            if r in ('user', 'person', 'human'): 
                return 'human'
            if r in ('model', 'bot', 'agent', 'assistant', 'ai'): 
                return 'ai'
            if r in ('system', 'sys', 'instruction', 'developer'): 
                return 'system'
            if r in ('tool', 'function', 'observation'): 
                return 'tool'
        
        raise ValueError(f"Invalid role: {role}. Must be 'human', 'ai', 'system', or 'tool'")
    
    @model_validator(mode='after')
    def validate_role_integrity(self) -> 'Message':
        """Enforces role-specific field constraints to prevent invalid history states."""
        
        if self.role == 'ai' and self.tool_call_id:
            raise ValueError("An 'ai' message cannot have a 'tool_call_id'. Use 'tool_calls' instead.")
            
        if self.role == 'tool':
            if not self.tool_call_id:
                raise ValueError("A 'tool' role message MUST have a 'tool_call_id' to map to a request.")
            if self.tool_calls:
                raise ValueError("A 'tool' role message cannot contain nested 'tool_calls'.")
                
        if self.role in ('human', 'system') and (self.tool_calls or self.tool_call_id):
            raise ValueError(f"Role '{self.role}' cannot contain tool-related fields.")
            
        return self
    
    def is_system_msg(self) -> bool:
        if self.role == 'system':
            return True
        else:
            return False
        
class MessageBucket(BaseModel):
    """
    Represents a bucket of messages for conversation segmentation.
    
    Buckets enable "infinite memory" by breaking long conversations into
    manageable segments. When a bucket reaches its threshold, it's summarized
    and closed, and a new bucket becomes active.
    
    Workflow:
    1. Messages are added to the active bucket
    2. When threshold is reached, bucket is summarized and closed
    3. New active bucket is created
    4. Old summaries provide context for new conversations
    """
    # Identity
    bucket_id: str = Field(..., description="Unique identifier for this bucket.")
    owner_id: str = Field(..., description="User/owner identifier for multi-tenant support.")
    chat_id: str = Field(..., description="Conversation/session identifier.")
    
    # Content
    messages: List[Message] = Field(default_factory=list, description="List of Message objects in this bucket.")
    summary: Optional[str] = Field(None, description="Summary of the bucket's conversation (set on rollover).")
    
    # State
    is_active: bool = Field(True, description="True if this bucket accepts new messages.")
    position: int = Field(0, description="Position in the conversation sequence (0 = first bucket).")
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        description="When the bucket was created."
    )
    closed_at: Optional[datetime] = Field(
        None, 
        description="When the bucket was closed (summarized and made inactive)."
    )
    
    # Metrics (useful for token-based rollover decisions)
    estimated_tokens: Optional[int] = Field(
        None, 
        description="Estimated token count of all messages (optional, for threshold checks)."
    )
    
    def message_count(self) -> int:
        """Get the number of messages in this bucket."""
        return len(self.messages)
    
    def interaction_count(self) -> int:
        """Get the number of interactions (message pairs) in this bucket."""
        return len(self.messages) // 2
    
    def add_message(self, message: Message) -> None:
        """Add a message to this bucket (if active)."""
        if not self.is_active:
            raise ValueError("Cannot add messages to a closed bucket.")
        self.messages.append(message)
    
    def close(self, summary: Optional[str] = None) -> None:
        """Close this bucket, optionally setting its summary."""
        self.is_active = False
        self.closed_at = datetime.now(timezone.utc)
        if summary:
            self.summary = summary


class ToolResultEnvelope(BaseModel):
    """Canonical tool outcome envelope used across invoke/stream flows."""

    status: Literal["success", "error"] = Field(..., description="Normalized tool execution status")
    result: Optional[Any] = Field(None, description="Tool result payload when status='success'")
    error: Optional[str] = Field(None, description="Error message when status='error'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional execution metadata")

    @model_validator(mode='after')
    def validate_status_payload(self) -> 'ToolResultEnvelope':
        """Ensure the status and payload fields remain coherent."""
        if self.status == "success" and self.error:
            raise ValueError("ToolResultEnvelope success status cannot include an error")
        if self.status == "error" and not self.error:
            raise ValueError("ToolResultEnvelope error status requires an error message")
        return self

    def to_json(self) -> str:
        """Serialize envelope to JSON for Message(role='tool')."""
        return self.model_dump_json()

    @classmethod
    def from_message_content(cls, content: Any) -> 'ToolResultEnvelope':
        """Parse JSON tool message content into a canonical envelope.

        Legacy non-JSON content is treated as a successful tool result.
        """
        if isinstance(content, dict):
            payload = content
        elif isinstance(content, str):
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                return cls(status="success", result=content)
        else:
            return cls(status="success", result=content)

        if isinstance(payload, dict) and payload.get("status") in {"success", "error"}:
            return cls(
                status=payload["status"],
                result=payload.get("result"),
                error=payload.get("error"),
                metadata=payload.get("metadata") or {},
            )

        # Legacy dict payloads are treated as successful result bodies.
        return cls(status="success", result=payload)

    @classmethod
    def from_tool_execution_result(cls, tool_name: str, raw_result: Any) -> 'ToolResultEnvelope':
        """Normalize tool execution outputs to the canonical envelope."""
        if isinstance(raw_result, Exception):
            return cls(
                status="error",
                error=str(raw_result),
                metadata={
                    "tool_name": tool_name,
                    "error_type": type(raw_result).__name__,
                },
            )

        if isinstance(raw_result, dict):
            # Already canonical envelope
            if raw_result.get("status") in {"success", "error"}:
                metadata = dict(raw_result.get("metadata") or {})
                metadata.setdefault("tool_name", tool_name)
                return cls(
                    status=raw_result["status"],
                    result=raw_result.get("result"),
                    error=raw_result.get("error"),
                    metadata=metadata,
                )

            metadata: Dict[str, Any] = {"tool_name": tool_name}
            for key in ("attempt", "max_attempts", "latency_ms", "error_type"):
                if key in raw_result and raw_result.get(key) is not None:
                    metadata[key] = raw_result.get(key)

            if raw_result.get("success"):
                return cls(
                    status="success",
                    result=raw_result.get("result"),
                    metadata=metadata,
                )

            return cls(
                status="error",
                error=raw_result.get("error") or "Unknown tool error",
                metadata=metadata,
            )

        return cls(
            status="success",
            result=raw_result,
            metadata={"tool_name": tool_name},
        )


class ChatResponse(BaseModel):
    """
    Universal response format from all LLM clients.
    Normalizes provider-specific responses to a standard format.
    """
    content: str = Field(..., description="The text content of the response")
    role: Literal["ai"] = Field(default="ai", description="Always 'ai' for responses")
    
    # Optional fields
    thinking: Optional[str] = Field(None, description="Internal reasoning (for o1, DeepSeek-R1, etc.)")
    thinking_tokens: Optional[int] = Field(None, description="Tokens used in thinking")
    
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool/function calls requested by LLM")
    
    finish_reason: Optional[str] = Field(None, description="Why generation stopped (stop, length, tool_calls, etc.)")
    
    # Usage/metrics
    prompt_tokens: Optional[int] = Field(None, description="Tokens in prompt")
    completion_tokens: Optional[int] = Field(None, description="Tokens in completion")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    
    # Raw response for advanced use
    raw_response: Optional[Any] = Field(None, description="Original provider response object")
    
    def to_message(self) -> Message:
        """Convert ChatResponse to Message for storage."""
        return Message(
            role=self.role,
            content=self.content,
            thinking=self.thinking,
            thinking_tokens=self.thinking_tokens,
            tool_calls=self.tool_calls
        )


class ToolCallEvent(BaseModel):
    """
    Event emitted by the streaming orchestrator when a tool is dispatched or completes.

    Consumers can use these to drive real-time UI step indicators without
    monkey-patching tools or polling a side-channel bus.

    Two chunks are emitted per tool call:
      1. status="start"   — immediately before execution begins
      2. status="success" or status="error" — immediately after execution completes

    ``tool_call_id`` matches the LLM's original ToolCall.id, enabling correlation
    when multiple tools run concurrently.
    """
    tool_call_id: str = Field(..., description="ID of the originating ToolCall (for correlation)")
    tool_name: str = Field(..., description="Name of the tool being called")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Optional[Any] = Field(None, description="Tool result (set on success/error)")
    error: Optional[str] = Field(None, description="Error message if status='error'")
    status: Literal["start", "success", "error"] = Field(..., description="Lifecycle phase of this event")


class StreamChunk(BaseModel):
    """
    Single chunk of streaming content from an LLM.
    
    Used by streaming clients to yield content progressively.
    """
    content: str = Field(default="", description="Content chunk from the LLM")
    is_finished: bool = Field(default=False, description="Whether this is the final chunk")
    finish_reason: Optional[str] = Field(None, description="Why generation stopped (stop, length, etc.)")
    thinking: Optional[str] = Field(None, description="Internal reasoning (for o1, DeepSeek-R1, etc.)")
    thinking_tokens: Optional[int] = Field(None, description="Tokens used in thinking for this stream response")

    # If the LLM decides to call a tool mid-stream, the Client normalizes it here
    tool_calls: Optional[List[ToolCall]] = Field(
        None,
        description="Tool/function calls requested by the LLM during streaming"
    )

    # Emitted during tool dispatch phases — None for normal content/thinking chunks
    tool_call: Optional[ToolCallEvent] = Field(
        None,
        description="Tool lifecycle event (start/success/error). None for content chunks."
    )

    # Provider-agnostic usage stats (tokens used) injected on the final chunk
    usage: Optional[Dict[str, int]] = Field(default=None)
    
    def __str__(self) -> str:
        if self.content:
            return self.content
        if self.thinking:
            return self.thinking
        return ""


# ============================================================================
# Agent2Agent (A2A) Protocol Models (FR-005)
# Aligned with A2A v1.0.0 Specification for multi-agent interoperability.
# ============================================================================

class A2ASupportedInterface(BaseModel):
    """Network bindings where the agent can be reached."""
    url: str = Field(..., description="Endpoint URL for the A2A binding.")
    protocolBinding: Literal["JSONRPC", "GRPC", "HTTP+JSON"] = Field(
        default="JSONRPC", 
        description="The transport protocol."
    )
    protocolVersion: str = Field(default="1.0", description="A2A protocol version.")

class A2ACapabilities(BaseModel):
    """Flags indicating supported interaction modes."""
    streaming: bool = Field(default=True, description="Supports Server-Sent Events (SSE) streaming.")
    pushNotifications: bool = Field(default=False, description="Supports async webhook callbacks.")
    extendedAgentCard: bool = Field(default=False, description="Has private, authenticated capabilities.")

class A2ASkill(BaseModel):
    """A specific capability or tool this agent exposes."""
    id: str = Field(..., description="Unique identifier for the skill (e.g., tool name).")
    name: str = Field(..., description="Human-readable name.")
    description: str = Field(..., description="Detailed description of what the skill does.")
    inputModes: List[str] = Field(
        default_factory=lambda: ["application/json"], 
        description="MIME types accepted as input."
    )
    outputModes: List[str] = Field(
        default_factory=lambda: ["application/json"], 
        description="MIME types returned."
    )
    tags: Optional[List[str]] = Field(None, description="Categorization tags.")
    examples: Optional[List[str]] = Field(None, description="Example prompts to trigger this skill.")

class A2AAgentCard(BaseModel):
    """
    Standard A2A v1.0 Agent Card (Manifest).
    Acts as the public discovery document for the agent.
    """
    name: str = Field(..., description="Display name of the agent.")
    description: str = Field(..., description="Primary purpose of the agent.")
    version: str = Field(default="1.0.0", description="Semantic version of the agent deployment.")
    
    # Metadata
    iconUrl: Optional[str] = Field(None, description="URL to an avatar/icon.")
    documentationUrl: Optional[str] = Field(None, description="URL to human-readable docs.")
    
    # Networking & Discovery
    supportedInterfaces: List[A2ASupportedInterface] = Field(
        default_factory=list,
        description="Endpoints to reach the agent."
    )
    capabilities: A2ACapabilities = Field(
        default_factory=A2ACapabilities,
        description="Core interaction capabilities."
    )
    
    # Data Formats
    defaultInputModes: List[str] = Field(
        default_factory=lambda: ["text/plain", "application/json"],
        description="Default MIME types accepted."
    )
    defaultOutputModes: List[str] = Field(
        default_factory=lambda: ["text/plain", "application/json"],
        description="Default MIME types returned."
    )
    
    # Tool/Capability Surface Area
    skills: List[A2ASkill] = Field(
        default_factory=list,
        description="The publicized tools/skills this agent can perform."
    )
    
    def to_json(self) -> str:
        """Serialize the Agent Card to A2A compliant JSON."""
        return self.model_dump_json(exclude_none=True)
