from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional, List, Any, Dict
from datetime import datetime, timezone
from collections.abc import AsyncGenerator

class Message(BaseModel):
    """
    Models that represents a message in a conversation.
    Role is normalized to 'human', 'ai', 'system', and 'tool'.
    Stores thinking tokens for advanced models
    """

    role: Literal['human', 'ai', 'system', 'tool'] = Field(..., description="Role of the message sender: 'human', 'ai', 'system', or 'tool'.")
    content: str = Field(..., description="Content of the message.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the message in UTC.")
    
    
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

    
class ToolCall(BaseModel):
    """Represents a tool/function call from the LLM."""
    id: str = Field(..., description="Unique identifier for this tool call")
    name: str = Field(..., description="Name of the tool/function to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool")
    thought_signature: Optional[str] = Field(
        None, 
        description="Encrypted thinking state signature (Gemini 3+). Must be preserved and sent back for multi-turn function calling."
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


class StreamChunk(BaseModel):
    """
    Single chunk of streaming content from an LLM.
    
    Used by streaming clients to yield content progressively.
    """
    content: str = Field(default="", description="Content chunk from the LLM")
    is_finished: bool = Field(default=False, description="Whether this is the final chunk")
    finish_reason: Optional[str] = Field(None, description="Why generation stopped (stop, length, etc.)")
    thinking: Optional[str] = Field(None, description="Internal reasoning (for o1, DeepSeek-R1, etc.)")

    # If the LLM decides to call a tool mid-stream, the Client normalizes it here
    tool_calls: Optional[List[ToolCall]] = Field(
        None,
        description="Tool/function calls requested by the LLM during streaming"
    )


    
    # Provider-agnostic usage stats (tokens used) injected on the final chunk
    usage: Optional[Dict[str, int]] = Field(default=None)
    
    def __str__(self) -> str:
        return self.content
