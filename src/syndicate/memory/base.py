from abc import ABC, abstractmethod
from typing import List, Optional, Any
from .summarizers import Summarizer
from ..communication_models import MessageBucket, Message

class BaseChatMemory(ABC):
    """
    Abstract base class for sequential conversation history.
    
    Handles time-ordered message storage for maintaining conversation context.
    Think of this as "short-term memory" - what was just discussed.
    
    Features:
    - Multi-tenant support (owner_id, chat_id)
    - Optional bucket-based rollover for long conversations
    - Automatic summarization on rollover
    - Hierarchical context (active bucket + summaries of past buckets)
    
    Implementations: InMemory, MongoDB, Redis, PostgreSQL, etc.
    """

    def __init__(
        self,
        rollover_enabled: bool = False,
        max_interactions_per_bucket: int = 10,
        max_tokens_per_bucket: Optional[int] = None,
        summarizer: Optional["Summarizer"] = None,
        preserve_closed_buckets: bool = True,
        supports_summarization: bool = True,
        soft_delete: bool = True,
    ):
        """
        Initialize chat memory with optional rollover configuration.
        
        Args:
            rollover_enabled: Enable automatic rollover to new message buckets
                when thresholds are reached.
            max_interactions_per_bucket: Maximum number of interactions
                (user + ai message pairs) per bucket before rollover.
            max_tokens_per_bucket: Optional token-based threshold for rollover.
                If set, rollover occurs when estimated tokens exceed this limit.
            summarizer: Agent or Callable for summarizing buckets on rollover.
                Required if rollover_enabled=True and supports_summarization=True.
                Use create_default_summarizer(llm_client) for simple cases.
            preserve_closed_buckets: If True, closed buckets are kept in storage.
                If False, only the summary is preserved and messages are deleted.
            supports_summarization: If False, rollover will be disabled or
                summarizer validation will be skipped. Set to False for memory
                types that don't support summarization (e.g., LocalMemory).
            soft_delete: If True, deletion APIs will mark messages as deleted
                and hide them from history. If False, deletion APIs will remove
                messages permanently.
        
        Raises:
            ValueError: If rollover_enabled=True and supports_summarization=True
                but no summarizer is provided, or if thresholds are invalid.
        """
        self.rollover_enabled = rollover_enabled
        self.max_interactions_per_bucket = max_interactions_per_bucket
        self.max_tokens_per_bucket = max_tokens_per_bucket
        self.summarizer = summarizer
        self.preserve_closed_buckets = preserve_closed_buckets
        self.supports_summarization = supports_summarization
        self.soft_delete = soft_delete
        
        # Validation - only required if rollover is enabled AND this memory type
        # supports summarization
        if self.rollover_enabled and self.supports_summarization:
            if self.summarizer is None:
                raise ValueError(
                    "rollover_enabled=True requires a summarizer. "
                    "Provide an Agent or use create_default_summarizer(your_llm_client)"
                )
            if not isinstance(self.max_interactions_per_bucket, int) or self.max_interactions_per_bucket < 1:
                raise ValueError(
                    "max_interactions_per_bucket must be a positive integer >= 1"
                )
            if self.max_tokens_per_bucket is not None:
                if not isinstance(self.max_tokens_per_bucket, int) or self.max_tokens_per_bucket < 100:
                    raise ValueError(
                        "max_tokens_per_bucket must be a positive integer >= 100"
                    )
    
    # =========================================================================
    # mind ABSTRACT METHODS - Must be implemented by all subclasses
    # =========================================================================
    
    @abstractmethod
    async def add_message(
        self, 
        message: Message, 
        owner_id: str, 
        chat_id: str,
        **kwargs
    ) -> None:
        """
        Add a message to the active bucket of a conversation.
        
        If rollover_enabled is True, implementations should:
        1. Check if rollover threshold is reached via should_rollover()
        2. If yes, call rollover_history() before adding the message
        3. Add message to the active bucket
        
        Args:
            message: The Message object to store
            owner_id: User/owner identifier for multi-tenant support
            chat_id: Conversation/session identifier
            **kwargs: Implementation-specific options
        """
        pass
    
    @abstractmethod
    async def get_history(
        self, 
        owner_id: str, 
        chat_id: str, 
        limit: Optional[int] = None,
        include_context_summary: bool = True
    ) -> List[Message]:
        """
        Get conversation history from the active bucket.
        
        If include_context_summary is True and there are closed buckets,
        prepend a context message with summaries of previous buckets.
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            limit: Max messages to return from active bucket (None = all)
            include_context_summary: Include summary of closed buckets as 
                first message for context continuity
            
        Returns:
            List of Messages in chronological order
        """
        pass

    async def get_full_history(
        self,
        owner_id: str,
        chat_id: str,
        limit: Optional[int] = None,
        include_closed_buckets: bool = True,
        include_deleted: bool = False,
        include_context_summary: bool = False,
    ) -> List[Message]:
        """Get flattened history for display and auditing use cases.

        Default behavior is a safe fallback to ``get_history()`` so custom
        backends that haven't implemented full-history retrieval keep working.
        Built-in persistent backends override this to include closed buckets.

        Args:
            owner_id: User/owner identifier.
            chat_id: Conversation/session identifier.
            limit: Optional maximum number of messages to return.
            include_closed_buckets: Include closed buckets when backend supports
                bucket history traversal.
            include_deleted: Include soft-deleted messages when supported.
            include_context_summary: Optionally prepend context summary message.

        Returns:
            List of Messages in chronological order.
        """
        _ = include_closed_buckets
        _ = include_deleted
        return await self.get_history(
            owner_id=owner_id,
            chat_id=chat_id,
            limit=limit,
            include_context_summary=include_context_summary,
        )
    
    @abstractmethod
    async def clear(self, owner_id: str, chat_id: str) -> None:
        """
        Clear all messages and buckets for a specific conversation.
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
        """
        pass
    
    @abstractmethod
    async def get_message_count(self, owner_id: str, chat_id: str) -> int:
        """
        Get total number of messages in the active bucket.
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            
        Returns:
            Number of messages in active bucket
        """
        pass

    @abstractmethod
    async def delete_message(
        self,
        owner_id: str,
        chat_id: str,
        index: int = -1,
        hard_delete: Optional[bool] = None,
    ) -> bool:
        """Delete one message from active history by visible index.

        Args:
            owner_id: User/owner identifier.
            chat_id: Conversation/session identifier.
            index: Visible message index (supports negative indexing).
            hard_delete: Optional per-call override. If True, remove message
                permanently. If False, mark as deleted. If None, use
                memory-level soft_delete configuration.

        Returns:
            True when a message was deleted, False otherwise.
        """
        pass

    @abstractmethod
    async def delete_last_message(
        self,
        owner_id: str,
        chat_id: str,
        role: Optional[str] = None,
        hard_delete: Optional[bool] = None,
    ) -> bool:
        """Delete the last visible message, optionally constrained by role.

        Args:
            owner_id: User/owner identifier.
            chat_id: Conversation/session identifier.
            role: Optional normalized role filter (human/ai/system/tool).
            hard_delete: Optional per-call override. If True, remove message
                permanently. If False, mark as deleted. If None, use
                memory-level soft_delete configuration.

        Returns:
            True when a message was deleted, False otherwise.
        """
        pass

    def _should_hard_delete(self, hard_delete: Optional[bool]) -> bool:
        """Resolve effective deletion mode from call override + config."""
        if hard_delete is None:
            return not self.soft_delete
        return bool(hard_delete)
    
    # =========================================================================
    # BUCKET MANAGEMENT - Abstract methods for bucket operations
    # =========================================================================
    
    @abstractmethod
    async def get_active_bucket(
        self, 
        owner_id: str, 
        chat_id: str
    ) -> Optional["MessageBucket"]:
        """
        Get the currently active bucket for a conversation.
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            
        Returns:
            The active MessageBucket, or None if no bucket exists
        """
        pass
    
    @abstractmethod
    async def create_bucket(
        self, 
        owner_id: str, 
        chat_id: str,
        position: Optional[int] = None
    ) -> "MessageBucket":
        """
        Create a new active bucket for a conversation.
        
        If there's an existing active bucket, it should be marked inactive first.
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            position: Optional position index (auto-incremented if not provided)
            
        Returns:
            The newly created MessageBucket
        """
        pass
    
    @abstractmethod
    async def get_bucket_summaries(
        self, 
        owner_id: str, 
        chat_id: str
    ) -> List[str]:
        """
        Get summaries from all closed (inactive) buckets for a conversation.
        
        Used for building context from previous conversation segments.
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            
        Returns:
            List of summary strings in chronological order (oldest first)
        """
        pass
    
    @abstractmethod
    async def update_bucket_summary(
        self, 
        bucket_id: str, 
        summary: str
    ) -> None:
        """
        Update the summary field of a bucket.
        
        Args:
            bucket_id: The bucket identifier
            summary: The summary text to store
        """
        pass
    
    @abstractmethod
    async def close_bucket(
        self, 
        bucket_id: str
    ) -> None:
        """
        Mark a bucket as inactive (closed).
        
        Args:
            bucket_id: The bucket identifier to close
        """
        pass
    
    # =========================================================================
    # ROLLOVER LOGIC - Default implementation, can be overridden
    # =========================================================================
    
    async def should_rollover(self, owner_id: str, chat_id: str) -> bool:
        """
        Check if the active bucket has reached rollover threshold.
        
        Checks both interaction count and token count (if configured).
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            
        Returns:
            True if rollover should occur
        """
        if not self.rollover_enabled:
            return False
        
        bucket = await self.get_active_bucket(owner_id, chat_id)
        if bucket is None:
            return False
        
        # Check interaction threshold
        message_count = len(bucket.messages)
        if message_count >= self.max_interactions_per_bucket * 2:  # *2 for pairs
            return True
        
        # Check token threshold if configured
        if self.max_tokens_per_bucket is not None:
            estimated_tokens = self._estimate_bucket_tokens(bucket)
            if estimated_tokens >= self.max_tokens_per_bucket:
                return True
        
        return False
    
    def _estimate_bucket_tokens(self, bucket: "MessageBucket") -> int:
        """
        Estimate token count for a bucket (rough heuristic).
        
        Uses ~4 characters per token as a rough estimate.
        Override for more accurate counting with actual tokenizers.
        
        Args:
            bucket: The MessageBucket to estimate
            
        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg.content) for msg in bucket.messages)
        return total_chars // 4  # Rough estimate: 4 chars per token
    
    async def rollover_history(
        self,
        owner_id: str,
        chat_id: str,
        summarize: bool = True,
    ) -> Optional[str]:
        """
        Perform rollover: close current bucket, summarize, create new bucket.
        
        Workflow:
        1. Get the active bucket
        2. If summarize=True, generate summary using the configured summarizer
        3. Save summary to the bucket
        4. Mark bucket as inactive
        5. Create new active bucket
        6. Optionally delete messages if preserve_closed_buckets=False
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            summarize: Whether to generate a summary (default True)
            
        Returns:
            The generated summary, or None if summarize=False
        """
        from .summarizers import resolve_summarizer
        
        # Get current active bucket
        bucket = await self.get_active_bucket(owner_id, chat_id)
        if bucket is None or not bucket.messages:
            # No bucket or empty bucket - just create a new one
            await self.create_bucket(owner_id, chat_id)
            return None
        
        summary = None
        
        # Generate summary if requested
        if summarize and self.summarizer:
            summary = await resolve_summarizer(self.summarizer, bucket.messages)
            await self.update_bucket_summary(bucket.bucket_id, summary)
        
        # Close the current bucket
        await self.close_bucket(bucket.bucket_id)
        
        # Create new active bucket
        new_position = (bucket.position or 0) + 1
        await self.create_bucket(owner_id, chat_id, position=new_position)
        
        return summary
    
    async def get_context_summary(self, owner_id: str, chat_id: str) -> Optional[str]:
        """
        Build a context summary from all closed buckets.
        
        If there are multiple closed buckets, aggregates their summaries
        into a cohesive context block.
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            
        Returns:
            Aggregated context summary, or None if no closed buckets
        """
        from .summarizers import summarize_summaries
        
        summaries = await self.get_bucket_summaries(owner_id, chat_id)
        
        if not summaries:
            return None
        
        if len(summaries) == 1:
            return summaries[0]
        
        # Multiple summaries - aggregate them
        if self.summarizer:
            return await summarize_summaries(summaries, self.summarizer)
        else:
            # Fallback: just concatenate
            return "\n\n".join(f"[Part {i+1}]: {s}" for i, s in enumerate(summaries))
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    async def add_message_from_provider(
        self, 
        message: Any, 
        owner_id: str, 
        chat_id: str,
        provider: Optional[str] = None
    ) -> None:
        """
        Add a message from any provider format (Gemini, OpenAI, Anthropic).
        
        Automatically parses provider-specific formats into our Message model.
        
        Args:
            message: Message in any format (dict, object, Message)
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            provider: Optional provider hint ('gemini', 'openai', 'anthropic')
        """
        if isinstance(message, Message):
            parsed = message
        elif isinstance(message, dict):
            parsed = Message.model_validate(message)
        elif hasattr(message, "model_dump") and callable(message.model_dump):
            parsed = Message.model_validate(message.model_dump())
        elif hasattr(message, "role") and hasattr(message, "content"):
            payload = {
                "role": getattr(message, "role"),
                "content": getattr(message, "content"),
                "timestamp": getattr(message, "timestamp", None),
                "tool_calls": getattr(message, "tool_calls", None),
                "thinking": getattr(message, "thinking", None),
                "thinking_tokens": getattr(message, "thinking_tokens", None),
                "tool_call_id": getattr(message, "tool_call_id", None),
            }
            parsed = Message.model_validate({k: v for k, v in payload.items() if v is not None})
        else:
            provider_hint = f" provider={provider!r}" if provider is not None else ""
            raise ValueError(
                "Unsupported message format for add_message_from_provider: "
                f"{type(message).__name__}.{provider_hint}"
            )

        await self.add_message(parsed, owner_id, chat_id)


