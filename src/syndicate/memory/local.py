"""
Local in-memory memory implementation.

Provides simple RAM-based storage for conversation history.
No summarization support - messages are stored as-is.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from .base import BaseChatMemory
from ..communication_models import Message, MessageBucket
from .summarizers import Summarizer


class LocalMemory(BaseChatMemory):
    """
    In-memory implementation of BaseChatMemory.

    Stores conversation history in RAM using dictionaries keyed by
    (owner_id, chat_id). No summarization support - messages are
    stored as-is in active buckets.

    Features:
    - Multi-tenant support (owner_id, chat_id)
    - Simple bucket-based storage
    - No summarization (as requested)
    - All messages preserved in active buckets

    Note: Messages are lost when the process terminates.
    """

    def __init__(
        self,
        rollover_enabled: bool = False,
        max_interactions_per_bucket: int = 10,
        max_tokens_per_bucket: Optional[int] = None,
        summarizer: Optional[Summarizer] = None,
        preserve_closed_buckets: bool = True,
    ):
        """
        Initialize local memory with optional rollover configuration.

        Args:
            rollover_enabled: Enable automatic rollover to new message buckets
                when thresholds are reached.
            max_interactions_per_bucket: Maximum number of interactions
                (user + ai message pairs) per bucket before rollover.
            max_tokens_per_bucket: Optional token-based threshold for rollover.
                If set, rollover occurs when estimated tokens exceed this limit.
            summarizer: Not used in LocalMemory (no summarization support).
            preserve_closed_buckets: If True, closed buckets are kept in storage.
                If False, only the summary is preserved and messages are deleted.
                Note: In LocalMemory, this only affects whether inactive buckets
                are kept in memory.
        """
        # Note: summarizer is accepted for API compatibility but not used
        super().__init__(
            rollover_enabled=rollover_enabled,
            max_interactions_per_bucket=max_interactions_per_bucket,
            max_tokens_per_bucket=max_tokens_per_bucket,
            summarizer=None,  # No summarization support
            preserve_closed_buckets=preserve_closed_buckets,
            supports_summarization=False,  # LocalMemory doesn't support summarization
        )
        
        # Active bucket storage: { (owner_id, chat_id): MessageBucket }
        self._buckets: Dict[tuple, MessageBucket] = {}
        # Closed bucket storage: { (owner_id, chat_id): [MessageBucket, ...] }
        self._closed_buckets: Dict[tuple, List[MessageBucket]] = {}

    # =========================================================================
    # ABSTRACT METHODS - Core Chat Memory Operations
    # =========================================================================

    async def add_message(
        self,
        message: Message,
        owner_id: str,
        chat_id: str,
        **kwargs
    ) -> None:
        """
        Add a message to the active bucket of a conversation.

        If rollover_enabled is True, checks if rollover threshold is reached
        and creates a new bucket if needed.

        Args:
            message: The Message object to store
            owner_id: User/owner identifier for multi-tenant support
            chat_id: Conversation/session identifier
            **kwargs: Implementation-specific options
        """
        # Ensure bucket exists
        bucket = await self.get_active_bucket(owner_id, chat_id)
        if bucket is None:
            bucket = await self.create_bucket(owner_id, chat_id)
        
        # Check for rollover if enabled.
        # defer_rollover=True means the caller (e.g. _store_interaction) already
        # handled the rollover check once for the whole batch; skip it here to
        # prevent a bucket split mid-interaction.
        if self.rollover_enabled and not kwargs.get("defer_rollover", False):
            if await self.should_rollover(owner_id, chat_id):
                await self.rollover_history(owner_id, chat_id, summarize=False)
                # Get the newly created bucket
                bucket = await self.get_active_bucket(owner_id, chat_id)
        
        # Add message to bucket
        bucket.add_message(message)

    async def get_history(
        self,
        owner_id: str,
        chat_id: str,
        limit: Optional[int] = None,
        include_context_summary: bool = True
    ) -> List[Message]:
        """
        Get conversation history from the active bucket.

        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            limit: Max messages to return from active bucket (None = all)
            include_context_summary: Ignored in LocalMemory (no summaries)
            
        Returns:
            List of Messages in chronological order
        """
        bucket = await self.get_active_bucket(owner_id, chat_id)
        if bucket is None:
            return []
        
        messages = bucket.messages.copy()
        
        if limit is not None:
            messages = messages[-limit:]  # Get most recent messages
        
        return messages

    async def clear(self, owner_id: str, chat_id: str) -> None:
        """
        Clear all messages and buckets for a specific conversation.

        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
        """
        key = (owner_id, chat_id)
        if key in self._buckets:
            del self._buckets[key]
        if key in self._closed_buckets:
            del self._closed_buckets[key]

    async def get_message_count(self, owner_id: str, chat_id: str) -> int:
        """
        Get total number of messages in the active bucket.

        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            
        Returns:
            Number of messages in active bucket
        """
        bucket = await self.get_active_bucket(owner_id, chat_id)
        if bucket is None:
            return 0
        return bucket.message_count()

    # =========================================================================
    # BUCKET MANAGEMENT METHODS
    # =========================================================================

    async def get_active_bucket(
        self,
        owner_id: str,
        chat_id: str
    ) -> Optional[MessageBucket]:
        """
        Get the currently active bucket for a conversation.

        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            
        Returns:
            The active MessageBucket, or None if no bucket exists
        """
        key = (owner_id, chat_id)
        bucket = self._buckets.get(key)
        if bucket and bucket.is_active:
            return bucket
        return None

    async def create_bucket(
        self,
        owner_id: str,
        chat_id: str,
        position: Optional[int] = None
    ) -> MessageBucket:
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
        key = (owner_id, chat_id)
        
        # If there's an existing active bucket, close it
        existing_bucket = self._buckets.get(key)
        if existing_bucket and existing_bucket.is_active:
            existing_bucket.is_active = False
            existing_bucket.closed_at = datetime.now(timezone.utc)
            if self.preserve_closed_buckets:
                self._closed_buckets.setdefault(key, []).append(existing_bucket)
        
        if position is None:
            if existing_bucket is not None:
                position = existing_bucket.position + 1
            else:
                position = len(self._closed_buckets.get(key, []))

        # Create new bucket
        bucket_id = f"{owner_id}_{chat_id}_{position}"
        new_bucket = MessageBucket(
            bucket_id=bucket_id,
            owner_id=owner_id,
            chat_id=chat_id,
            position=position,
        )
        
        self._buckets[key] = new_bucket
        return new_bucket

    async def get_bucket_summaries(
        self,
        owner_id: str,
        chat_id: str
    ) -> List[str]:
        """
        Get summaries from all closed (inactive) buckets for a conversation.

        In LocalMemory, this returns an empty list since no summarization
        is supported.

        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
            
        Returns:
            Empty list (no summaries in LocalMemory)
        """
        return []

    async def update_bucket_summary(
        self,
        bucket_id: str,
        summary: str
    ) -> None:
        """
        Update the summary field of a bucket.

        In LocalMemory, this is a no-op since no summarization is supported.

        Args:
            bucket_id: The bucket identifier
            summary: The summary text to store (ignored)
        """
        # No-op - no summarization support
        pass

    async def close_bucket(
        self,
        bucket_id: str
    ) -> None:
        """
        Mark a bucket as inactive (closed).

        Args:
            bucket_id: The bucket identifier to close
        """
        # Find and close the bucket
        for key, bucket in list(self._buckets.items()):
            if bucket.bucket_id == bucket_id:
                bucket.is_active = False
                bucket.closed_at = datetime.now(timezone.utc)
                if self.preserve_closed_buckets:
                    self._closed_buckets.setdefault(key, []).append(bucket)
                del self._buckets[key]
                break

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_all_buckets(
        self,
        owner_id: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> List[MessageBucket]:
        """
        Get all buckets for a conversation or all conversations.

        Args:
            owner_id: Optional user/owner identifier filter
            chat_id: Optional conversation/session identifier filter
            
        Returns:
            List of MessageBucket objects
        """
        if owner_id is None and chat_id is None:
            all_closed = [b for buckets in self._closed_buckets.values() for b in buckets]
            return list(self._buckets.values()) + all_closed
        
        key = (owner_id, chat_id)
        if owner_id is not None and chat_id is not None:
            result = []
            active = self._buckets.get(key)
            if active is not None:
                result.append(active)
            result.extend(self._closed_buckets.get(key, []))
            return result
        
        # Filter by owner_id or chat_id
        results = []
        for k, bucket in self._buckets.items():
            if owner_id is not None and k[0] == owner_id:
                results.append(bucket)
            elif chat_id is not None and k[1] == chat_id:
                results.append(bucket)
        for k, buckets in self._closed_buckets.items():
            if owner_id is not None and k[0] == owner_id:
                results.extend(buckets)
            elif chat_id is not None and k[1] == chat_id:
                results.extend(buckets)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        total_messages = sum(bucket.message_count() for bucket in self._buckets.values()) + sum(
            bucket.message_count() for buckets in self._closed_buckets.values() for bucket in buckets
        )
        active_buckets = sum(
            1 for bucket in self._buckets.values() 
            if bucket.is_active
        )
        closed_buckets = sum(len(buckets) for buckets in self._closed_buckets.values())
        
        return {
            "total_messages": total_messages,
            "active_buckets": active_buckets,
            "closed_buckets": closed_buckets,
            "total_buckets": active_buckets + closed_buckets,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


__all__ = ['LocalMemory']
