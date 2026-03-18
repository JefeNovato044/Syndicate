"""
MongoDB-based chat memory implementation.

Provides persistent storage for conversation history using MongoDB with
full support for bucket-based rollover and summarization.

Uses PyMongo's native async API (replacement for Motor).
"""

from typing import Optional, List, Dict, Any
import logging
from uuid import uuid4
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from .base import BaseChatMemory
from .summarizers import Summarizer
from ..communication_models import Message, MessageBucket


logger = logging.getLogger(__name__)


class MongoMemory(BaseChatMemory):
    """
    MongoDB implementation of BaseChatMemory.

    Stores conversation history in MongoDB using a single collection with
    embedded messages. Supports full summarization and bucket-based rollover.

    Features:
    - Persistent multi-tenant storage (owner_id, chat_id)
    - Bucket-based rollover with automatic summarization
    - Hierarchical context (active bucket + summaries of past buckets)
    - Efficient indexing for fast queries

    Collection Structure:
    ```
    {
        "_id": ObjectId,
        "bucket_id": str,
        "owner_id": str,
        "chat_id": str,
        "messages": [
            {"role": str, "content": str, "timestamp": datetime, ...}
        ],
        "summary": str | null,
        "is_active": bool,
        "position": int,
        "created_at": datetime,
        "closed_at": datetime | null,
        "estimated_tokens": int | null
    }
    ```
    """

    def __init__(
        self,
        database: AsyncDatabase,
        collection_name: str = "chat_buckets",
        rollover_enabled: bool = False,
        max_interactions_per_bucket: int = 10,
        max_tokens_per_bucket: Optional[int] = None,
        summarizer: Optional[Summarizer] = None,
        preserve_closed_buckets: bool = True,
    ):
        """
        Initialize MongoDB memory with database connection.

        Args:
            database: AsyncDatabase instance from PyMongo async client.
                Example: async_client["my_database"]
            collection_name: Name of the collection to store buckets.
                Defaults to "chat_buckets".
            rollover_enabled: Enable automatic rollover to new message buckets
                when thresholds are reached.
            max_interactions_per_bucket: Maximum number of interactions
                (user + ai message pairs) per bucket before rollover.
            max_tokens_per_bucket: Optional token-based threshold for rollover.
                If set, rollover occurs when estimated tokens exceed this limit.
            summarizer: Agent or Callable for summarizing buckets on rollover.
                Required if rollover_enabled=True.
            preserve_closed_buckets: If True, closed buckets are kept in storage.
                If False, only the summary is preserved and messages are deleted.
        """
        super().__init__(
            rollover_enabled=rollover_enabled,
            max_interactions_per_bucket=max_interactions_per_bucket,
            max_tokens_per_bucket=max_tokens_per_bucket,
            summarizer=summarizer,
            preserve_closed_buckets=preserve_closed_buckets,
            supports_summarization=True,
        )

        self._database = database
        self._collection_name = collection_name
        self._collection: AsyncCollection = database[collection_name]
        self._indexes_created = False

    async def _ensure_indexes(self) -> None:
        """Create indexes for efficient queries (idempotent)."""
        if self._indexes_created:
            return

        # Compound index for finding active bucket per conversation
        await self._collection.create_index(
            [("owner_id", ASCENDING), ("chat_id", ASCENDING), ("is_active", DESCENDING)],
            name="owner_chat_active_idx",
        )

        # Index for finding closed buckets (for summaries)
        await self._collection.create_index(
            [("owner_id", ASCENDING), ("chat_id", ASCENDING), ("position", ASCENDING)],
            name="owner_chat_position_idx",
        )

        # Enforce at most one active bucket per conversation
        await self._collection.create_index(
            [("owner_id", ASCENDING), ("chat_id", ASCENDING)],
            name="uq_one_active_bucket_per_chat",
            unique=True,
            partialFilterExpression={"is_active": True},
        )

        # Unique index on bucket_id
        await self._collection.create_index(
            [("bucket_id", ASCENDING)],
            name="bucket_id_idx",
            unique=True,
        )

        self._indexes_created = True

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
        and performs rollover with summarization if needed.

        Args:
            message: The Message object to store
            owner_id: User/owner identifier for multi-tenant support
            chat_id: Conversation/session identifier
            **kwargs: Implementation-specific options
        """
        await self._ensure_indexes()

        # Ensure bucket exists
        bucket = await self.get_active_bucket(owner_id, chat_id)
        if bucket is None:
            bucket = await self.create_bucket(owner_id, chat_id)

        # Check for rollover if enabled
        if self.rollover_enabled:
            if await self.should_rollover(owner_id, chat_id):
                logger.info(
                    "[MongoMemory] Triggering rollover for %s/%s - bucket full",
                    owner_id,
                    chat_id,
                )
                await self.rollover_history(owner_id, chat_id, summarize=True)
                # Get the newly created bucket
                bucket = await self.get_active_bucket(owner_id, chat_id)
                if bucket is not None:
                    logger.info(
                        "[MongoMemory] Rollover complete - new bucket created at position %s",
                        bucket.position,
                    )

        # Serialize message to dict for MongoDB
        message_dict = self._message_to_dict(message)

        # Add message to bucket in MongoDB
        await self._collection.update_one(
            {"bucket_id": bucket.bucket_id},
            {"$push": {"messages": message_dict}}
        )

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
        await self._ensure_indexes()

        messages: List[Message] = []

        # Include context summary if requested
        if include_context_summary:
            context = await self.get_context_summary(owner_id, chat_id)
            if context:
                # Prepend context as a system message
                messages.append(Message(
                    role="system",
                    content=f"Previous conversation context:\n{context}"
                ))

        # Get active bucket messages
        bucket = await self.get_active_bucket(owner_id, chat_id)
        if bucket is None:
            return messages

        bucket_messages = bucket.messages.copy()
        if limit is not None:
            bucket_messages = bucket_messages[-limit:]

        messages.extend(bucket_messages)
        return messages

    async def clear(self, owner_id: str, chat_id: str) -> None:
        """
        Clear all messages and buckets for a specific conversation.

        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
        """
        await self._ensure_indexes()

        await self._collection.delete_many({
            "owner_id": owner_id,
            "chat_id": chat_id
        })

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
        await self._ensure_indexes()

        doc = await self._collection.find_one(
            {
                "owner_id": owner_id,
                "chat_id": chat_id,
                "is_active": True,
            },
            sort=[("position", DESCENDING)],
        )

        if doc is None:
            return None

        return self._doc_to_bucket(doc)

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
        await self._ensure_indexes()

        # Determine position if not provided
        if position is None:
            # Get the highest position for this conversation
            last_bucket = await self._collection.find_one(
                {"owner_id": owner_id, "chat_id": chat_id},
                sort=[("position", DESCENDING)]
            )
            position = (last_bucket["position"] + 1) if last_bucket else 0

        # Close any existing active bucket
        await self._collection.update_many(
            {"owner_id": owner_id, "chat_id": chat_id, "is_active": True},
            {"$set": {"is_active": False, "closed_at": datetime.now(timezone.utc)}}
        )

        # Create new bucket
        bucket_id = str(uuid4())
        now = datetime.now(timezone.utc)

        bucket_doc = {
            "bucket_id": bucket_id,
            "owner_id": owner_id,
            "chat_id": chat_id,
            "messages": [],
            "summary": None,
            "is_active": True,
            "position": position,
            "created_at": now,
            "closed_at": None,
            "estimated_tokens": None,
        }

        try:
            await self._collection.insert_one(bucket_doc)
        except DuplicateKeyError:
            existing = await self._collection.find_one(
                {
                    "owner_id": owner_id,
                    "chat_id": chat_id,
                    "is_active": True,
                },
                sort=[("position", DESCENDING)],
            )
            if existing is not None:
                return self._doc_to_bucket(existing)
            raise

        return MessageBucket(
            bucket_id=bucket_id,
            owner_id=owner_id,
            chat_id=chat_id,
            messages=[],
            summary=None,
            is_active=True,
            position=position,
            created_at=now,
            closed_at=None,
            estimated_tokens=None,
        )

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
        await self._ensure_indexes()

        cursor = self._collection.find(
            {
                "owner_id": owner_id,
                "chat_id": chat_id,
                "is_active": False,
                "summary": {"$ne": None}
            },
            projection={"summary": 1, "position": 1},
            sort=[("position", ASCENDING)]
        )

        summaries = []
        async for doc in cursor:
            if doc.get("summary"):
                summaries.append(doc["summary"])

        return summaries

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
        await self._collection.update_one(
            {"bucket_id": bucket_id},
            {"$set": {"summary": summary}}
        )

    async def close_bucket(
        self,
        bucket_id: str
    ) -> None:
        """
        Mark a bucket as inactive (closed).

        Args:
            bucket_id: The bucket identifier to close
        """
        await self._collection.update_one(
            {"bucket_id": bucket_id},
            {"$set": {
                "is_active": False,
                "closed_at": datetime.now(timezone.utc)
            }}
        )

        # If not preserving closed buckets, clear messages but keep summary
        if not self.preserve_closed_buckets:
            await self._collection.update_one(
                {"bucket_id": bucket_id},
                {"$set": {"messages": []}}
            )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert a Message to a MongoDB-storable dict."""
        doc = {
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
        }

        # Optional fields
        if message.tool_calls is not None:
            doc["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in message.tool_calls
            ]
        if message.thinking is not None:
            doc["thinking"] = message.thinking
        if message.thinking_tokens is not None:
            doc["thinking_tokens"] = message.thinking_tokens
        if message.tool_call_id is not None:
            doc["tool_call_id"] = message.tool_call_id

        return doc

    def _dict_to_message(self, doc: Dict[str, Any]) -> Message:
        """Convert a MongoDB document to a Message."""
        from ..communication_models import ToolCall

        tool_calls = None
        if doc.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"]
                )
                for tc in doc["tool_calls"]
            ]

        return Message(
            role=doc["role"],
            content=doc["content"],
            timestamp=doc.get("timestamp", datetime.now(timezone.utc)),
            tool_calls=tool_calls,
            thinking=doc.get("thinking"),
            thinking_tokens=doc.get("thinking_tokens"),
            tool_call_id=doc.get("tool_call_id"),
        )

    def _doc_to_bucket(self, doc: Dict[str, Any]) -> MessageBucket:
        """Convert a MongoDB document to a MessageBucket."""
        messages = [self._dict_to_message(m) for m in doc.get("messages", [])]

        return MessageBucket(
            bucket_id=doc["bucket_id"],
            owner_id=doc["owner_id"],
            chat_id=doc["chat_id"],
            messages=messages,
            summary=doc.get("summary"),
            is_active=doc.get("is_active", True),
            position=doc.get("position", 0),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
            closed_at=doc.get("closed_at"),
            estimated_tokens=doc.get("estimated_tokens"),
        )

    async def get_all_buckets(
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
        await self._ensure_indexes()

        query: Dict[str, Any] = {}
        if owner_id is not None:
            query["owner_id"] = owner_id
        if chat_id is not None:
            query["chat_id"] = chat_id

        cursor = self._collection.find(query, sort=[("position", ASCENDING)])

        buckets = []
        async for doc in cursor:
            buckets.append(self._doc_to_bucket(doc))

        return buckets

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        await self._ensure_indexes()

        total_buckets = await self._collection.count_documents({})
        active_buckets = await self._collection.count_documents({"is_active": True})
        closed_buckets = await self._collection.count_documents({"is_active": False})

        # Count total messages via aggregation (pushed to MongoDB, not in-memory)
        pipeline = [
            {"$project": {"message_count": {"$size": "$messages"}}},
            {"$group": {"_id": None, "total": {"$sum": "$message_count"}}}
        ]
        total_messages = 0
        cursor = await self._collection.aggregate(pipeline)
        async for doc in cursor:
            total_messages = doc.get("total", 0)
            break

        return {
            "total_messages": total_messages,
            "active_buckets": active_buckets,
            "closed_buckets": closed_buckets,
            "total_buckets": total_buckets,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_indexes()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # No cleanup needed - client is managed externally
        pass


__all__ = ["MongoMemory"]