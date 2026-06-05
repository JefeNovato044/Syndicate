"""
SQLite/PostgreSQL-based chat memory implementation.

Provides persistent storage for conversation history using SQLAlchemy async
with support for both SQLite (development/local) and PostgreSQL (production).

Uses SQLAlchemy 2.0+ async API with aiosqlite and asyncpg drivers.
"""

import json
import logging
import asyncio
import base64
import binascii
from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    DateTime,
    Text,
    Index,
    select,
    update,
    delete,
    func,
    text,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.exc import IntegrityError

from .base import BaseChatMemory
from .summarizers import Summarizer
from ..communication_models import Message, MessageBucket, ToolCall


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


def _make_bucket_model(table_name: str = "chat_buckets"):
    """
    Create a ChatBucket ORM model bound to a specific table name.

    Returns a unique class each time so that multiple SqlitePostgresMemory
    instances with different table names never collide on the shared
    class-level __tablename__.
    """
    # Reuse the default model if the name hasn't changed (common path)
    if table_name == "chat_buckets":
        return ChatBucket

    # Dynamic model with full schema bound to custom table name.
    # Use explicit columns (instead of subclassing ChatBucket) so SQLAlchemy can
    # resolve index/constraint columns deterministically during class mapping.
    attrs = {
        "__tablename__": table_name,
        "id": Column(Integer, primary_key=True, autoincrement=True),
        "bucket_id": Column(String(36), unique=True, nullable=False, index=True),
        "owner_id": Column(String(255), nullable=False, index=True),
        "chat_id": Column(String(255), nullable=False, index=True),
        "messages": Column(Text, nullable=False, default="[]"),
        "summary": Column(Text, nullable=True),
        "is_active": Column(Boolean, nullable=False, default=True, index=True),
        "position": Column(Integer, nullable=False, default=0, index=True),
        "created_at": Column(DateTime, nullable=False, default=datetime.now(timezone.utc)),
        "closed_at": Column(DateTime, nullable=True),
        "estimated_tokens": Column(Integer, nullable=True),
        "__table_args__": (
            Index(f"idx_{table_name}_owner_chat_active", "owner_id", "chat_id", "is_active"),
            Index(f"idx_{table_name}_owner_chat_position", "owner_id", "chat_id", "position"),
            Index(
                f"uq_{table_name}_one_active_bucket_per_chat",
                "owner_id",
                "chat_id",
                unique=True,
                sqlite_where=text("is_active = 1"),
                postgresql_where=text("is_active = true"),
            ),
            {"extend_existing": True},
        ),
    }
    model = type(f"ChatBucket_{table_name}", (Base,), attrs)
    return model


class ChatBucket(Base):
    """
    SQLAlchemy model for chat buckets.
    
    Maps to the chat_buckets table with the following structure:
    - bucket_id: Unique identifier for the bucket
    - owner_id: User/owner identifier for multi-tenant support
    - chat_id: Conversation/session identifier
    - messages: JSON array of message objects
    - summary: Optional summary text for closed buckets
    - is_active: Boolean flag for active/closed status
    - position: Integer position in conversation sequence
    - created_at: Timestamp when bucket was created
    - closed_at: Timestamp when bucket was closed (nullable)
    - estimated_tokens: Optional token count estimate
    """
    __tablename__ = "chat_buckets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bucket_id = Column(String(36), unique=True, nullable=False, index=True)
    owner_id = Column(String(255), nullable=False, index=True)
    chat_id = Column(String(255), nullable=False, index=True)
    messages = Column(Text, nullable=False, default="[]")
    summary = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    position = Column(Integer, nullable=False, default=0, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    closed_at = Column(DateTime, nullable=True)
    estimated_tokens = Column(Integer, nullable=True)
    
    # Composite indexes for efficient queries
    __table_args__ = (
        Index("idx_owner_chat_active", "owner_id", "chat_id", "is_active"),
        Index("idx_owner_chat_position", "owner_id", "chat_id", "position"),
        Index(
            "uq_one_active_bucket_per_chat",
            "owner_id",
            "chat_id",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active = true"),
        ),
    )


class SqlitePostgresMemory(BaseChatMemory):
    """
    SQLite/PostgreSQL implementation of BaseChatMemory.
    
    Stores conversation history in a relational database using SQLAlchemy async.
    Supports both SQLite (for development/local use) and PostgreSQL (for production).
    
    Features:
    - Persistent multi-tenant storage (owner_id, chat_id)
    - Bucket-based rollover with automatic summarization
    - Hierarchical context (active bucket + summaries of past buckets)
    - Efficient indexing for fast queries
    - Works with SQLite (aiosqlite) or PostgreSQL (asyncpg)
    
    Database URL Examples:
    - SQLite file: "sqlite+aiosqlite:///./chat_history.db"
    - SQLite memory: "sqlite+aiosqlite:///:memory:"
    - PostgreSQL: "postgresql+asyncpg://user:pass@localhost:5432/syndicate"
    """

    def __init__(
        self,
        database_url: str,
        table_name: str = "chat_buckets",
        rollover_enabled: bool = False,
        max_interactions_per_bucket: int = 10,
        max_tokens_per_bucket: Optional[int] = None,
        summarizer: Optional[Summarizer] = None,
        preserve_closed_buckets: bool = True,
        soft_delete: bool = True,
    ):
        """
        Initialize SQLite/PostgreSQL memory with database connection.
        
        Args:
            database_url: Database connection URL
                - SQLite: "sqlite+aiosqlite:///./chat.db" or "sqlite+aiosqlite:///:memory:"
                - PostgreSQL: "postgresql+asyncpg://user:pass@host:port/dbname"
            table_name: Name of the table to store buckets.
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
            soft_delete: If True, deletion APIs mark messages as deleted and
                filter them from history. If False, messages are removed.
        """
        super().__init__(
            rollover_enabled=rollover_enabled,
            max_interactions_per_bucket=max_interactions_per_bucket,
            max_tokens_per_bucket=max_tokens_per_bucket,
            summarizer=summarizer,
            preserve_closed_buckets=preserve_closed_buckets,
            supports_summarization=True,
            soft_delete=soft_delete,
        )
        
        self._database_url = database_url
        self._table_name = table_name
        self._bucket_model = _make_bucket_model(table_name)
        self._engine: Optional[AsyncEngine] = None
        self._session_maker: Optional[async_sessionmaker[AsyncSession]] = None
        self._tables_created = False
        self._tables_lock = asyncio.Lock()
    
    def _get_engine(self) -> AsyncEngine:
        """Lazy initialization of async engine."""
        if self._engine is None:
            # Enable echo for debugging (optional, can be removed in production)
            self._engine = create_async_engine(
                self._database_url,
                echo=False,
                future=True,
            )
            self._session_maker = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._engine
    
    async def _ensure_tables(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        if self._tables_created:
            return

        async with self._tables_lock:
            if self._tables_created:
                return

            engine = self._get_engine()

            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._tables_created = True
    
    def _get_session(self) -> AsyncSession:
        """Get a database session."""
        if self._session_maker is None:
            self._get_engine()
        return self._session_maker()

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
        await self._ensure_tables()
        
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
                logger.info(
                    "[SqlitePostgresMemory] Triggering rollover for %s/%s - bucket full",
                    owner_id,
                    chat_id,
                )
                await self.rollover_history(owner_id, chat_id, summarize=True)
                # Get the newly created bucket
                bucket = await self.get_active_bucket(owner_id, chat_id)
                if bucket is not None:
                    logger.info(
                        "[SqlitePostgresMemory] Rollover complete - new bucket created at position %s",
                        bucket.position,
                    )
        
        # Serialize message to dict
        message_dict = self._message_to_dict(message)
        
        # Add message to bucket using a single transaction.
        # For PostgreSQL, apply row-level lock to avoid lost updates.
        # For SQLite, FOR UPDATE is not supported and writes are serialized.
        async with self._get_session() as session:
            async with session.begin():
                M = self._bucket_model
                stmt = select(M).where(M.bucket_id == bucket.bucket_id)
                if not self._database_url.startswith("sqlite"):
                    stmt = stmt.with_for_update()

                result = await session.execute(stmt)
                db_bucket = result.scalar_one_or_none()

                if db_bucket:
                    messages = json.loads(db_bucket.messages)
                    messages.append(message_dict)
                    db_bucket.messages = json.dumps(messages)

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
        await self._ensure_tables()
        
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
        
        async with self._get_session() as session:
            M = self._bucket_model
            result = await session.execute(
                select(M.messages).where(
                    M.owner_id == owner_id,
                    M.chat_id == chat_id,
                    M.is_active.is_(True)
                ).order_by(M.position.desc()).limit(1)
            )
            row = result.first()

        if row is None:
            return messages

        raw_messages = json.loads(row.messages or "[]")
        visible_messages = [
            self._dict_to_message(raw)
            for raw in raw_messages
            if not raw.get("$deleted", False)
        ]
        if limit is not None:
            visible_messages = visible_messages[-limit:]

        messages.extend(visible_messages)
        return messages

    async def get_full_history(
        self,
        owner_id: str,
        chat_id: str,
        limit: Optional[int] = None,
        include_closed_buckets: bool = True,
        include_deleted: bool = False,
        include_context_summary: bool = False,
    ) -> List[Message]:
        """Get flattened history for a conversation across bucket boundaries."""
        await self._ensure_tables()

        context_message: Optional[Message] = None
        if include_context_summary and not include_closed_buckets:
            context = await self.get_context_summary(owner_id, chat_id)
            if context:
                context_message = Message(
                    role="system",
                    content=f"Previous conversation context:\n{context}",
                )

        messages: List[Message] = []
        async with self._get_session() as session:
            M = self._bucket_model
            if include_closed_buckets:
                result = await session.execute(
                    select(M.messages).where(
                        M.owner_id == owner_id,
                        M.chat_id == chat_id,
                    ).order_by(M.position.asc())
                )
                raw_rows = result.all()
            else:
                result = await session.execute(
                    select(M.messages).where(
                        M.owner_id == owner_id,
                        M.chat_id == chat_id,
                        M.is_active.is_(True),
                    ).order_by(M.position.desc()).limit(1)
                )
                raw_rows = result.all()

        for row in raw_rows:
            raw_messages = json.loads(row.messages or "[]")
            for raw in raw_messages:
                if not include_deleted and raw.get("$deleted", False):
                    continue
                messages.append(self._dict_to_message(raw))

        if limit is not None:
            messages = messages[-limit:]

        if context_message is not None:
            return [context_message] + messages
        return messages

    async def clear(self, owner_id: str, chat_id: str) -> None:
        """
        Clear all messages and buckets for a specific conversation.
        
        Args:
            owner_id: User/owner identifier
            chat_id: Conversation/session identifier
        """
        await self._ensure_tables()
        
        async with self._get_session() as session:
            M = self._bucket_model
            await session.execute(
                delete(M).where(
                    M.owner_id == owner_id,
                    M.chat_id == chat_id
                )
            )
            await session.commit()

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

    async def delete_message(
        self,
        owner_id: str,
        chat_id: str,
        index: int = -1,
        hard_delete: Optional[bool] = None,
    ) -> bool:
        """Delete one visible message from the active bucket by index."""
        await self._ensure_tables()

        async with self._get_session() as session:
            async with session.begin():
                M = self._bucket_model
                stmt = select(M).where(
                    M.owner_id == owner_id,
                    M.chat_id == chat_id,
                    M.is_active.is_(True)
                ).order_by(M.position.desc()).limit(1)
                if not self._database_url.startswith("sqlite"):
                    stmt = stmt.with_for_update()

                result = await session.execute(stmt)
                db_bucket = result.scalar_one_or_none()
                if db_bucket is None:
                    return False

                raw_messages = json.loads(db_bucket.messages or "[]")
                visible_positions = [
                    i for i, msg in enumerate(raw_messages)
                    if not msg.get("$deleted", False)
                ]
                if not visible_positions:
                    return False

                visible_index = self._normalize_index(index, len(visible_positions))
                if visible_index is None:
                    return False

                target_index = visible_positions[visible_index]
                if self._should_hard_delete(hard_delete):
                    raw_messages.pop(target_index)
                else:
                    raw_messages[target_index]["$deleted"] = True
                    raw_messages[target_index]["$deleted_at"] = datetime.now(timezone.utc).isoformat()

                db_bucket.messages = json.dumps(raw_messages)

        return True

    async def delete_last_message(
        self,
        owner_id: str,
        chat_id: str,
        role: Optional[str] = None,
        hard_delete: Optional[bool] = None,
    ) -> bool:
        """Delete the last visible message, optionally filtered by role."""
        await self._ensure_tables()
        role_filter = self._normalize_role_filter(role)

        async with self._get_session() as session:
            async with session.begin():
                M = self._bucket_model
                stmt = select(M).where(
                    M.owner_id == owner_id,
                    M.chat_id == chat_id,
                    M.is_active.is_(True)
                ).order_by(M.position.desc()).limit(1)
                if not self._database_url.startswith("sqlite"):
                    stmt = stmt.with_for_update()

                result = await session.execute(stmt)
                db_bucket = result.scalar_one_or_none()
                if db_bucket is None:
                    return False

                raw_messages = json.loads(db_bucket.messages or "[]")
                target_index: Optional[int] = None

                for idx in range(len(raw_messages) - 1, -1, -1):
                    message = raw_messages[idx]
                    if message.get("$deleted", False):
                        continue
                    if role_filter is not None and message.get("role") != role_filter:
                        continue
                    target_index = idx
                    break

                if target_index is None:
                    return False

                if self._should_hard_delete(hard_delete):
                    raw_messages.pop(target_index)
                else:
                    raw_messages[target_index]["$deleted"] = True
                    raw_messages[target_index]["$deleted_at"] = datetime.now(timezone.utc).isoformat()

                db_bucket.messages = json.dumps(raw_messages)

        return True

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
        await self._ensure_tables()
        
        async with self._get_session() as session:
            M = self._bucket_model
            result = await session.execute(
                select(M).where(
                    M.owner_id == owner_id,
                    M.chat_id == chat_id,
                    M.is_active.is_(True)
                ).order_by(M.position.desc()).limit(1)
            )
            db_bucket = result.scalar_one_or_none()
            
            if db_bucket is None:
                return None
            
            return self._db_to_bucket(db_bucket)

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
        await self._ensure_tables()
        
        async with self._get_session() as session:
            M = self._bucket_model
            # Determine position if not provided
            if position is None:
                result = await session.execute(
                    select(func.max(M.position)).where(
                        M.owner_id == owner_id,
                        M.chat_id == chat_id
                    )
                )
                max_position = result.scalar()
                position = (max_position or -1) + 1
            
            # Close any existing active bucket
            await session.execute(
                update(M).where(
                    M.owner_id == owner_id,
                    M.chat_id == chat_id,
                    M.is_active.is_(True)
                ).values(
                    is_active=False,
                    closed_at=datetime.now(timezone.utc)
                )
            )
            
            # Create new bucket
            bucket_id = str(uuid4())
            now = datetime.now(timezone.utc)
            
            db_bucket = M(
                bucket_id=bucket_id,
                owner_id=owner_id,
                chat_id=chat_id,
                messages="[]",
                summary=None,
                is_active=True,
                position=position,
                created_at=now,
                closed_at=None,
                estimated_tokens=None,
            )
            
            session.add(db_bucket)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing_result = await session.execute(
                    select(M).where(
                        M.owner_id == owner_id,
                        M.chat_id == chat_id,
                        M.is_active.is_(True),
                    ).order_by(M.position.desc()).limit(1)
                )
                existing_bucket = existing_result.scalar_one_or_none()
                if existing_bucket is not None:
                    return self._db_to_bucket(existing_bucket)
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
        await self._ensure_tables()
        
        async with self._get_session() as session:
            M = self._bucket_model
            result = await session.execute(
                select(M.summary, M.position).where(
                    M.owner_id == owner_id,
                    M.chat_id == chat_id,
                    M.is_active.is_(False),
                    M.summary.is_not(None)
                ).order_by(M.position.asc())
            )
            
            summaries = []
            for row in result:
                if row.summary:
                    summaries.append(row.summary)
            
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
        await self._ensure_tables()
        
        async with self._get_session() as session:
            M = self._bucket_model
            await session.execute(
                update(M).where(
                    M.bucket_id == bucket_id
                ).values(
                    summary=summary
                )
            )
            await session.commit()

    async def close_bucket(
        self,
        bucket_id: str
    ) -> None:
        """
        Mark a bucket as inactive (closed).
        
        Args:
            bucket_id: The bucket identifier to close
        """
        await self._ensure_tables()
        
        async with self._get_session() as session:
            M = self._bucket_model
            # Mark as closed
            await session.execute(
                update(M).where(
                    M.bucket_id == bucket_id
                ).values(
                    is_active=False,
                    closed_at=datetime.now(timezone.utc)
                )
            )
            
            # If not preserving closed buckets, clear messages but keep summary
            if not self.preserve_closed_buckets:
                await session.execute(
                    update(M).where(
                        M.bucket_id == bucket_id
                    ).values(
                        messages="[]"
                    )
                )
            
            await session.commit()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _encode_thought_signature(
        thought_signature: Optional[bytes | str],
    ) -> Optional[bytes | str | Dict[str, str]]:
        """Serialize thought signatures for JSON-safe persistence."""
        if thought_signature is None or isinstance(thought_signature, str):
            return thought_signature

        if isinstance(thought_signature, bytes):
            return {
                "encoding": "base64",
                "value": base64.b64encode(thought_signature).decode("ascii"),
            }

        return str(thought_signature)

    @staticmethod
    def _decode_thought_signature(
        thought_signature: Any,
    ) -> Optional[bytes | str]:
        """Deserialize persisted thought signatures into ToolCall-compatible types."""
        if thought_signature is None:
            return None

        if isinstance(thought_signature, dict) and thought_signature.get("encoding") == "base64":
            encoded_value = thought_signature.get("value")
            if not isinstance(encoded_value, str):
                return None
            try:
                return base64.b64decode(encoded_value, validate=True)
            except (ValueError, TypeError, binascii.Error):
                logger.warning("Invalid base64 thought_signature payload found in memory")
                return None

        return thought_signature

    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert a Message to a dict for JSON storage."""
        doc = {
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat() if message.timestamp else datetime.now(timezone.utc).isoformat(),
        }
        
        # Optional fields
        if message.tool_calls is not None:
            doc["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "thought_signature": self._encode_thought_signature(tc.thought_signature),
                }
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
        """Convert a dict to a Message."""
        tool_calls = None
        if doc.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                    thought_signature=self._decode_thought_signature(tc.get("thought_signature")),
                )
                for tc in doc["tool_calls"]
            ]
        
        # Parse timestamp
        timestamp = doc.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                timestamp = datetime.now(timezone.utc)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        return Message(
            role=doc["role"],
            content=doc["content"],
            timestamp=timestamp,
            tool_calls=tool_calls,
            thinking=doc.get("thinking"),
            thinking_tokens=doc.get("thinking_tokens"),
            tool_call_id=doc.get("tool_call_id"),
        )

    def _db_to_bucket(self, db_bucket: ChatBucket) -> MessageBucket:
        """Convert a database row to a MessageBucket."""
        messages = [self._dict_to_message(m) for m in json.loads(db_bucket.messages or "[]")]
        
        return MessageBucket(
            bucket_id=db_bucket.bucket_id,
            owner_id=db_bucket.owner_id,
            chat_id=db_bucket.chat_id,
            messages=messages,
            summary=db_bucket.summary,
            is_active=db_bucket.is_active,
            position=db_bucket.position,
            created_at=db_bucket.created_at,
            closed_at=db_bucket.closed_at,
            estimated_tokens=db_bucket.estimated_tokens,
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
        await self._ensure_tables()
        
        async with self._get_session() as session:
            M = self._bucket_model
            query = select(M).order_by(M.position.asc())
            
            if owner_id is not None:
                query = query.where(M.owner_id == owner_id)
            if chat_id is not None:
                query = query.where(M.chat_id == chat_id)
            
            result = await session.execute(query)
            buckets = []
            for db_bucket in result.scalars().all():
                buckets.append(self._db_to_bucket(db_bucket))
            
            return buckets

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        await self._ensure_tables()
        
        async with self._get_session() as session:
            M = self._bucket_model
            # Total buckets
            result = await session.execute(select(func.count(M.id)))
            total_buckets = result.scalar() or 0
            
            # Active buckets
            result = await session.execute(
                select(func.count(M.id)).where(M.is_active.is_(True))
            )
            active_buckets = result.scalar() or 0
            
            # Closed buckets
            closed_buckets = total_buckets - active_buckets
            
            # Total messages (by parsing JSON)
            result = await session.execute(select(M.messages))
            total_messages = 0
            for row in result:
                messages = json.loads(row.messages or "[]")
                total_messages += len(messages)
            
            return {
                "total_messages": total_messages,
                "active_buckets": active_buckets,
                "closed_buckets": closed_buckets,
                "total_buckets": total_buckets,
            }

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_tables()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._engine:
            await self._engine.dispose()

    @staticmethod
    def _normalize_index(index: int, length: int) -> Optional[int]:
        """Convert Python-style negative index to non-negative index."""
        if length <= 0:
            return None
        if index < 0:
            index = length + index
        if 0 <= index < length:
            return index
        return None

    @staticmethod
    def _normalize_role_filter(role: Optional[str]) -> Optional[str]:
        """Normalize optional role filter to core role names."""
        if role is None:
            return None
        normalized = role.lower().strip()
        if normalized in ("user", "person", "human"):
            return "human"
        if normalized in ("model", "bot", "agent", "assistant", "ai"):
            return "ai"
        if normalized in ("system", "sys", "instruction", "developer"):
            return "system"
        if normalized in ("tool", "function", "observation"):
            return "tool"
        return normalized
