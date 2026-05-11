# Chat Memory Rollback and Regeneration

This guide explains how to use the Rollback and Regeneration APIs in Syndicate to manage chat history and correct agent mistakes.

---

## Overview

Syndicate provides built-in mechanisms for:
1. **Rollback**: Deleting specific messages from chat history (Soft or Hard delete).
2. **Regeneration**: Removing the last interaction and re-invoking the agent with the same user prompt.

These features are handled by the `BaseChatMemory` contract and integrated into `BaseAgent`.

---

## Rollback API

You can remove messages from the active bucket of the memory backend.

### Soft Delete (Default)
By default, deleting a message marks it as deleted in the database but keeps the record. Deleted messages are automatically filtered out when calling `get_history()`.

```python
# Delete a specific message by index
await agent.memory.delete_message(
    owner_id="user123",
    chat_id="chat456",
    index=5  # 0-based index in the history
)

# Delete the last AI message
await agent.memory.delete_last_message(
    owner_id="user123",
    chat_id="chat456",
    role="assistant"
)
```

### Hard Delete
To permanently remove a message from storage, set `hard_delete=True`.

```python
await agent.memory.delete_message(
    owner_id="user123",
    chat_id="chat456",
    index=5,
    hard_delete=True
)
```

### Configuration
You can control the default soft-delete behavior via `memory_config` in your agent:

```python
from syndicate.memory import LocalMemory

memory = LocalMemory(
    soft_delete=False  # All deletions will be hard deletes
)
```

---

## Regeneration API

The `regenerate_response` method on `BaseAgent` allows you to "undo" the lastTurn and try again. 

### How it works
1. It identifies the last `user` message in the active history.
2. It **permanently deletes** (hard delete) everything *after* that user message.
3. It re-invokes the agent using that same user prompt.

```python
# Invoke once
response = await agent.invoke("Explain quantum physics", session_id="session_1")

# If you don't like the answer, regenerate!
new_response = await agent.regenerate_response(session_id="session_1")
```

### Important Notes
- **Scope**: Both Rollback and Regeneration currently only operate on the **Active Bucket** of the chat history.
- **Permanent Change**: `regenerate_response` uses **Hard Delete** for truncation. Once triggered, the previous intermediate turns or the failed assistant response are gone.

---

## Full History API

For UI timelines, exports, or audits, use `get_full_history()`.

```python
# Full flattened history: closed buckets + active bucket
messages = await agent.get_full_history(
    owner_id="user123",
    chat_id="chat456",
)

# Active bucket only
active_messages = await agent.get_full_history(
    owner_id="user123",
    chat_id="chat456",
    include_closed_buckets=False,
)

# Include soft-deleted messages for admin/audit views
all_messages = await agent.get_full_history(
    owner_id="user123",
    chat_id="chat456",
    include_deleted=True,
)
```

### Parameters
- `include_closed_buckets` (default `True`): include historical closed buckets.
- `include_deleted` (default `False`): include messages marked with `$deleted`.
- `limit` (optional): return only the most recent N messages.
- `include_context_summary` (default `False`): prepend summary context when reading active-only history.

### Caveat
- If `preserve_closed_buckets=False`, closed bucket messages are physically cleared on close. In this mode, `get_full_history()` cannot reconstruct those old raw messages and only active-bucket messages remain available.

---

## Backend Support

The following memory backends fully support Rollback and Regeneration:
- `LocalMemory` (RAM)
- `MongoMemory` (MongoDB Atlas)
- `SqlitePostgresMemory` (SQLite & PostgreSQL)

Each backend implements the `$deleted` marker for soft-deletes to ensure consistency across distributed environments.
