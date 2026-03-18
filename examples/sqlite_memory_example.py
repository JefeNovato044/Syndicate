"""
SQLite/PostgreSQL Memory Example

Demonstrates how to use SqlitePostgresMemory for persistent conversation storage
with both SQLite (development) and PostgreSQL (production).
"""

import asyncio
import os
from dotenv import load_dotenv

from syndicate.memory import SqlitePostgresMemory
from syndicate.communication_models import Message
from syndicate.clients.gemini import GeminiClient
from syndicate.memory.summarizers import create_default_summarizer


# Load environment variables
load_dotenv()


async def sqlite_example():
    """Example using SQLite for local/development use."""
    print("=" * 60)
    print("SQLite Memory Example")
    print("=" * 60)
    
    # Create memory instance with SQLite
    # Using a file-based database for persistence
    memory = SqlitePostgresMemory(
        database_url="sqlite+aiosqlite:///./examples_chat.db",
        rollover_enabled=False,  # Disable rollover for simple example
    )
    
    try:
        # Add some messages
        await memory.add_message(
            Message(role="user", content="Hello, how are you?"),
            owner_id="user1",
            chat_id="session1"
        )
        
        await memory.add_message(
            Message(role="assistant", content="I'm doing well, thank you! How can I help you today?"),
            owner_id="user1",
            chat_id="session1"
        )
        
        await memory.add_message(
            Message(role="user", content="Can you explain what SQLAlchemy is?"),
            owner_id="user1",
            chat_id="session1"
        )
        
        # Get conversation history
        history = await memory.get_history(
            owner_id="user1",
            chat_id="session1"
        )
        
        print("\nConversation History:")
        print("-" * 40)
        for msg in history:
            print(f"{msg.role}: {msg.content}")
        
        # Get message count
        count = await memory.get_message_count("user1", "session1")
        print(f"\nTotal messages: {count}")
        
        # Get stats
        stats = await memory.get_stats()
        print(f"\nMemory Stats: {stats}")
        
    finally:
        # Cleanup: clear the conversation
        await memory.clear("user1", "session1")
        print("\nConversation cleared.")


async def sqlite_with_rollover_example():
    """Example using SQLite with automatic rollover and summarization."""
    print("\n" + "=" * 60)
    print("SQLite Memory with Rollover Example")
    print("=" * 60)
    
    # Initialize LLM client for summarization
    gemini_client = GeminiClient(
        model_name="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Create memory with rollover enabled
    memory = SqlitePostgresMemory(
        database_url="sqlite+aiosqlite:///./examples_chat_rollover.db",
        rollover_enabled=True,
        max_interactions_per_bucket=3,  # Rollover after 3 interactions (6 messages)
        summarizer=create_default_summarizer(gemini_client),
        preserve_closed_buckets=True,
    )
    
    try:
        # Add multiple messages to trigger rollover
        messages = [
            ("user", "What is Python?"),
            ("assistant", "Python is a high-level programming language..."),
            ("user", "What are decorators in Python?"),
            ("assistant", "Decorators are functions that modify other functions..."),
            ("user", "Can you show me an example?"),
            ("assistant", "Sure! Here's a simple decorator example..."),
            ("user", "What about async/await?"),
            ("assistant", "Async/await enables asynchronous programming..."),
        ]
        
        for role, content in messages:
            await memory.add_message(
                Message(role=role, content=content),
                owner_id="user1",
                chat_id="session_rollover"
            )
            print(f"Added {role} message: {content[:50]}...")
        
        # Get full history with context summary
        history = await memory.get_history(
            owner_id="user1",
            chat_id="session_rollover",
            include_context_summary=True
        )
        
        print("\n" + "-" * 40)
        print("Full History (with context summary):")
        print("-" * 40)
        for msg in history:
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"{msg.role}: {preview}")
        
        # Get bucket summaries
        summaries = await memory.get_bucket_summaries("user1", "session_rollover")
        print(f"\nBucket Summaries ({len(summaries)} closed buckets):")
        for i, summary in enumerate(summaries, 1):
            print(f"  Bucket {i}: {summary[:80]}...")
        
        # Get all buckets
        all_buckets = await memory.get_all_buckets("user1", "session_rollover")
        print(f"\nTotal buckets: {len(all_buckets)}")
        for bucket in all_buckets:
            status = "active" if bucket.is_active else "closed"
            print(f"  Bucket {bucket.position}: {status}, {bucket.message_count()} messages")
        
    finally:
        # Cleanup
        await memory.clear("user1", "session_rollover")
        print("\nConversation cleared.")


async def postgres_example():
    """Example using PostgreSQL for production use."""
    print("\n" + "=" * 60)
    print("PostgreSQL Memory Example")
    print("=" * 60)
    
    # Get PostgreSQL connection URL from environment
    postgres_url = os.getenv(
        "POSTGRES_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/syndicate"
    )
    
    try:
        # Create memory instance with PostgreSQL
        memory = SqlitePostgresMemory(
            database_url=postgres_url,
            table_name="chat_buckets",
            rollover_enabled=False,
        )
        
        # Test connection by adding a message
        await memory.add_message(
            Message(role="user", content="Testing PostgreSQL connection"),
            owner_id="user1",
            chat_id="postgres_test"
        )
        
        history = await memory.get_history("user1", "postgres_test")
        print(f"PostgreSQL connection successful! Retrieved {len(history)} message(s).")
        
        # Cleanup
        await memory.clear("user1", "postgres_test")
        
    except Exception as e:
        print(f"PostgreSQL example skipped: {e}")
        print("Set POSTGRES_URL environment variable to use PostgreSQL.")


async def multi_tenant_example():
    """Example demonstrating multi-tenant support."""
    print("\n" + "=" * 60)
    print("Multi-Tenant Memory Example")
    print("=" * 60)
    
    memory = SqlitePostgresMemory(
        database_url="sqlite+aiosqlite:///./examples_multi_tenant.db",
    )
    
    try:
        # User 1 conversation
        await memory.add_message(
            Message(role="user", content="Hi, I'm Alice!"),
            owner_id="alice",
            chat_id="general"
        )
        
        # User 2 conversation (same chat_id, different owner)
        await memory.add_message(
            Message(role="user", content="Hello, I'm Bob!"),
            owner_id="bob",
            chat_id="general"
        )
        
        # Get Alice's history
        alice_history = await memory.get_history("alice", "general")
        print(f"\nAlice's messages: {len(alice_history)}")
        for msg in alice_history:
            print(f"  {msg.content}")
        
        # Get Bob's history
        bob_history = await memory.get_history("bob", "general")
        print(f"\nBob's messages: {len(bob_history)}")
        for msg in bob_history:
            print(f"  {msg.content}")
        
        # Get stats
        stats = await memory.get_stats()
        print(f"\nTotal stats: {stats}")
        
    finally:
        await memory.clear("alice", "general")
        await memory.clear("bob", "general")
        print("\nConversations cleared.")


async def main():
    """Run all examples."""
    print("\nSyndicate SQLite/PostgreSQL Memory Examples\n")
    
    # Run examples
    await sqlite_example()
    await sqlite_with_rollover_example()
    await postgres_example()
    await multi_tenant_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())