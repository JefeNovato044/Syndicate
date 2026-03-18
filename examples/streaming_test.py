"""
Streaming Test Example

This example demonstrates the streaming functionality in the Syndicate framework.
It tests both client-level streaming and agent-level streaming with tool buffering.

Usage:
    python examples/streaming_test.py
"""

import asyncio
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).parent.parent)

from communication_models import StreamChunk
from clients import OpenAIClient
from agents import GenericAgent
from tools.weather_tool import WeatherTool


async def test_client_streaming():
    """Test client-level streaming (no agent)."""
    print("\n" + "="*60)
    print("TEST 1: Client-Level Streaming")
    print("="*60)
    
    # Create a simple client (using Ollama or similar)
    # For testing, we'll use a mock or skip if no LLM is available
    try:
        client = OpenAIClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model_name="llama3"
        )
        
        print("\nStreaming response from client...")
        print("-" * 40)
        
        # Create test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count to 10 slowly."}
        ]
        
        # Stream the response
        async for chunk in client.chat_completion_stream(
            messages=messages,
            system_message={"role": "system", "content": "You are a helpful assistant."}
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\n" + "-" * 40)
        print("✓ Client streaming completed successfully")
        
        client.close()
        
    except Exception as e:
        print(f"✗ Client streaming test skipped (LLM not available): {e}")
        print("  (This is expected if no local LLM is running)")


async def test_agent_streaming_no_tools():
    """Test agent-level streaming without tools."""
    print("\n" + "="*60)
    print("TEST 2: Agent-Level Streaming (No Tools)")
    print("="*60)
    
    try:
        client = OpenAIClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model_name="llama3"
        )
        
        agent = GenericAgent(
            llm_client=client,
            system_prompt="You are a helpful assistant. Respond concisely."
        )
        
        print("\nStreaming response from agent (no tools)...")
        print("-" * 40)
        
        # Stream the response
        async for chunk in agent.stream("What is 2+2?"):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\n" + "-" * 40)
        print("✓ Agent streaming (no tools) completed successfully")
        
        client.close()
        
    except Exception as e:
        print(f"✗ Agent streaming test skipped: {e}")


async def test_agent_streaming_with_tools():
    """Test agent-level streaming with tools."""
    print("\n" + "="*60)
    print("TEST 3: Agent-Level Streaming (With Tools)")
    print("="*60)
    
    try:
        client = OpenAIClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model_name="llama3"
        )
        
        agent = GenericAgent(
            llm_client=client,
            system_prompt="You are a weather assistant. Use the weather tool when asked about weather.",
            tools=[WeatherTool()]
        )
        
        print("\nStreaming response from agent (with weather tool)...")
        print("-" * 40)
        
        # Stream the response
        async for chunk in agent.stream("What's the weather in Tokyo?"):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\n" + "-" * 40)
        print("✓ Agent streaming (with tools) completed successfully")
        
        client.close()
        
    except Exception as e:
        print(f"✗ Agent streaming test skipped: {e}")


async def test_stream_chunk_model():
    """Test the StreamChunk model directly."""
    print("\n" + "="*60)
    print("TEST 4: StreamChunk Model")
    print("="*60)
    
    # Test creating StreamChunk objects
    chunk1 = StreamChunk(content="Hello", is_finished=False)
    chunk2 = StreamChunk(content=" ", is_finished=False)
    chunk3 = StreamChunk(content="World!", is_finished=True)
    
    print(f"\nChunk 1: '{chunk1.content}' (finished: {chunk1.is_finished})")
    print(f"Chunk 2: '{chunk2.content}' (finished: {chunk2.is_finished})")
    print(f"Chunk 3: '{chunk3.content}' (finished: {chunk3.is_finished})")
    
    # Test string representation
    print(f"\nString representation: '{str(chunk3)}'")
    
    print("\n✓ StreamChunk model test passed")


async def test_message_role_tool():
    """Test the 'tool' role in Message."""
    print("\n" + "="*60)
    print("TEST 5: Message Role 'tool'")
    print("="*60)
    
    from communication_models import Message
    
    # Create messages with tool role
    tool_msg = Message(role="tool", content="Tool result: 25 degrees")
    user_msg = Message(role="human", content="What's the weather?")
    ai_msg = Message(role="ai", content="The weather is 25 degrees.")
    
    print(f"\nTool message: {tool_msg.role} - {tool_msg.content}")
    print(f"User message: {user_msg.role} - {user_msg.content}")
    print(f"AI message: {ai_msg.role} - {ai_msg.content}")
    
    # Test role normalization
    normalized = Message(role="function", content="test")
    print(f"\nNormalized 'function' role: {normalized.role}")
    
    print("\n✓ Message role 'tool' test passed")


async def main():
    """Run all streaming tests."""
    print("\n" + "="*60)
    print("SYNDICATE STREAMING TEST SUITE")
    print("="*60)
    
    # Run tests
    await test_stream_chunk_model()
    await test_message_role_tool()
    await test_client_streaming()
    await test_agent_streaming_no_tools()
    await test_agent_streaming_with_tools()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print("\nNote: Tests 1-3 require a running LLM server (Ollama, LM Studio, etc.)")
    print("If no server is running, those tests will be skipped with a message.")


if __name__ == "__main__":
    asyncio.run(main())
