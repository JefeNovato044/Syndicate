"""
Async Agent Usage Example

Demonstrates the async-first architecture of Syndicate.
Shows how to use both sync and async agents, concurrent tool execution,
and the asyncio.gather() pattern for parallel operations.

Run with: python examples/async_usage_example.py
"""

import asyncio
import time
from agents import GenericAgent, SyncGenericAgent
from clients import GeminiClient
from tools.weather_tool import WeatherTool


async def example_async_agent():
    """Example using async GenericAgent."""
    print("=" * 60)
    print("Example 1: Async GenericAgent")
    print("=" * 60)
    
    # Initialize client
    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE"
    )
    
    # Create async agent with tools
    agent = GenericAgent(
        llm_client=client,
        system_prompt="You are a helpful assistant with weather knowledge.",
        tools=[WeatherTool()]
    )
    
    # Use async invoke
    response = await agent.invoke("What's the weather in Tokyo?")
    print(f"Response: {response}\n")


async def example_concurrent_tools():
    """Example demonstrating concurrent tool execution."""
    print("=" * 60)
    print("Example 2: Concurrent Tool Execution")
    print("=" * 60)
    
    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE"
    )
    
    agent = GenericAgent(
        llm_client=client,
        system_prompt="You are a research assistant.",
        tools=[WeatherTool()]
    )
    
    # Simulate multiple concurrent tool calls
    print("Executing multiple tool calls concurrently...")
    start_time = time.time()
    
    # Create multiple async tasks
    tasks = [
        agent.execute_tool("get_weather", city="Tokyo"),
        agent.execute_tool("get_weather", city="New York"),
        agent.execute_tool("get_weather", city="London"),
    ]
    
    # Execute all concurrently
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    for i, result in enumerate(results, 1):
        print(f"  Tool {i}: {result}")
    print()


async def example_multi_agent_parallel():
    """Example of running multiple agents in parallel."""
    print("=" * 60)
    print("Example 3: Multi-Agent Parallel Execution")
    print("=" * 60)
    
    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE"
    )
    
    # Create multiple agents
    agent1 = GenericAgent(
        llm_client=client,
        system_prompt="You are a Python expert.",
        name="PythonExpert"
    )
    
    agent2 = GenericAgent(
        llm_client=client,
        system_prompt="You are a JavaScript expert.",
        name="JSExpert"
    )
    
    # Run agents in parallel
    print("Running multiple agents concurrently...")
    start_time = time.time()
    
    tasks = [
        agent1.invoke("Explain async/await in Python"),
        agent2.invoke("Explain async/await in JavaScript"),
    ]
    
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    for i, (agent, response) in enumerate(zip(["PythonExpert", "JSExpert"], results), 1):
        print(f"\n{agent}:")
        print(f"  {response}")
    print()


def example_sync_agent():
    """Example using sync SyncGenericAgent for backward compatibility."""
    # print("=" * 60)
    # print("Example 4: Sync GenericAgent (Backward Compatible)")
    # print("=" * 60)
    # 
    # client = GeminiClient(
    #     model_name="gemini-2.0-flash-exp",
    #     api_key="YOUR_API_KEY_HERE"
    # )
    # 
    # # Create sync agent
    # agent = SyncGenericAgent(
    #     llm_client=client,
    #     system_prompt="You are a helpful assistant.",
    #     name="SyncAgent"
    # )
    # 
    # # Use sync invoke (no await needed)
    # response = agent.invoke("Hello, how are you?")
    # print(f"Response: {response}\n")
    pass

async def example_async_tool():
    """Example of creating an async tool."""
    print("=" * 60)
    print("Example 5: Async Tool Implementation")
    print("=" * 60)
    
    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE"
    )
    
    # Create an async tool
    class AsyncWeatherTool:
        name = "async_weather"
        description = "Get weather asynchronously (simulated)"
        
        async def run(self, city: str) -> str:
            # Simulate async operation
            await asyncio.sleep(1)
            return f"Async weather for {city}: 72°F, sunny"
    
    agent = GenericAgent(
        llm_client=client,
        system_prompt="You are a weather assistant.",
        tools=[AsyncWeatherTool()]
    )
    
    # Execute async tool
    result = await agent.execute_tool("async_weather", city="San Francisco")
    print(f"Async tool result: {result}\n")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SYNDICATE ASYNC AGENT EXAMPLES")
    print("=" * 60 + "\n")
    
    try:
        await example_async_agent()
        await example_concurrent_tools()
        await example_multi_agent_parallel()
        example_sync_agent()
        await example_async_tool()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: Make sure to set your Gemini API key in the code.")
        print("Replace 'YOUR_API_KEY_HERE' with your actual API key.")


if __name__ == "__main__":
    asyncio.run(main())
