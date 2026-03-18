"""
OpenAI-Compatible Client Usage Example

Demonstrates how to use the OpenAIClient for local models (Ollama, LM Studio, vLLM)
and OpenAI cloud API. This enables using local LLMs with the Syndicate framework.

Run with: python examples/openai_usage_example.py

Prerequisites:
- Ollama running: ollama serve
- Or LM Studio running: LM Studio server
- Or OpenAI API key

Example models:
- Ollama: llama3, mistral, codellama, qwen2.5
- LM Studio: any model loaded in the UI
- OpenAI: gpt-4, gpt-3.5-turbo
"""

import asyncio
from clients import OpenAIClient
from agents import GenericAgent
from tools.weather_tool import WeatherTool


def example_ollama():
    """Example using Ollama (local model)."""
    print("=" * 60)
    print("Example 1: Ollama (Local Model)")
    print("=" * 60)
    
    # Initialize client for Ollama
    client = OpenAIClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't require a real key
        model_name="llama3",  # Or: mistral, codellama, qwen2.5
        temperature=0.7
    )
    
    # Create agent
    agent = GenericAgent(
        llm_client=client,
        system_prompt="You are a helpful assistant using Ollama.",
        name="OllamaAgent"
    )
    
    # Use sync invoke
    response = agent.invoke("What is the capital of France?")
    print(f"Response: {response}\n")


async def example_lm_studio():
    """Example using LM Studio (local model)."""
    print("=" * 60)
    print("Example 2: LM Studio (Local Model)")
    print("=" * 60)
    
    # Initialize client for LM Studio
    client = OpenAIClient(
        base_url="http://localhost:1234/v1",
        api_key="dummy",  # LM Studio doesn't require a real key
        model_name="local-model",  # Model name as shown in LM Studio
        temperature=0.7
    )
    
    # Create agent
    agent = GenericAgent(
        llm_client=client,
        system_prompt="You are a helpful assistant using LM Studio.",
        name="LMStudioAgent"
    )
    
    # Use async invoke
    response = await agent.invoke("Write a haiku about coding.")
    print(f"Response: {response}\n")


async def example_openai_cloud():
    """Example using OpenAI cloud API."""
    print("=" * 60)
    print("Example 3: OpenAI Cloud API")
    print("=" * 60)
    
    # Initialize client for OpenAI
    client = OpenAIClient(
        base_url="https://api.openai.com/v1",
        api_key="YOUR_OPENAI_API_KEY_HERE",  # Replace with your actual key
        model_name="gpt-4",
        temperature=0.7
    )
    
    # Create agent
    agent = GenericAgent(
        llm_client=client,
        system_prompt="You are a helpful assistant using OpenAI.",
        name="OpenAIAgent"
    )
    
    # Use async invoke
    response = await agent.invoke("Explain quantum computing in simple terms.")
    print(f"Response: {response}\n")


async def example_with_tools():
    """Example using OpenAI client with tools."""
    print("=" * 60)
    print("Example 4: OpenAI Client with Tools")
    print("=" * 60)
    
    # Initialize client for Ollama
    client = OpenAIClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model_name="llama3",
        temperature=0.7
    )
    
    # Create agent with weather tool
    agent = GenericAgent(
        llm_client=client,
        system_prompt="You are a weather assistant with access to weather tools.",
        tools=[WeatherTool()]
    )
    
    # Use async invoke - agent will call the tool
    response = await agent.invoke("What's the weather in Tokyo?")
    print(f"Response: {response}\n")


async def example_concurrent_local_models():
    """Example running multiple local models concurrently."""
    print("=" * 60)
    print("Example 5: Concurrent Local Models")
    print("=" * 60)
    
    # Create multiple clients for different models
    clients = [
        OpenAIClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model_name="llama3",
            name="llama3"
        ),
        OpenAIClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model_name="mistral",
            name="mistral"
        )
    ]
    
    # Create agents for each model
    agents = [
        GenericAgent(
            llm_client=client,
            system_prompt="You are a helpful assistant.",
            name=client.name
        )
        for client in clients
    ]
    
    # Run agents in parallel
    print("Running multiple local models concurrently...")
    start_time = asyncio.get_event_loop().time()
    
    tasks = [
        agent.invoke("What is 2+2?")
        for agent in agents
    ]
    
    results = await asyncio.gather(*tasks)
    
    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    for i, (agent, response) in enumerate(zip(agents, results), 1):
        print(f"\n{agent.name}:")
        print(f"  {response}")
    print()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SYNDICATE OPENAI-COMPATIBLE CLIENT EXAMPLES")
    print("=" * 60 + "\n")
    
    print("Note: Make sure your local model server is running:")
    print("  - Ollama:  ollama serve")
    print("  - LM Studio: Start the server in LM Studio")
    print()
    
    try:
        # Example 1: Ollama (local)
        example_ollama()
        
        # Example 2: LM Studio (local)
        await example_lm_studio()
        
        # Example 3: OpenAI cloud (requires API key)
        # Uncomment to test with OpenAI:
        # await example_openai_cloud()
        
        # Example 4: With tools
        await example_with_tools()
        
        # Example 5: Concurrent local models
        await example_concurrent_local_models()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your local model server is running")
        print("2. Check the base_url is correct (default: http://localhost:11434/v1)")
        print("3. Verify the model name is correct (llama3, mistral, etc.)")


if __name__ == "__main__":
    asyncio.run(main())
