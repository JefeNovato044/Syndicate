"""
Syndicate - Agent Framework

Main entry point / quick smoke-test for the framework.
For full usage examples see the examples/ directory.
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


async def main():
    """Quick smoke-test: create a GenericAgent and invoke it."""
    from syndicate import GenericAgent, LocalMemory
    from syndicate.clients.gemini import GeminiClient

    client = GeminiClient(
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    memory = LocalMemory()

    agent = GenericAgent(
        llm_client=client,
        memory=memory,
        system_prompt="You are a helpful assistant. Be concise.",
    )

    print(f"Agent: {agent.name}  |  Provider: {agent.provider}")
    response = await agent.invoke("Say hello in one sentence.")
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
