"""
Custom Agent Example

Demonstrates how to create custom agents by extending BaseAgent.

Important:
- Do not override invoke() or stream() in custom agents.
- Override _run_agent(messages, **kwargs) for custom behavior.

Run with: python examples/custom_agent_example.py
"""

import asyncio
import time
from typing import List, Optional

from pydantic import BaseModel, Field

from syndicate.agents import BaseAgent
from syndicate.clients import GeminiClient
from syndicate.communication_models import Message
from syndicate.tools.base_tool import BaseTool


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query")


class WebSearchTool(BaseTool):
    """Simple async demo tool for search-like responses."""

    name = "web_search"
    description = "Search the web for information (simulated)."
    args_schema = WebSearchArgs

    async def run(self, **kwargs) -> str:
        args = self.args_schema(**kwargs)
        await asyncio.sleep(0.4)
        return f"Search results for '{args.query}': Found 10 relevant articles"


class CodeAnalysisArgs(BaseModel):
    code: str = Field(..., description="Code snippet to analyze")


class CodeAnalysisTool(BaseTool):
    """Simple async demo tool for code analysis."""

    name = "code_analysis"
    description = "Analyze code for potential issues (simulated)."
    args_schema = CodeAnalysisArgs

    async def run(self, **kwargs) -> str:
        args = self.args_schema(**kwargs)
        await asyncio.sleep(0.2)
        lines = len(args.code.splitlines())
        return f"Analyzed {lines} lines. No critical issues found; improve naming/readability."


class ResearchAgent(BaseAgent):
    """Custom agent that preprocesses short research prompts before orchestration."""

    name = "ResearchAgent"
    system_prompt = (
        "You are a research assistant. Provide detailed, well-structured answers. "
        "Use tools when they can improve factual accuracy."
    )

    def __init__(
        self,
        llm_client: GeminiClient,
        memory=None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs,
    ):
        super().__init__(
            llm_client=llm_client,
            memory=memory,
            tools=tools or [WebSearchTool()],
            **kwargs,
        )

    def _preprocess_query(self, query: str) -> str:
        if len(query.split()) < 4:
            return f"Research topic: {query}. Provide a concise but complete overview."
        return query

    async def _run_agent(self, messages: List[Message], **kwargs) -> str:
        # Customize only the last user turn before the default orchestration loop.
        if messages and messages[-1].role == "human":
            messages[-1].content = self._preprocess_query(messages[-1].content)

        return await super()._run_agent(messages, **kwargs)


class CodeReviewAgent(BaseAgent):
    """Custom agent that reframes user input into structured code-review prompts."""

    name = "CodeReviewAgent"

    def __init__(self, llm_client: GeminiClient, code_language: str = "python", memory=None, **kwargs):
        self.code_language = code_language
        prompt = (
            f"You are a code-review expert for {code_language}. "
            "Focus on correctness, maintainability, security, and performance."
        )

        super().__init__(
            llm_client=llm_client,
            memory=memory,
            system_prompt=prompt,
            tools=[CodeAnalysisTool()],
            **kwargs,
        )

    def _extract_code(self, text: str) -> str:
        if "```" not in text:
            return text

        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
        return text

    async def _run_agent(self, messages: List[Message], **kwargs) -> str:
        if messages and messages[-1].role == "human":
            code = self._extract_code(messages[-1].content)
            messages[-1].content = (
                f"Review this {self.code_language} code and suggest improvements:\n\n{code}"
            )

        return await super()._run_agent(messages, **kwargs)


class TimestampedAgent(BaseAgent):
    """Custom agent that post-processes the final response."""

    name = "TimestampedAgent"
    system_prompt = "You are a helpful assistant."

    async def _run_agent(self, messages: List[Message], **kwargs) -> str:
        started = time.perf_counter()
        response = await super()._run_agent(messages, **kwargs)
        elapsed = time.perf_counter() - started
        return f"[{elapsed:.2f}s] {response}"


async def example_custom_research_agent():
    print("=" * 60)
    print("Example 1: Custom ResearchAgent")
    print("=" * 60)

    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE",
    )

    agent = ResearchAgent(llm_client=client)
    response = await agent.invoke("Latest developments in AI")
    print(f"Response: {response}\n")


async def example_custom_code_review_agent():
    print("=" * 60)
    print("Example 2: Custom CodeReviewAgent")
    print("=" * 60)

    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE",
    )

    agent = CodeReviewAgent(llm_client=client, code_language="python")
    code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
    response = await agent.invoke(f"Review this code:\n{code}")
    print(f"Response: {response}\n")


async def example_custom_post_processing():
    print("=" * 60)
    print("Example 3: Custom Post-Processing")
    print("=" * 60)

    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE",
    )

    agent = TimestampedAgent(llm_client=client, memory=None)
    response = await agent.invoke("Say hello in one sentence.")
    print(f"Response: {response}\n")


async def example_parallel_custom_agents():
    print("=" * 60)
    print("Example 4: Multiple Custom Agents in Parallel")
    print("=" * 60)

    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE",
    )

    research_agent = ResearchAgent(llm_client=client, name="ResearchAgent")
    review_agent = CodeReviewAgent(llm_client=client, code_language="javascript", name="JSReviewAgent")

    print("Running multiple custom agents concurrently...")
    start_time = time.time()

    tasks = [
        research_agent.invoke("What is async/await?"),
        review_agent.invoke("Review this JS code:\nconsole.log('hello');"),
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    for i, result in enumerate(results, 1):
        print(f"\nAgent {i}:\n  {result}")
    print()


async def main():
    print("\n" + "=" * 60)
    print("SYNDICATE CUSTOM AGENT EXAMPLES")
    print("=" * 60 + "\n")

    try:
        await example_custom_research_agent()
        await example_custom_code_review_agent()
        await example_custom_post_processing()
        await example_parallel_custom_agents()

        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: Make sure to set your Gemini API key in the code.")
        print("Replace 'YOUR_API_KEY_HERE' with your actual API key.")


if __name__ == "__main__":
    asyncio.run(main())
