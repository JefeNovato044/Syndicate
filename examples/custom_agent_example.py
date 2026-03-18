"""
Custom Agent Example

Demonstrates how to create a custom agent by extending BaseAgent.
Shows async implementation, tool integration, and invoke usage.

Run with: python examples/custom_agent_example.py
"""

import asyncio
import time
from typing import Optional, List
from agents import BaseAgent
from clients import GeminiClient
from tools.base_tool import BaseTool
from communication_models import Message


class ResearchAgent(BaseAgent):
    """
    A custom research agent that specializes in gathering information.
    
    This example shows how to:
    1. Extend BaseAgent
    2. Implement the async invoke() method
    3. Add custom tools
    4. Use memory for conversation history
    """
    
    def __init__(
        self,
        llm_client: GeminiClient,
        system_prompt: str = "You are a research assistant. Provide detailed, well-structured answers.",
        name: str = "ResearchAgent",
        tools: Optional[List[BaseTool]] = None,
        memory=None
    ):
        super().__init__(
            llm_client=llm_client,
            memory=memory,
            system_prompt=system_prompt,
            name=name,
            tools=tools or []
        )
    
    async def invoke(self, user_input: str, owner_id: str = "default", chat_id: str = "default") -> str:
        """
        Custom invoke implementation for research agent.
        
        This method can be customized to:
        - Add custom pre-processing
        - Implement special logic
        - Add custom post-processing
        - Use different tool calling strategies
        
        Args:
            user_input: The user's research query
            owner_id: User identifier for memory
            chat_id: Conversation identifier for memory
            
        Returns:
            Research response
        """
        # Custom pre-processing: Extract key terms
        query = self._preprocess_query(user_input)
        
        # Build messages with history
        messages = self._build_messages(query, owner_id, chat_id)
        
        # Format tools for the LLM provider
        formatted_tools = self._format_tools()
        
        try:
            # Get response from LLM
            if formatted_tools:
                response = await self.llm.chat_completion_async(
                    messages=messages,
                    system_message=self.system_prompt,
                    tools=formatted_tools
                )
            else:
                response = await self.llm.chat_completion_async(
                    messages=messages,
                    system_message=self.system_prompt
                )
            
            # Handle tool calling loop
            if response.tool_calls:
                final_response = await self._execute_tool_loop(messages, formatted_tools)
                # Store in memory
                await self._store_interaction(query, final_response, owner_id, chat_id)
                return final_response
            
            # No tool calls - direct response
            await self._store_interaction(query, response.content, owner_id, chat_id)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            error_msg = f"Research error: {str(e)}"
            print(f"[{self.name}] {error_msg}")
            return error_msg
    
    def _preprocess_query(self, query: str) -> str:
        """
        Custom query preprocessing.
        
        Override this to add custom logic like:
        - Query expansion
        - Entity extraction
        - Query rewriting
        """
        # Example: Add context if query is short
        if len(query.split()) < 3:
            return f"Research topic: {query}. Please provide comprehensive information."
        return query


class CodeReviewAgent(BaseAgent):
    """
    A custom code review agent.
    
    This example shows:
    1. Custom system prompt
    2. Specialized tool usage
    3. Async implementation
    """
    
    def __init__(
        self,
        llm_client: GeminiClient,
        code_language: str = "python",
        name: str = "CodeReviewAgent"
    ):
        system_prompt = (
            f"You are a code review expert specializing in {code_language}. "
            "Provide constructive feedback on code quality, security, performance, and best practices."
        )
        
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            name=name,
            tools=[]  # Can add tools later
        )
        
        self.code_language = code_language
    
    async def invoke(self, user_input: str, owner_id: str = "default", chat_id: str = "default") -> str:
        """
        Custom invoke for code review.
        
        Args:
            user_input: Code to review (can include code block)
            owner_id: User identifier
            chat_id: Conversation identifier
            
        Returns:
            Code review feedback
        """
        # Extract code from user input if present
        code = self._extract_code(user_input)
        
        # Build prompt with code context
        enhanced_input = f"Please review this {self.code_language} code:\n\n{code}"
        
        # Build messages
        messages = self._build_messages(enhanced_input, owner_id, chat_id)
        
        try:
            response = await self.llm.chat_completion_async(
                messages=messages,
                system_message=self.system_prompt
            )
            
            await self._store_interaction(enhanced_input, response.content, owner_id, chat_id)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"Code review error: {str(e)}"
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code from text (simple implementation).
        Override for more sophisticated extraction.
        """
        # Simple heuristic: look for code blocks
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        return text


# Example tools for custom agents
class WebSearchTool(BaseTool):
    """Example tool for research agent."""
    name = "web_search"
    description = "Search the web for information"
    
    async def run(self, query: str) -> str:
        # Simulate async web search
        await asyncio.sleep(0.5)
        return f"Search results for '{query}': Found 10 relevant articles"


class CodeAnalysisTool(BaseTool):
    """Example tool for code review agent."""
    name = "code_analysis"
    description = "Analyze code for potential issues"
    
    async def run(self, code: str) -> str:
        # Simulate async code analysis
        await asyncio.sleep(0.3)
        return "Analysis: No critical issues found. Minor improvements suggested for readability."


async def example_custom_agent():
    """Example using a custom ResearchAgent."""
    print("=" * 60)
    print("Example 1: Custom ResearchAgent")
    print("=" * 60)
    
    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE"
    )
    
    # Create custom agent with tools
    agent = ResearchAgent(
        llm_client=client,
        system_prompt="You are a research assistant. Provide detailed, well-structured answers.",
        tools=[WebSearchTool()]
    )
    
    # Use async invoke
    response = await agent.invoke("What are the latest developments in AI?")
    print(f"Response: {response}\n")


async def example_code_review_agent():
    """Example using a custom CodeReviewAgent."""
    print("=" * 60)
    print("Example 2: Custom CodeReviewAgent")
    print("=" * 60)
    
    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE"
    )
    
    # Create code review agent
    agent = CodeReviewAgent(
        llm_client=client,
        code_language="python"
    )
    
    code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
    
    # Use async invoke
    response = await agent.invoke(f"Review this code:\n{code}")
    print(f"Response: {response}\n")


async def example_custom_invoke_logic():
    """Example showing custom invoke logic."""
    print("=" * 60)
    print("Example 3: Custom Invoke Logic")
    print("=" * 60)
    
    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE"
    )
    
    class CustomAgent(BaseAgent):
        def __init__(self, llm_client):
            super().__init__(
                llm_client=llm_client,
                system_prompt="You are a helpful assistant."
            )
        
        async def invoke(self, user_input: str, **kwargs) -> str:
            # Custom logic: Add timestamp to all responses
            timestamp = asyncio.get_event_loop().time()
            
            # Call parent invoke
            response = await super().invoke(user_input, **kwargs)
            
            # Custom post-processing
            return f"[{timestamp:.2f}s] {response}"
    
    agent = CustomAgent(client)
    response = await agent.invoke("Hello!")
    print(f"Response with custom logic: {response}\n")


async def example_multiple_custom_agents():
    """Example running multiple custom agents in parallel."""
    print("=" * 60)
    print("Example 4: Multiple Custom Agents in Parallel")
    print("=" * 60)
    
    client = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        api_key="YOUR_API_KEY_HERE"
    )
    
    # Create multiple custom agents
    research_agent = ResearchAgent(
        llm_client=client,
        name="ResearchAgent"
    )
    
    code_agent = CodeReviewAgent(
        llm_client=client,
        code_language="javascript",
        name="JSReviewAgent"
    )
    
    # Run agents in parallel
    print("Running multiple custom agents concurrently...")
    start_time = time.time()
    
    tasks = [
        research_agent.invoke("What is async/await?"),
        code_agent.invoke("Review this JS code:\nconsole.log('hello');"),
    ]
    
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    for i, result in enumerate(results, 1):
        print(f"\nAgent {i}:\n  {result}")
    print()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SYNDICATE CUSTOM AGENT EXAMPLES")
    print("=" * 60 + "\n")
    
    try:
        await example_custom_agent()
        await example_code_review_agent()
        await example_custom_invoke_logic()
        await example_multiple_custom_agents()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: Make sure to set your Gemini API key in the code.")
        print("Replace 'YOUR_API_KEY_HERE' with your actual API key.")


if __name__ == "__main__":
    asyncio.run(main())
