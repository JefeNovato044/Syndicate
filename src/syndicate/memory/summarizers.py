# core/memory/summarizers.py
"""
Summarization utilities for memory bucket rollover.

Provides a tiered abstraction for summarization:
- Callable: Simple function for basic summarization (recommended for most cases)
- Agent: Full agent for complex summarization (RAG-enhanced, multi-step, etc.)

The memory module normalizes both to a common interface internally.
"""

from typing import Protocol, Union, List, Callable, Awaitable, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agents.base import BaseAgent
    from ..clients import Client

from ..communication_models import Message


class SummarizerCallable(Protocol):
    """
    Protocol for summarizer callables.
    
    Any async function matching this signature can be used as a summarizer:
        async def my_summarizer(messages: List[Message]) -> str: ...
    """
    async def __call__(self, messages: List[Message]) -> str: ...


# Type alias for what developers can provide as a summarizer
Summarizer = Union["BaseAgent", SummarizerCallable, Callable[[List[Message]], Awaitable[str]]]


# =============================================================================
# RESOLUTION & EXECUTION
# =============================================================================

async def resolve_summarizer(
    summarizer: Summarizer,
    messages: List[Message]
) -> str:
    """
    Execute any summarizer type and return the summary string.
    
    Normalizes different summarizer types to a common execution path:
    - BaseAgent: Calls agent.invoke() with formatted conversation prompt
    - Callable: Direct invocation with messages list
    
    Args:
        summarizer: An Agent instance or async callable
        messages: List of messages to summarize
        
    Returns:
        Summary string
        
    Raises:
        TypeError: If summarizer is not a recognized type
        ValueError: If summarizer returns invalid response
    """
    # Avoid circular import - check at runtime
    from ..agents.base import BaseAgent
    
    if isinstance(summarizer, BaseAgent):
        # Agent path - format messages as prompt and invoke the agent.
        # Use dedicated owner/chat IDs to avoid polluting user conversations.
        formatted = format_messages_for_summary(messages)
        response = await summarizer.invoke(
            formatted,
            owner_id="memory-summarizer",
            chat_id="memory-summarizer",
        )
        
        # Validate agent response
        if not isinstance(response, str):
            raise ValueError(
                f"Summarization agent returned invalid response type: {type(response)}. "
                "Expected a string response from agent.invoke()."
            )
        return response
    
    elif callable(summarizer):
        # Callable path - direct invocation
        result = await summarizer(messages)
        
        # Validate callable response
        if not isinstance(result, str):
            raise ValueError(
                f"Summarizer callable returned {type(result)}, expected str."
            )
        return result
    
    else:
        raise TypeError(
            f"Summarizer must be a BaseAgent or async Callable, got {type(summarizer)}"
        )


def format_messages_for_summary(messages: List[Message]) -> str:
    """
    Format messages into a summarization prompt for agents.
    
    Creates a structured conversation representation that's easy for
    LLMs to understand and summarize.
    
    Args:
        messages: List of Message objects
        
    Returns:
        Formatted prompt string
    """
    lines = []
    for msg in messages:
        role = "User" if msg.role == "human" else "Assistant"
        # Truncate very long messages to avoid token bloat
        content = msg.content[:2000] + "..." if len(msg.content) > 2000 else msg.content
        lines.append(f"[{role}]: {content}")
    
    conversation = "\n".join(lines)
    
    return (
        "Summarize the following conversation concisely. "
        "Capture the key topics discussed, decisions made, and any important context. "
        "Keep the summary under 500 words.\n\n"
        f"CONVERSATION:\n{conversation}\n\n"
        "SUMMARY:"
    )


# =============================================================================
# DEFAULT SUMMARIZER FACTORY
# =============================================================================

def create_default_summarizer(llm_client: Client) -> SummarizerCallable:
    """
    Factory that creates a summarizer callable bound to a specific LLM client.
    
    Use this when you want simple summarization without creating a full agent.
    The memory module stays decoupled from LLM client specifics.
    
    Args:
        llm_client: Any LLM client with chat_completion_async() method
        
    Returns:
        Async callable suitable for use as a summarizer
        
    Example:
        from syndicate.clients.openai import OpenAIClient
        
        client = OpenAIClient(base_url="...", model_name="...")
        summarizer = create_default_summarizer(client)
        
        memory = MongoMemory(
            rollover_enabled=True,
            summarizer=summarizer
        )
    """
    async def summarize(messages: List[Message]) -> str:
        from ..communication_models import Message as Msg
        
        prompt = format_messages_for_summary(messages)
        
        # Try Syndicate client method (chat_completion_async)
        if hasattr(llm_client, 'chat_completion_async'):
            user_msg = Msg(role="human", content=prompt)
            system_msg = Msg(role="system", content="You are a helpful assistant that summarizes conversations concisely.")
            response = await llm_client.chat_completion_async(
                messages=[user_msg],
                system_message=system_msg
            )
        # Fallback to common LLM client method signatures
        elif hasattr(llm_client, 'generate'):
            response = await llm_client.generate(prompt)
        elif hasattr(llm_client, 'chat'):
            response = await llm_client.chat([{"role": "user", "content": prompt}])
        elif hasattr(llm_client, 'complete'):
            response = await llm_client.complete(prompt)
        else:
            raise TypeError(
                f"LLM client {type(llm_client)} does not have a recognized method "
                "(chat_completion_async, generate, chat, or complete)"
            )
        
        # Extract content from response
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        else:
            raise ValueError(f"Could not extract content from LLM response: {type(response)}")
    
    return summarize


# =============================================================================
# SUMMARY AGGREGATION (for hierarchical summarization)
# =============================================================================

async def summarize_summaries(
    summaries: List[str],
    summarizer: Summarizer
) -> str:
    """
    Create a meta-summary from multiple bucket summaries.
    
    Used for hierarchical summarization when a conversation spans
    many buckets. Condenses all previous summaries into one cohesive
    context block.
    
    Args:
        summaries: List of individual bucket summaries
        summarizer: Summarizer to use for aggregation
        
    Returns:
        Aggregated meta-summary
    """
    if not summaries:
        return ""
    
    if len(summaries) == 1:
        return summaries[0]
    
    # Create a fake "message" list for the summarizer
    # Each summary becomes a "message" to summarize
    pseudo_messages = [
        Message(role="ai", content=f"[Conversation Part {i+1}]: {summary}")
        for i, summary in enumerate(summaries)
    ]
    
    return await resolve_summarizer(summarizer, pseudo_messages)


__all__ = [
    'SummarizerCallable',
    'Summarizer',
    'resolve_summarizer',
    'format_messages_for_summary',
    'create_default_summarizer',
    'summarize_summaries',
]
