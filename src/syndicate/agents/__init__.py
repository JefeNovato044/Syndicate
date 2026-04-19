"""
Agents module - Base agent and generic agent implementations.

Async-First Architecture:
    - BaseAgent.invoke() is now async
    - GenericAgent.invoke() is now async
    - Use invoke_sync() when you need blocking usage in sync contexts
    - Use GenericAgent for async workflows with asyncio

Example:
    # Async usage (recommended for production)
    agent = GenericAgent(client)
    response = await agent.invoke("Hello")
    
    # Sync usage (for simple scripts)
    agent = GenericAgent(client)
    response = agent.invoke_sync("Hello")
"""

from .base import BaseAgent
from .generic import GenericAgent
from .runtime import AgentRuntime

__all__ = ['BaseAgent', 'GenericAgent', 'AgentRuntime']
