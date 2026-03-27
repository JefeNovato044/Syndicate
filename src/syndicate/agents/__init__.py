"""
Agents module - Base agent and generic agent implementations.

Async-First Architecture:
    - BaseAgent.invoke() is now async
    - GenericAgent.invoke() is now async
    - Use SyncGenericAgent for backward compatibility with sync code
    - Use GenericAgent for async workflows with asyncio

Example:
    # Async usage (recommended for production)
    agent = GenericAgent(client)
    response = await agent.invoke("Hello")
    
    # Sync usage (for simple scripts)
    agent = SyncGenericAgent(client)
    response = agent.invoke("Hello")
"""

from .base import BaseAgent
from .generic import GenericAgent
from .runtime import AgentRuntime

__all__ = ['BaseAgent', 'GenericAgent', 'AgentRuntime']
