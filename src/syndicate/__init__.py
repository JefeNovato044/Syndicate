"""
Syndicate AI - Agentic Framework

A Python framework for building AI agents with memory, tools, and skills.
"""

# Import core models for public API
from .communication_models import (
    Message,
    MessageBucket,
    ToolCall,
    ChatResponse,
    StreamChunk,
)

# Import registry for agent discovery
from .registry import AgentRegistry
from .mcp import MCPSessionManager, MCPSubTool

# Import agents module
from .agents import BaseAgent, GenericAgent

# Import clients module
from .clients import Client, GeminiClient, OpenAIClient

# Import memory module
from .memory import BaseChatMemory, LocalMemory

# Import skills module
from .skills import SkillModule, create_skill_module, SkillRegistry

__all__ = [
    # Core models
    'Message',
    'MessageBucket',
    'ToolCall',
    'ChatResponse',
    'StreamChunk',
    # Registries
    'AgentRegistry',
    'MCPSessionManager',
    'MCPSubTool',
    # Agents
    'BaseAgent',
    'GenericAgent',
    # Clients
    'Client',
    'GeminiClient',
    'OpenAIClient',
    # Memory
    'BaseChatMemory',
    'LocalMemory',
    # Skills
    'SkillModule',
    'create_skill_module',
    'SkillRegistry',
]

__version__ = "0.1.0"
