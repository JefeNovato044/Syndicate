# agents/generic.py
"""
GenericAgent: Zero-boilerplate agent for prototyping.

This agent provides a simple, configurable interface for quick testing
and prototyping without requiring custom agent classes.

It relies 100% on the Motherboard's Hybrid API and Template Method pattern.
All orchestration logic is inherited from BaseAgent.

Example:
    from agents import GenericAgent
    from clients import GeminiClient
    
    client = GeminiClient(api_key="...")
    agent = GenericAgent(
        llm_client=client,
        system_prompt="You are a Python expert",
        tools=[CodeExecutorTool()]
    )
    
    response = agent.invoke_sync("Write a hello world")
"""

from typing import Optional, List
from .base import BaseAgent


class GenericAgent(BaseAgent):
    """
    A configurable agent with no predefined behavior.
    
    Perfect for:
    - Testing new tools
    - Quick prototypes
    - Interactive sessions (Jupyter notebooks)
    - Workflow experiments
    
    This agent relies 100% on the Motherboard's Hybrid API and Template Method pattern.
    All orchestration logic is inherited from BaseAgent.
    
    Attributes:
        system_prompt: Custom instructions for the agent (class-level default)
        All BaseAgent parameters (memory, skills, etc.)
    
    Example:
        # Minimal setup
        agent = GenericAgent(
            llm_client=client,
            system_prompt="You are helpful"
        )
        
        # With tools and memory
        agent = GenericAgent(
            llm_client=client,
            system_prompt="You are a researcher",
            tools=[SearchTool(), BrowserTool()],
            memory=MongoChatMemory(mongo_client, "db")
        )
    """
    
    # Declarative class attributes (Hybrid API - defaults)
    name: str = "GenericAgent"
    system_prompt: str = (
        "You are a helpful AI assistant. "
        "Respond clearly and concisely to user queries. "
        "Use available tools when appropriate to provide accurate information."
    )
    tools: list = None
    max_iterations: int = 5
    
    def __init__(
        self,
        llm_client,
        system_prompt: Optional[str] = None,
        memory=None,
        tools: Optional[List] = None,
        skills=None,
        name: Optional[str] = None,
        vision_client=None,
        audio_client=None,
        **kwargs
    ):
        """
        Initialize a GenericAgent.
        
        Args:
            llm_client: The LLM client to use (required)
            system_prompt: Custom system instructions (optional, overrides class default)
            memory: Memory implementation (optional)
            tools: List of tools to register (optional)
            skills: Domain knowledge modules to inject (optional)
            name: Agent identifier for logging (optional, overrides class default)
            vision_client: Vision model client (optional)
            audio_client: Audio model client (optional)
            **kwargs: Additional BaseAgent parameters (e.g., max_iterations)
        
        Example:
            from core.clients import GeminiClient
            from core.memory import MongoChatMemory
            from core.tools import WeatherTool
            
            client = GeminiClient(api_key="...")
            memory = MongoChatMemory(mongo_client, "mydb")
            
            agent = GenericAgent(
                llm_client=client,
                system_prompt="You are a weather assistant",
                tools=[WeatherTool()],
                memory=memory
            )
        """
        # Hybrid API: kwargs override class attributes
        super().__init__(
            llm_client=llm_client,
            memory=memory,
            system_prompt=system_prompt,
            name=name,
            tools=tools,
            skills=skills,
            vision_client=vision_client,
            audio_client=audio_client,
            **kwargs
        )
