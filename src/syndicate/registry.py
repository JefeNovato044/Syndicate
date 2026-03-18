from typing import Dict, Optional, List, TYPE_CHECKING
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from .agents.base import BaseAgent
    from .tools.base_tool import BaseTool


class AgentRegistry:
    """
    Central registry for agent discovery.
    
    Allows agents to:
    - Register themselves on creation
    - Find other agents by name
    - Get agents as tools for delegation
    
    Uses WeakValueDictionary to avoid memory leaks - 
    agents are automatically removed when no longer referenced.
    
    Thread safety:
        This is a process-global singleton. The lazy init of ``_agents``
        is safe under a single-threaded asyncio event loop but is NOT
        safe if ``register()`` / ``get()`` are called from multiple OS
        threads concurrently.  If you need multi-threaded access, wrap
        calls with a ``threading.Lock``.
    
    Testing:
        Call ``AgentRegistry.clear()`` in test fixtures (setup/teardown)
        to prevent cross-test state contamination.
    """
    
    _agents: Dict[str, "BaseAgent"] = None  # Initialized on first access
    
    @classmethod
    def _get_agents(cls) -> Dict[str, "BaseAgent"]:
        if cls._agents is None:
            cls._agents = WeakValueDictionary()
        return cls._agents
    
    @classmethod
    def register(cls, agent: "BaseAgent") -> None:
        """
        Register an agent for discovery.
        
        Args:
            agent: Agent instance to register
        """
        cls._get_agents()[agent.name] = agent
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            name: Agent name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in cls._get_agents():
            del cls._get_agents()[name]
            return True
        return False
    
    @classmethod
    def get(cls, name: str) -> Optional["BaseAgent"]:
        """
        Get an agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None if not found
        """
        return cls._get_agents().get(name)
    
    @classmethod
    def get_or_raise(cls, name: str) -> "BaseAgent":
        """
        Get an agent by name, raising if not found.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance
            
        Raises:
            KeyError: If agent not found
        """
        agent = cls.get(name)
        if agent is None:
            available = list(cls._get_agents().keys())
            raise KeyError(
                f"Agent '{name}' not found. "
                f"Available agents: {available}"
            )
        return agent
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """Get list of all registered agent names."""
        return list(cls._get_agents().keys())
    
    @classmethod
    def get_agent_as_tool(cls, name: str) -> "BaseTool":
        """
        Get an agent wrapped as a tool for delegation.
        
        Args:
            name: Agent name
            
        Returns:
            AgentAsTool instance
        """
        from .tools.agent_tool import AgentAsTool
        agent = cls.get_or_raise(name)
        return AgentAsTool(agent)
    
    @classmethod
    def get_all_as_tools(cls, exclude: Optional[List[str]] = None) -> List["BaseTool"]:
        """
        Get all registered agents as tools.
        
        Args:
            exclude: Optional list of agent names to exclude
            
        Returns:
            List of AgentAsTool instances
        """
        from .tools.agent_tool import AgentAsTool
        exclude = exclude or []
        return [
            AgentAsTool(agent) 
            for name, agent in cls._get_agents().items() 
            if name not in exclude
        ]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents. Useful for testing."""
        cls._get_agents().clear()
