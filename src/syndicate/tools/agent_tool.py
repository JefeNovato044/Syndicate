from typing import TYPE_CHECKING, Optional, Any
from pydantic import BaseModel, Field
from .base_tool import BaseTool

if TYPE_CHECKING:
    from ..agents.base import BaseAgent


class AgentDelegationArgs(BaseModel):
    """Arguments for delegating a task to another agent."""
    
    task: str = Field(
        ..., 
        description="The task or question to delegate to the agent. Be specific and provide all necessary context."
    )
    context: Optional[str] = Field(
        None,
        description="Additional context that might help the agent complete the task."
    )


class AgentAsTool(BaseTool):
    """
    Wraps any BaseAgent as a tool for agent-to-agent delegation.
    
    This enables:
    - Kiriko can delegate to ResearchAgent
    - Any agent can delegate to DocumentWriter
    - Hierarchical agent architectures
    - Specialist agent composition
    
    Example:
        research_agent = ResearchAgent(llm_client=client)
        kiriko = create_kiriko(
            llm_client=client,
            additional_tools=[AgentAsTool(research_agent)]
        )
        # Now Kiriko can delegate research tasks
    """
    
    args_schema = AgentDelegationArgs
    
    def __init__(self, agent: "BaseAgent", custom_description: Optional[str] = None):
        """
        Wrap an agent as a tool.
        
        Args:
            agent: The agent to wrap
            custom_description: Override the auto-generated description
        """
        self.agent = agent
        self._custom_description = custom_description
    
    @property
    def name(self) -> str:
        """Tool name derived from agent name."""
        # Sanitize agent name for tool compatibility
        safe_name = self.agent.name.lower().replace(" ", "_").replace("-", "_")
        return f"delegate_to_{safe_name}"
    
    @property
    def description(self) -> str:
        """Tool description from agent info."""
        if self._custom_description:
            return self._custom_description
        
        # Build description directly from agent's system prompt (avoid get_info — it's async-unsafe)
        system_prompt = getattr(self.agent, '_base_system_prompt', '') or getattr(self.agent, 'system_prompt', '')
        # system_prompt may be a Message object — extract content string if so
        if hasattr(system_prompt, 'content'):
            system_prompt = system_prompt.content
        
        # Truncate if too long
        prompt_preview = system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt
        
        return (
            f"Delegate a task to {self.agent.name}. "
            f"This agent specializes in: {prompt_preview}"
        )
    
    def run(self, **kwargs) -> str:
        """
        Sync fallback — not used by the framework (run_async is called instead).
        Exists for direct manual invocation from sync contexts only.
        """
        args = self.args_schema(**kwargs)
        full_message = f"{args.task}\n\nAdditional context:\n{args.context}" if args.context else args.task
        return self.agent.invoke_sync(full_message, owner_id="delegation", chat_id="delegation")

    async def run_async(self, **kwargs) -> str:
        """
        Execute delegation to the wrapped agent (async-native).
        
        Bypasses the sync→to_thread→invoke_sync round-trip entirely.
        The framework calls this method directly during tool execution.
        
        Args:
            **kwargs: Arguments matching AgentDelegationArgs
                - task: The task to delegate
                - context: Optional additional context
                
        Returns:
            The delegated agent's response
        """
        args = self.args_schema(**kwargs)
        full_message = f"{args.task}\n\nAdditional context:\n{args.context}" if args.context else args.task
        return await self.agent.invoke(
            full_message,
            owner_id="delegation",
            chat_id="delegation",
        )
    
    # Override class methods to work with instance properties
    @classmethod
    def get_gemini_tool_schema(cls):
        """Cannot use classmethod - need instance."""
        raise NotImplementedError("Use instance.to_format() instead")
    
    @classmethod
    def get_openai_tool_schema(cls):
        """Cannot use classmethod - need instance."""
        raise NotImplementedError("Use instance.to_format() instead")
    
    def to_format(self, format_type: str):
        """
        Convert to provider-specific format.
        Overrides class method to use instance properties.
        """
        if format_type == "gemini":
            from .base_tool import _clean_schema_for_gemini
            return {
                "name": self.name,
                "description": self.description,
                "parameters": _clean_schema_for_gemini(
                    self.args_schema.model_json_schema()
                ),
            }
        elif format_type == "openai":
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.args_schema.model_json_schema()
                }
            }
        elif format_type == "langchain":
            from langchain.tools import StructuredTool
            return StructuredTool.from_function(
                func=self.run,
                name=self.name,
                description=self.description,
                args_schema=self.args_schema
            )
        else:
            raise ValueError(f"Unknown format: {format_type}")