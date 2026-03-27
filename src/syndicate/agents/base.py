# agents/base_agent.py

import logging
from abc import ABC
import asyncio
from contextvars import ContextVar
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from collections.abc import AsyncGenerator

from ..communication_models import Message, ToolCall, StreamChunk

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..tools.agent_tool import AgentAsTool
    from .runtime import AgentRuntime
    from ..skills import SkillModule
    from ..clients import Client


class BaseAgent(ABC):
    """
    Plug-and-play agent architecture (motherboard design).
    
    Just plug in:
    - LLM client (auto-detects provider type)
    - Tools (auto-format for detected provider)
    - memory store (optional)
    
    No manual configuration needed - components auto-configure based on what's plugged in.
    
    ## Architecture
    
    This class implements the Template Method Pattern with a Hybrid API.
    
    ### Layered Design
    - Shell API (Public): invoke(), invoke_sync(), stream()
    - Core Engine (Override): _run_agent() ← CHILD CLASSES OVERRIDE THIS ONLY
    - Orchestration Engines: _orchestrate_invoke(), _orchestrate_stream()
    
    ### Hybrid API
    
    Declarative (Class Attributes):
        name: str = "MyAgent"
        system_prompt: str = "You are helpful"
        tools: list = []
        max_iterations: int = 5
    
    Functional (Runtime Override):
        agent = MyAgent(llm_client=client, system_prompt="You are a researcher")
    
    ### Sync/Async Support
    
    The framework handles both sync and async contexts automatically:
        # Async context (existing code)
        response = await agent.invoke("Hello!")
        
        # Sync context (new convenience)
        response = agent.invoke_sync("Hello!")
    
    ### Template Method Pattern
    
    DO NOT OVERRIDE: invoke(), invoke_sync(), stream()
    ONLY OVERRIDE: _run_agent(messages) - Returns final response string
    
    ### Example: Creating a Custom Agent
    
        class ResearchAgent(BaseAgent):
            name: str = "ResearchAgent"
            system_prompt: str = "You are a research assistant"
            
            async def _run_agent(self, messages):
                # Custom research logic
                return "Research results"
        
        # Usage - works in both sync and async contexts
        agent = ResearchAgent(llm_client=client)
        response = agent.invoke_sync("Tell me about Python")
    """

    # Declarative class attributes (Hybrid API - defaults)
    name: str = "UnnamedAgent"
    system_prompt: str = ""
    tools: list = None
    max_iterations: int = 5

    def __init__(
        self,
        llm_client: "Client",
        memory,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
        tools: Optional[List] = None,
        skills: Optional[List["SkillModule"]] = None,
        vision_client = None,
        audio_client = None,
        verbose: bool = False,
        **kwargs
    ):
        # Hybrid API: kwargs override class attributes
        self.verbose = verbose
        self.llm = llm_client
        self.memory = memory
        
        # Use kwargs with fallback to class attributes
        self.name = kwargs.get("name", name or self.__class__.name)
        self.system_prompt = kwargs.get("system_prompt", system_prompt or self.__class__.system_prompt)
        self.tools = list(kwargs.get("tools", tools or self.__class__.tools) or [])
        self.max_iterations = kwargs.get("max_iterations", self.__class__.max_iterations)
        
        self.skills: List["SkillModule"] = skills or []  # Skill modules
        
        # Auto-detect provider from client (motherboard auto-detection)
        self.provider = self._detect_provider(llm_client)
        
        # Build complete system prompt with skills
        self._base_system_prompt = self.system_prompt
        self.system_prompt = self._build_system_prompt_with_skills()
        
        # Collect tools from skills
        self._integrate_skill_tools()
        
        self.metadata = {}  # For storing agent-specific metadata
        
        # Cached tool wrapper
        self._as_tool_cache: Optional["AgentAsTool"] = None
        self._as_runtime_cache: Optional["AgentRuntime"] = None

        # Request-local snapshots (prevents cross-request state contamination)
        self._request_system_prompt_ctx: ContextVar[Optional[str]] = ContextVar(
            f"base_agent_system_prompt_{id(self)}",
            default=None,
        )
        self._request_formatted_tools_ctx: ContextVar[Optional[List[Any]]] = ContextVar(
            f"base_agent_formatted_tools_{id(self)}",
            default=None,
        )
        self._request_tool_map_ctx: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
            f"base_agent_tool_map_{id(self)}",
            default=None,
        )

    def _detect_provider(self, llm_client) -> str:
        """
        Auto-detect LLM provider from client (like detecting RAM or CPU type).
        Checks for provider_type attribute or class name patterns.
        """
        # Check for explicit provider_type tag (preferred)
        if hasattr(llm_client, 'provider_type'):
            return llm_client.provider_type
        
        # Fallback: detect from class name
        class_name = llm_client.__class__.__name__.lower()
        
        if 'gemini' in class_name:
            return 'gemini'
        elif 'openai' in class_name or 'gpt' in class_name:
            return 'openai'
        elif 'anthropic' in class_name or 'claude' in class_name:
            return 'anthropic'
        elif 'langchain' in class_name:
            return 'langchain'
        else:
            # Default fallback
            return 'gemini'
        
        
    def _build_system_prompt_with_skills(self) -> str:
        """
        Build complete system prompt by integrating skill modules.
        Skills are sorted by priority (higher first) and appended to base prompt.
        """
        if not self.skills:
            return self._base_system_prompt
        
        # Sort skills by priority (descending)
        sorted_skills = sorted(self.skills, key=lambda s: s.priority, reverse=True)
        
        # Build prompt sections
        sections = []
        
        # Base prompt first
        if self._base_system_prompt:
            sections.append(self._base_system_prompt)
        
        # Add skill sections
        if sorted_skills:
            sections.append("\n---\n# Installed Skill Modules\n")
            for skill in sorted_skills:
                sections.append(skill.to_prompt_section())
                sections.append("")  # Blank line between skills
        
        return "\n".join(sections)
    
    def _integrate_skill_tools(self) -> None:
        """
        Collect tools from all installed skill modules.
        """
        for skill in self.skills:
            for tool in skill.get_tools():
                if tool not in self.tools:
                    self.tools.append(tool)
    
    def install_skill(self, skill: "SkillModule") -> "BaseAgent":
        """
        Install a skill module at runtime.
        
        Args:
            skill: SkillModule to install
            
        Returns:
            self (for chaining)
        """
        if skill not in self.skills:
            self.skills.append(skill)
            # Rebuild system prompt
            self.system_prompt = self._build_system_prompt_with_skills()
            # Add skill's tools
            for tool in skill.get_tools():
                if tool not in self.tools:
                    self.tools.append(tool)
        return self
    
    def uninstall_skill(self, skill_name: str) -> bool:
        """
        Remove a skill module by name.
        
        Removes the skill's tools from the agent and rebuilds the system prompt.
        
        Args:
            skill_name: Name of skill to remove
            
        Returns:
            True if removed, False if not found
        """
        # Find the skill to remove
        skill_to_remove = None
        for s in self.skills:
            if s.name == skill_name:
                skill_to_remove = s
                break
        
        if skill_to_remove is None:
            return False
        
        # Remove the skill's tools from agent
        skill_tool_set = set(id(t) for t in skill_to_remove.get_tools())
        self.tools = [t for t in self.tools if id(t) not in skill_tool_set]
        
        # Remove the skill itself
        self.skills = [s for s in self.skills if s.name != skill_name]
        
        # Rebuild system prompt
        self.system_prompt = self._build_system_prompt_with_skills()
        return True
    
    def list_skills(self) -> List[str]:
        """Get names of all installed skills."""
        return [s.name for s in self.skills]    
    
    # ==================== SHELL API (State - Memory & Fallbacks) ====================
    # These methods manage Memory and Fallbacks. MUST NOT be overridden by child classes.
    
    async def _run_agent(self, messages: List[Message]) -> str:
        """
        The default core orchestration engine.
        
        This is the ONLY method child classes should override for custom logic.
        It delegates to the appropriate orchestration engine.
        
        Args:
            messages: Current message history
            
        Returns:
            Final response string after all tool calls are complete
        """
        # Format tools for provider
        
        # Run the blocking orchestration engine
        return await self._orchestrate_invoke(messages)
    
    async def invoke(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default"
    ) -> str:
        """
        Main interface for interacting with the agent (async).
        
        Builds history, runs the agent core, stores interaction, and returns response.
        
        Args:
            user_input: The user's message
            owner_id: User identifier for memory (default: "default")
            chat_id: Conversation identifier for memory (default: "default")
            
        Returns:
            The agent's response as a string
            
        Example:
            response = await agent.invoke("What's the weather?")
        """
        # Build messages with history (this adds the user message at the end)
        messages = await self._build_messages(user_input, owner_id, chat_id)
        
        # Capture the user message (last message added by _build_messages)
        user_message = messages[-1]
        
        # Capture initial length for slicing after orchestration (excluding user message)
        initial_length = len(messages)

        # Snapshot mutable runtime config for this request (prompt + tools)
        request_config = self._snapshot_request_config()
        prompt_token = self._request_system_prompt_ctx.set(request_config["system_prompt"])
        formatted_tools_token = self._request_formatted_tools_ctx.set(request_config["formatted_tools"])
        tool_map_token = self._request_tool_map_ctx.set(request_config["tool_map"])

        try:
            # Run the agent core (this is where child classes can override _run_agent)
            final_text = await self._run_agent(messages)

            # Slice the mutated list to get only new messages (AI responses, tool calls, etc.)
            new_messages = messages[initial_length:]

            # Store the complete interaction (user message + agent responses)
            messages_to_store = [user_message] + new_messages
            await self._store_interaction(messages_to_store, owner_id, chat_id)
        finally:
            self._request_tool_map_ctx.reset(tool_map_token)
            self._request_formatted_tools_ctx.reset(formatted_tools_token)
            self._request_system_prompt_ctx.reset(prompt_token)
        
        return final_text
    
    def invoke_sync(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default"
    ) -> str:
        """
        Synchronous wrapper for invoke() for quick prototyping and testing.
        
        Must be called from a synchronous context (no running event loop).
        For async contexts, use `await agent.invoke(...)` directly.
        
        Args:
            user_input: The user's message
            owner_id: User identifier for memory (default: "default")
            chat_id: Conversation identifier for memory (default: "default")
            
        Returns:
            The agent's response as a string
            
        Raises:
            RuntimeError: If called from within a running async event loop.
            
        Example:
            response = agent.invoke_sync("What's the weather?")
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running — safe to use asyncio.run()
            return asyncio.run(self.invoke(user_input, owner_id, chat_id))
        
        raise RuntimeError(
            "invoke_sync() cannot be called from an async context. "
            "Use `await agent.invoke(...)` instead."
        )
    
    async def stream(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
        include_thinking: bool = False
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream the agent's response to the UI.
        
        Builds history, runs the streaming orchestration engine, stores interaction,
        and yields content chunks progressively.
        
        Args:
            user_input: The user's message
            owner_id: User identifier for memory (default: "default")
            chat_id: Conversation identifier for memory (default: "default")
            include_thinking: Whether to include thinking tokens in output
            
        Yields:
            StreamChunk objects with content chunks progressively
            
        Example:
            async for chunk in agent.stream("What's the weather?"):
                print(chunk.content, end="", flush=True)
        """
        # Build messages with history (this adds the user message at the end)
        messages = await self._build_messages(user_input, owner_id, chat_id)
        
        # Capture the user message (last message added by _build_messages)
        user_message = messages[-1]
        
        # Capture initial length for slicing after orchestration (excluding user message)
        initial_length = len(messages)

        # Snapshot mutable runtime config for this request (prompt + tools)
        request_config = self._snapshot_request_config()
        prompt_token = self._request_system_prompt_ctx.set(request_config["system_prompt"])
        formatted_tools_token = self._request_formatted_tools_ctx.set(request_config["formatted_tools"])
        tool_map_token = self._request_tool_map_ctx.set(request_config["tool_map"])

        # Run the streaming orchestration engine
        used_fallback_invoke = False
        try:
            async for chunk in self._orchestrate_stream(messages, include_thinking=include_thinking):
                # Yield content chunks to UI
                if include_thinking and chunk.thinking:
                    yield chunk
                if chunk.content:
                    yield chunk
        except NotImplementedError:
            # Fallback for agents that don't implement streaming
            used_fallback_invoke = True
            final_response = await self.invoke(user_input, owner_id, chat_id)
            yield StreamChunk(content=final_response, is_finished=True)
        finally:
            self._request_tool_map_ctx.reset(tool_map_token)
            self._request_formatted_tools_ctx.reset(formatted_tools_token)
            self._request_system_prompt_ctx.reset(prompt_token)

        if used_fallback_invoke:
            return

        # Slice the mutated list to get only new messages (AI responses, tool calls, etc.)
        new_messages = messages[initial_length:]

        # Store the complete interaction (user message + agent responses)
        messages_to_store = [user_message] + new_messages
        await self._store_interaction(messages_to_store, owner_id, chat_id)

    async def _build_messages(self, user_input: str, owner_id: str, chat_id: str) -> List[Message]:
        """
        Build message list from history and current input.
        
        Note: The system prompt is NOT included here. It is passed
        separately to the LLM client via the ``system_message`` parameter
        in ``chat_completion_async`` / ``chat_completion_stream``. This
        avoids duplication and lets each provider handle system prompts
        in its native way (Gemini: ``system_instruction`` config,
        OpenAI: prepended system message).
        
        Override this if you need custom message formatting.
        """
        messages = []
        
        # Add conversation history
        messages.extend(await self.get_history(owner_id, chat_id))
        
        # Add current user input
        messages.append(Message(content=user_input, role="human"))
        
        return messages
    
            

    async def get_history(self, owner_id: str = None, chat_id: str = None) -> List[Message]:
        """Get conversation history from memory store."""
        if hasattr(self.memory, 'get_history'):
            return await self.memory.get_history(owner_id=owner_id, chat_id=chat_id)
        return []

    async def clear_history(self, owner_id: str = None, chat_id: str = None) -> None:
        """Clear conversation history."""
        if hasattr(self.memory, 'clear'):
            await self.memory.clear(owner_id=owner_id, chat_id=chat_id)

    def add_tool(self, tool) -> None:
        """Add a tool/function to the agent."""
        if tool not in self.tools:
            self.tools.append(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool by name. Returns True if removed, False if not found."""
        original_length = len(self.tools)
        self.tools = [t for t in self.tools if getattr(t, 'name', None) != tool_name]
        return len(self.tools) < original_length
    
    def _materialize_tools(self, tool_defs: Optional[List[Any]] = None) -> List[Any]:
        """Instantiate class-based tools and return concrete tool instances."""
        materialized: List[Any] = []
        source_tools = tool_defs if tool_defs is not None else self.tools

        for tool in source_tools:
            if isinstance(tool, type):
                materialized.append(tool())
            else:
                materialized.append(tool)

        return materialized

    def get_formatted_tools(self, tool_defs: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Get tools formatted for the current LLM provider.
        Uses the tool's to_format() method if available.
        """
        formatted_tools = []
        for tool_instance in self._materialize_tools(tool_defs):
            
            # Check if it's a BaseTool with to_format method
            if hasattr(tool_instance, 'to_format'):
                # Pass format_type as first positional argument
                formatted_tools.append(tool_instance.to_format(self.provider))
            else:
                # Fallback for non-BaseTool objects
                formatted_tools.append(tool_instance)
        #print(formatted_tools)
        return formatted_tools

    def _snapshot_request_config(self) -> Dict[str, Any]:
        """
        Build an immutable per-request snapshot of mutable agent runtime config.

        Snapshot includes:
        - system prompt text
        - formatted tools passed to the provider
        - concrete tool instances used for execution dispatch
        """
        tool_instances = self._materialize_tools(self.tools)
        tool_map = {
            tool.name: tool
            for tool in tool_instances
            if hasattr(tool, 'name') and getattr(tool, 'name')
        }

        return {
            "system_prompt": self.system_prompt,
            "formatted_tools": self.get_formatted_tools(tool_instances),
            "tool_map": tool_map,
        }

    def _get_request_system_prompt(self) -> str:
        """Get request-local system prompt snapshot (fallback to current)."""
        return self._request_system_prompt_ctx.get() or self.system_prompt

    def _get_request_formatted_tools(self) -> List[Any]:
        """Get request-local formatted tools snapshot (fallback to current)."""
        formatted = self._request_formatted_tools_ctx.get()
        if formatted is not None:
            return formatted
        return self.get_formatted_tools()

    def _get_request_tool_map(self) -> Optional[Dict[str, Any]]:
        """Get request-local tool-instance map, if any."""
        return self._request_tool_map_ctx.get()

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments.
        Returns dict with success status and result/error.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool arguments
            
        Returns:
            Dict with success status and result/error
        """
        # Prefer request-local snapshot to avoid cross-request mutations.
        tool_instance = None
        request_tool_map = self._get_request_tool_map()
        if request_tool_map is not None:
            tool_instance = request_tool_map.get(tool_name)

        # Fallback for direct/manual calls outside invoke/stream
        if tool_instance is None:
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    # Instantiate if it's a class
                    tool_instance = tool() if isinstance(tool, type) else tool
                    break
        
        if not tool_instance:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        # Execute using async wrapper (handles both sync and async tools)
        try:
            result = await tool_instance.run_async(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
        
    async def _execute_tools_concurrently(self, tool_calls: List[ToolCall]) -> List[Message]:
        """
        Shared logic for both stream() and invoke().
        Takes a list of ToolCalls, runs them concurrently, and returns the exact
        List[Message] objects with role="tool" and matched tool_call_ids.
        """
        tool_tasks = [self.execute_tool(tc.name, **tc.arguments) for tc in tool_calls]
        results = await asyncio.gather(*tool_tasks, return_exceptions=True)

        tool_messages = []
        for tool_call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                content_str = f"Tool error: {str(result)}"
            elif isinstance(result, dict):
                if result.get("success"):
                    content_str = f"Tool '{tool_call.name}' result: {result['result']}"
                else:
                    content_str = f"Tool '{tool_call.name}' error: {result['error']}"
            else:
                content_str = f"Tool result: {result}"

            # Append a separate message for each tool with its specific ID
            tool_messages.append(
                Message(
                    role="tool",
                    content=content_str,
                    tool_call_id=tool_call.id
                ))

        return tool_messages

    async def _orchestrate_stream(
        self,
        messages: List[Message],
        formatted_tools: Optional[List] = None,
        include_thinking = False
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Streaming orchestration engine.
        
        Implements the multi-turn tool calling loop with streaming support.
        Yields content chunks to the UI while accumulating tool calls.
        
        Args:
            messages: Current message history
            formatted_tools: Formatted tools for the LLM provider
            include_thinking
            
        Yields:
            StreamChunk objects with content chunks progressively
        """
        max_iterations = getattr(self, 'max_iterations', 5)
        iteration = 0
        
        while iteration < max_iterations:
            accumulated_content = ""
            accumulated_thinking = ""
            accumulated_tool_calls = []
            
            # Call LLM with streaming
            active_tools = formatted_tools if formatted_tools is not None else self._get_request_formatted_tools()
            stream = self.llm.chat_completion_stream(
                messages=messages,
                system_message=Message(content=self._get_request_system_prompt(), role='system'),
                tools=active_tools
            )
            
            # Stream the response
            async for chunk in stream:

                # Yield thinking contents to the UI
                if chunk.thinking and include_thinking:
                    yield StreamChunk(
                        thinking=chunk.thinking,
                        is_finished=chunk.is_finished,
                        finish_reason=chunk.finish_reason,
                    )
                    accumulated_thinking += chunk.thinking or ""

                # Yield content chunks to the UI
                if chunk.content:
                    yield StreamChunk(
                        content=chunk.content,
                        is_finished=chunk.is_finished,
                        finish_reason=chunk.finish_reason,
                    )
                    accumulated_content += chunk.content
                elif chunk.is_finished and not chunk.thinking:
                    # Ensure callers waiting for terminal chunk get completion
                    # signal even when model emitted no textual content
                    # (e.g., tool-call-only turns).
                    yield StreamChunk(
                        is_finished=True,
                        finish_reason=chunk.finish_reason,
                    )
                
                # Accumulate tool calls JSON (silently)
                if chunk.tool_calls:
                    accumulated_tool_calls.extend(chunk.tool_calls)
            
            # After streaming completes, append the accumulated AI message to history
            messages.append(Message(
                role="ai",
                content=accumulated_content,
                tool_calls=accumulated_tool_calls
            ))
            
            # If there are tool calls, execute them and continue the loop
            if accumulated_tool_calls:
                tool_messages = await self._execute_tools_concurrently(accumulated_tool_calls)
                messages.extend(tool_messages)
                iteration += 1
                continue
            
            # No tool calls - we're done
            break

    async def _orchestrate_invoke(
        self,
        messages: List[Message]
    ) -> str:
        """
        Blocking orchestration engine.
        
        Implements the multi-turn tool calling loop with standard (non-streaming) calls.
        
        Args:
            messages: Current message history
            formatted_tools: Formatted tools for the LLM provider
            
        Returns:
            Final response string after all tool calls are complete
        """

        # Get request-local config snapshot if available
        formatted_tools = self._get_request_formatted_tools()
        system_prompt = self._get_request_system_prompt()
        
        max_iterations = getattr(self, 'max_iterations', 5)
        iteration = 0

        
        while iteration < max_iterations:
            # Call the standard (non-streaming) LLM
            response = await self.llm.chat_completion_async(
                messages=messages,
                system_message=Message(content=system_prompt, role='system'),
                tools=formatted_tools
            )
            
            # Append the response message to history
            messages.append(response.to_message())
            
            # If there are tool calls, execute them and continue the loop
            if response.tool_calls:
                tool_messages = await self._execute_tools_concurrently(response.tool_calls)
                messages.extend(tool_messages)
                iteration += 1
                continue
            
            # No tool calls - return the final string content
            return response.content if hasattr(response, 'content') else str(response)
        
        # Max iterations reached - return partial results
        return f"Error: Maximum tool call iterations ({max_iterations}) reached."
    
    # ==================== ARCHITECT'S SAFEGUARD ====================
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Architect's safeguard: Prevents developers from breaking the framework.
        
        If a child class overrides 'invoke' or 'stream', it's a violation of
        the Template Method pattern. Child classes should override _run_agent() instead.
        
        Raises:
            TypeError: If child class overrides invoke or stream
        """
        # Check if child class has overridden invoke or stream
        if 'invoke' in cls.__dict__ or 'stream' in cls.__dict__:
            overridden = [m for m in ('invoke', 'stream') if m in cls.__dict__]
            raise TypeError(
                f"Cannot override {', '.join(repr(m) for m in overridden)} in {cls.__name__}. "
                f"Child classes must override _run_agent() instead to preserve memory and framework safety. "
                f"See BaseAgent documentation for the Template Method pattern."
            )
        super().__init_subclass__(**kwargs)
        

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt."""
        self._base_system_prompt = prompt
        self.system_prompt = self._build_system_prompt_with_skills()

    def get_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return {
            "name": self.name,
            "provider": self.provider,  # Auto-detected provider
            "client_type": self.llm.__class__.__name__,
            "system_prompt": self.system_prompt,
            "tools": [getattr(t, 'name', str(t)) for t in self.tools],
            "skills": self.list_skills(),  # Installed skill modules
            "has_memory": self.memory is not None,
            "history_length": None,  # async — use await agent.get_history() for actual history
            "metadata": self.metadata
        }

    def _handle_error(self, error: Exception) -> str:
        """
        Handle errors that occur during agent execution.
        Override this to customize error handling.
        """
        return f"Error: {str(error)}"

    async def safe_handle(self, user_input: str) -> str:
        """
        Handle user input with automatic error catching.
        Returns error message if something goes wrong.
        
        Args:
            user_input: The user's message
            
        Returns:
            The agent's response or error message
        """
        try:
            return await self.invoke(user_input)
        except Exception as e:
            return self._handle_error(e)
    
    async def _store_interaction(
        self,
        messages: List[Message],
        owner_id: str,
        chat_id: str
    ) -> None:
        """Store conversation in memory if available."""
        if not self.memory:
            return
        
        try:
            # Store each message that was part of this interaction
            # This includes tool calls and results
            for msg in messages:
                await self.memory.add_message(msg, owner_id=owner_id, chat_id=chat_id)
            
        except Exception as e:
            logger.error(
                "[%s] Failed to persist conversation to memory: %s",
                self.name, e, exc_info=True,
            )

    def as_tool(self, custom_description: Optional[str] = None) -> "AgentAsTool":
        """
        Wrap this agent as a tool for use by other agents.
        
        Enables agent-to-agent delegation:
            research = ResearchAgent(client)
            kiriko = create_kiriko(
                llm_client=client,
                additional_tools=[research.as_tool()]
            )
        
        Args:
            custom_description: Optional override for tool description
            
        Returns:
            AgentAsTool wrapper for this agent
        """
        from ..tools.agent_tool import AgentAsTool
        
        # Return cached if no custom description and already created
        if custom_description is None and self._as_tool_cache is not None:
            return self._as_tool_cache
        
        tool = AgentAsTool(self, custom_description=custom_description)
        
        # Cache only if no custom description
        if custom_description is None:
            self._as_tool_cache = tool
        
        return tool

    def as_runtime(self) -> "AgentRuntime":
        """Return a runtime-only facade exposing invoke/stream/invoke_sync."""
        from .runtime import AgentRuntime

        if self._as_runtime_cache is None:
            self._as_runtime_cache = AgentRuntime(self)
        return self._as_runtime_cache
    
    def register(self) -> "BaseAgent":
        """
        Register this agent in the global registry for discovery.
        
        Returns:
            self (for chaining)
            
        Example:
            agent = ResearchAgent(client).register()
        """
        from ..registry import AgentRegistry
        AgentRegistry.register(self)
        return self

    # Sensorial implementation templates (vision, audio) can be added here

    def can_hear(self) -> bool:
        """Check if the agent has audio capabilities."""
        return hasattr(self, 'audio_client') and self.audio_client is not None


    def can_see(self) -> bool:
        """Check if the agent has vision capabilities."""
        return hasattr(self, 'vision_client') and self.vision_client is not None