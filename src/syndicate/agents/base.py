# agents/base_agent.py

import logging
from abc import ABC
import asyncio
import inspect
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from collections.abc import AsyncGenerator
from uuid import uuid4

from ..communication_models import (
    Message, 
    ToolCall, 
    StreamChunk, 
    ToolCallEvent, 
    ToolResultEnvelope,
    ChatResponse,
    A2AAgentCard
)
from ..protocols import Observer
from ..tools.base_tool import ToolExecutionPolicy

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
    max_total_tool_calls: Optional[int] = None
    max_concurrent_tool_calls: Optional[int] = None

    def __init__(
        self,
        llm_client: "Client",
        memory,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
        tools: Optional[List] = None,
        max_total_tool_calls: Optional[int] = None,
        max_concurrent_tool_calls: Optional[int] = None,
        skills: Optional[List["SkillModule"]] = None,
        observers: Optional[List[Observer]] = None,
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
        resolved_max_total_tool_calls = kwargs.get(
            "max_total_tool_calls",
            max_total_tool_calls
            if max_total_tool_calls is not None
            else getattr(self.__class__, "max_total_tool_calls", None),
        )
        resolved_max_concurrent_tool_calls = kwargs.get(
            "max_concurrent_tool_calls",
            max_concurrent_tool_calls
            if max_concurrent_tool_calls is not None
            else getattr(self.__class__, "max_concurrent_tool_calls", None),
        )
        self.max_total_tool_calls = self._normalize_guardrail_limit(
            resolved_max_total_tool_calls,
            "max_total_tool_calls",
            allow_zero=True,
        )
        self.max_concurrent_tool_calls = self._normalize_guardrail_limit(
            resolved_max_concurrent_tool_calls,
            "max_concurrent_tool_calls",
            allow_zero=False,
        )
        
        self.skills: List["SkillModule"] = skills or []  # Skill modules
        self.observers: List[Observer] = list(observers or [])
        
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
        self._request_observer_ctx: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
            f"base_agent_observer_{id(self)}",
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

    def get_manifest(self) -> "A2AAgentCard":
        """
        Generate an Agent2Agent (A2A) v1.0.0 compliant Agent Card.
        This provides a structured summary of the agent's capabilities (skills)
        and interaction modes, suitable for multi-agent swarm discovery.
        """
        from ..communication_models import A2AAgentCard, A2ASkill, A2ACapabilities
        
        # Build skills list dynamically from installed tools
        skills = []
        for tool in self.tools:
            tool_name = getattr(tool, 'name', None) or getattr(tool, '__name__', str(tool))
            tool_desc = getattr(tool, 'description', None) or getattr(tool, '__doc__', 'An AI capability.')
            
            skills.append(
                A2ASkill(
                    id=tool_name,
                    name=tool_name.replace("_", " ").title(),
                    description=tool_desc.strip(),
                    inputModes=["application/json"],
                    outputModes=["application/json"]
                )
            )

        # Truncate system prompt to form the agent description
        # Fallback to "General AI Agent" if no prompt exists.
        agent_desc = str(self.system_prompt).strip() if self.system_prompt else "General AI Agent"
        if len(agent_desc) > 200:
            agent_desc = agent_desc[:200] + "..."

        return A2AAgentCard(
            name=self.name,
            description=agent_desc,
            version="1.0.0",
            capabilities=A2ACapabilities(
                streaming=True,  # Native to Syndicate via .stream()
                pushNotifications=False,  # Can be added when we build async task observers
                extendedAgentCard=False
            ),
            skills=skills,
            # supportedInterfaces will be populated by the A2A Server Adapter later
        )
    
    # ==================== SHELL API (State - Memory & Fallbacks) ====================
    # These methods manage Memory and Fallbacks. MUST NOT be overridden by child classes.
    
    async def _run_agent(self, messages: List[Message], **kwargs) -> str:
        """
        The default core orchestration engine.
        
        This is the ONLY method child classes should override for custom logic.
        It delegates to the appropriate orchestration engine.
        
        Args:
            messages: Current message history
            **kwargs: Extra LLM parameters forwarded to the client
            
        Returns:
            Final response string after all tool calls are complete
        """
        # Format tools for provider
        
        # Run the blocking orchestration engine
        return await self._orchestrate_invoke(messages, **kwargs)
    
    async def invoke(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Main interface for interacting with the agent (async).
        
        Builds history, runs the agent core, stores interaction, and returns response.
        
        Args:
            user_input: The user's message
            owner_id: User identifier for memory (default: "default")
            chat_id: Conversation identifier for memory (default: "default")
            **kwargs: Extra LLM parameters forwarded to the client (e.g. stop, top_p)
            
        Returns:
            The agent's response as a string
            
        Example:
            response = await agent.invoke("What's the weather?")
        """
        emit_observers = not bool(kwargs.pop("_skip_observer_hooks", False))
        observer_context = self._build_request_observer_context(owner_id, chat_id, flow="invoke")
        observer_ctx = self._ensure_observer_context_var()
        observer_ctx_token = observer_ctx.set(observer_context)
        request_started_at = asyncio.get_running_loop().time()
        request_succeeded = False
        prompt_token = None
        formatted_tools_token = None
        tool_map_token = None

        if emit_observers:
            await self._emit_observer_hook(
                "on_request_start",
                phase="request",
                input_chars=len(user_input),
            )

        try:
            # Build messages with history (this adds the user message at the end)
            messages = await self._build_messages(user_input, owner_id, chat_id, metadata)

            # Capture the user message (last message added by _build_messages)
            user_message = messages[-1]

            # Capture initial length for slicing after orchestration (excluding user message)
            initial_length = len(messages)

            # Snapshot mutable runtime config for this request (prompt + tools)
            request_config = self._snapshot_request_config()
            prompt_token = self._request_system_prompt_ctx.set(request_config["system_prompt"])
            formatted_tools_token = self._request_formatted_tools_ctx.set(request_config["formatted_tools"])
            tool_map_token = self._request_tool_map_ctx.set(request_config["tool_map"])

            # Run the agent core (this is where child classes can override _run_agent)
            final_text = await self._run_agent(messages, **kwargs)

            # Slice the mutated list to get only new messages (AI responses, tool calls, etc.)
            new_messages = messages[initial_length:]

            # Store the complete interaction (user message + agent responses)
            messages_to_store = [user_message] + new_messages
            await self._store_interaction(messages_to_store, owner_id, chat_id)

            request_succeeded = True
            return final_text
        except asyncio.CancelledError as exc:
            if emit_observers:
                await self._emit_observer_error(
                    phase="request",
                    error=exc,
                    cancelled=True,
                )
            raise
        except Exception as exc:
            if emit_observers:
                await self._emit_observer_error(
                    phase="request",
                    error=exc,
                )
            raise
        finally:
            if tool_map_token is not None:
                self._request_tool_map_ctx.reset(tool_map_token)
            if formatted_tools_token is not None:
                self._request_formatted_tools_ctx.reset(formatted_tools_token)
            if prompt_token is not None:
                self._request_system_prompt_ctx.reset(prompt_token)

            if emit_observers:
                await self._emit_observer_hook(
                    "on_request_end",
                    phase="request",
                    success=request_succeeded,
                    latency_ms=self._elapsed_ms(request_started_at),
                )
            observer_ctx.reset(observer_ctx_token)
    
    def invoke_sync(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous wrapper for invoke() for quick prototyping and testing.
        
        Must be called from a synchronous context (no running event loop).
        Also works from interactive environments with a running loop (e.g. Jupyter
        notebooks) by delegating to a background thread.
        
        Args:
            user_input: The user's message
            owner_id: User identifier for memory (default: "default")
            chat_id: Conversation identifier for memory (default: "default")
            **kwargs: Extra LLM parameters forwarded to the client (e.g. stop, top_p)
            
        Returns:
            The agent's response as a string
            
        Example:
            response = agent.invoke_sync("What's the weather?")
        """
        import concurrent.futures

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running — safe to use asyncio.run() directly.
            return asyncio.run(self.invoke(user_input, owner_id, chat_id, metadata, **kwargs))

        # A loop is already running (e.g. Jupyter / IPython kernel).
        # Spin up a worker thread with its own event loop to avoid nesting.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, self.invoke(user_input, owner_id, chat_id, metadata, **kwargs))
            return future.result()
    
    async def stream(
        self,
        user_input: str,
        owner_id: str = "default",
        chat_id: str = "default",
        include_thinking: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
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
            **kwargs: Extra LLM parameters forwarded to the client (e.g. stop, top_p)
            
        Yields:
            StreamChunk objects with content chunks progressively
            
        Example:
            async for chunk in agent.stream("What's the weather?"):
                print(chunk.content, end="", flush=True)
        """
        emit_observers = not bool(kwargs.pop("_skip_observer_hooks", False))
        observer_context = self._build_request_observer_context(owner_id, chat_id, flow="stream")
        observer_ctx = self._ensure_observer_context_var()
        observer_ctx_token = observer_ctx.set(observer_context)
        request_started_at = asyncio.get_running_loop().time()
        request_succeeded = False
        prompt_token = None
        formatted_tools_token = None
        tool_map_token = None

        if emit_observers:
            await self._emit_observer_hook(
                "on_request_start",
                phase="request",
                input_chars=len(user_input),
            )

        try:
            # Build messages with history (this adds the user message at the end)
            messages = await self._build_messages(user_input, owner_id, chat_id, metadata)

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
            try:
                async for chunk in self._orchestrate_stream(messages, include_thinking=include_thinking, **kwargs):
                    # Yield content chunks to UI
                    if include_thinking and chunk.thinking:
                        yield chunk
                    if chunk.content:
                        yield chunk
                    if chunk.tool_call is not None:
                        yield chunk
            except NotImplementedError:
                # Fallback for agents that don't implement streaming
                final_response = await self.invoke(
                    user_input,
                    owner_id,
                    chat_id,
                    _skip_observer_hooks=True,
                    **kwargs,
                )
                yield StreamChunk(content=final_response, is_finished=True)
                request_succeeded = True
                return

            # Slice the mutated list to get only new messages (AI responses, tool calls, etc.)
            new_messages = messages[initial_length:]

            # Store the complete interaction (user message + agent responses)
            messages_to_store = [user_message] + new_messages
            await self._store_interaction(messages_to_store, owner_id, chat_id)
            request_succeeded = True

        except asyncio.CancelledError as exc:
            if emit_observers:
                await self._emit_observer_error(
                    phase="request",
                    error=exc,
                    cancelled=True,
                )
            raise
        except Exception as exc:
            if emit_observers:
                await self._emit_observer_error(
                    phase="request",
                    error=exc,
                )
            raise
        finally:
            if tool_map_token is not None:
                self._request_tool_map_ctx.reset(tool_map_token)
            if formatted_tools_token is not None:
                self._request_formatted_tools_ctx.reset(formatted_tools_token)
            if prompt_token is not None:
                self._request_system_prompt_ctx.reset(prompt_token)

            if emit_observers:
                await self._emit_observer_hook(
                    "on_request_end",
                    phase="request",
                    success=request_succeeded,
                    latency_ms=self._elapsed_ms(request_started_at),
                )
            observer_ctx.reset(observer_ctx_token)

    async def _build_messages(self, user_input: str, owner_id: str, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Message]:
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
        messages.append(Message(content=user_input, role="human", metadata=metadata or {}))
        
        return messages
    
            

    async def get_history(self, owner_id: str = None, chat_id: str = None) -> List[Message]:
        """Get conversation history from memory store."""
        if hasattr(self.memory, 'get_history'):
            return await self.memory.get_history(owner_id=owner_id, chat_id=chat_id)
        return []

    async def get_full_history(
        self,
        owner_id: str = None,
        chat_id: str = None,
        limit: Optional[int] = None,
        include_closed_buckets: bool = True,
        include_deleted: bool = False,
        include_context_summary: bool = False,
    ) -> List[Message]:
        """Get flattened conversation history for display or auditing."""
        if hasattr(self.memory, 'get_full_history'):
            return await self.memory.get_full_history(
                owner_id=owner_id,
                chat_id=chat_id,
                limit=limit,
                include_closed_buckets=include_closed_buckets,
                include_deleted=include_deleted,
                include_context_summary=include_context_summary,
            )
        if hasattr(self.memory, 'get_history'):
            return await self.memory.get_history(
                owner_id=owner_id,
                chat_id=chat_id,
                limit=limit,
                include_context_summary=include_context_summary,
            )
        return []

    async def clear_history(self, owner_id: str = None, chat_id: str = None) -> None:
        """Clear conversation history."""
        if hasattr(self.memory, 'clear'):
            await self.memory.clear(owner_id=owner_id, chat_id=chat_id)

    async def regenerate_response(
        self,
        owner_id: str = "default",
        chat_id: str = "default",
        target_index: Optional[int] = None,
        mode: str = "message",
        **kwargs,
    ) -> ChatResponse:
        """Regenerate an AI response by truncating active history and replaying.

        Args:
            mode: Truncation scope.
                - ``"message"`` (default): truncates only the final AI answer and
                  replays from the existing tool results.  Tool calls are NOT
                  re-executed.
                - ``"turn"``: truncates the entire AI turn — the initiating AI
                  tool-call message, all intermediate tool/result messages, and
                  the final answer — then replays the full agentic loop including
                  fresh tool executions.

        Behavior:
        - Operates on active bucket history only.
        - Uses hard delete during truncation for deterministic replay.
        - Returns ChatResponse for API consistency with client models.
        """
        if mode not in ("message", "turn"):
            raise ValueError(f"Invalid mode '{mode}': must be 'message' or 'turn'")
        if self.memory is None:
            raise ValueError("regenerate_response requires a configured memory backend")
        if not hasattr(self.memory, "get_history") or not hasattr(self.memory, "delete_message"):
            raise ValueError("memory backend must implement get_history and delete_message")

        emit_observers = not bool(kwargs.pop("_skip_observer_hooks", False))
        observer_context = self._build_request_observer_context(owner_id, chat_id, flow="regenerate")
        observer_ctx = self._ensure_observer_context_var()
        observer_ctx_token = observer_ctx.set(observer_context)
        request_started_at = asyncio.get_running_loop().time()
        request_succeeded = False
        prompt_token = None
        formatted_tools_token = None
        tool_map_token = None

        if emit_observers:
            await self._emit_observer_hook(
                "on_request_start",
                phase="request",
                action="regenerate",
            )

        try:
            history = await self.memory.get_history(
                owner_id=owner_id,
                chat_id=chat_id,
                include_context_summary=False,
            )
            if not history:
                raise ValueError("No conversation history found to regenerate.")

            if target_index is not None:
                if target_index < 0 or target_index >= len(history):
                    raise ValueError(f"Invalid target_index: {target_index}")
                ai_index = next(
                    (i for i in range(target_index, len(history)) if history[i].role == "ai"),
                    None,
                )
            else:
                ai_index = next(
                    (i for i in range(len(history) - 1, -1, -1) if history[i].role == "ai"),
                    None,
                )

            if ai_index is None:
                raise ValueError("No AI message found to regenerate.")

            # Determine where to start truncation based on mode.
            if mode == "turn":
                # Walk backwards from ai_index to find the first ai message of
                # this turn (the one that initiated the tool-call chain, if any).
                # Consecutive "ai" and "tool" messages belong to the same turn;
                # stop at the first "human" or other boundary.
                truncate_from = ai_index
                for i in range(ai_index - 1, -1, -1):
                    if history[i].role in ("ai", "tool"):
                        if history[i].role == "ai":
                            truncate_from = i
                    else:
                        break
            else:
                # "message" mode: only delete the final AI answer.
                truncate_from = ai_index

            # Truncate from selected point to the end of active visible history.
            delete_count = len(history) - truncate_from
            for _ in range(delete_count):
                deleted = await self.memory.delete_message(
                    owner_id=owner_id,
                    chat_id=chat_id,
                    index=truncate_from,
                    hard_delete=True,
                )
                if not deleted:
                    raise RuntimeError("Failed to truncate history for regeneration")

            messages = await self.get_history(owner_id, chat_id)
            initial_length = len(messages)

            # Snapshot mutable runtime config for this request (prompt + tools)
            request_config = self._snapshot_request_config()
            prompt_token = self._request_system_prompt_ctx.set(request_config["system_prompt"])
            formatted_tools_token = self._request_formatted_tools_ctx.set(request_config["formatted_tools"])
            tool_map_token = self._request_tool_map_ctx.set(request_config["tool_map"])

            regenerated_text = await self._run_agent(messages, **kwargs)
            new_messages = messages[initial_length:]
            if not new_messages:
                # Keep persistence resilient for custom _run_agent overrides
                # that return text without mutating message history.
                new_messages = [Message(role="ai", content=str(regenerated_text))]
            await self._store_interaction(new_messages, owner_id, chat_id)

            all_tool_calls = [
                tc
                for msg in new_messages
                if msg.role == "ai" and msg.tool_calls
                for tc in msg.tool_calls
            ]

            request_succeeded = True
            return ChatResponse(
                content=str(regenerated_text),
                tool_calls=all_tool_calls if all_tool_calls else None,
            )
        except asyncio.CancelledError as exc:
            if emit_observers:
                await self._emit_observer_error(
                    phase="request",
                    error=exc,
                    cancelled=True,
                )
            raise
        except Exception as exc:
            if emit_observers:
                await self._emit_observer_error(
                    phase="request",
                    error=exc,
                    action="regenerate",
                )
            raise
        finally:
            if tool_map_token is not None:
                self._request_tool_map_ctx.reset(tool_map_token)
            if formatted_tools_token is not None:
                self._request_formatted_tools_ctx.reset(formatted_tools_token)
            if prompt_token is not None:
                self._request_system_prompt_ctx.reset(prompt_token)

            if emit_observers:
                await self._emit_observer_hook(
                    "on_request_end",
                    phase="request",
                    action="regenerate",
                    success=request_succeeded,
                    latency_ms=self._elapsed_ms(request_started_at),
                )
            observer_ctx.reset(observer_ctx_token)

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

    def _build_request_observer_context(self, owner_id: str, chat_id: str, flow: str) -> Dict[str, Any]:
        """Build per-request observer metadata used to enrich hook payloads."""
        return {
            "request_id": str(uuid4()),
            "owner_id": owner_id,
            "chat_id": chat_id,
            "flow": flow,
            "model": self._get_model_name(),
        }

    def _ensure_observer_context_var(self) -> ContextVar[Optional[Dict[str, Any]]]:
        observer_ctx = getattr(self, "_request_observer_ctx", None)
        if observer_ctx is None:
            observer_ctx = ContextVar(
                f"base_agent_observer_{id(self)}_lazy",
                default=None,
            )
            self._request_observer_ctx = observer_ctx
        return observer_ctx

    def _get_model_name(self) -> Optional[str]:
        model_name = getattr(self.llm, "model_name", None)
        if model_name:
            return model_name
        model_name = getattr(self.llm, "model", None)
        if model_name:
            return str(model_name)
        if hasattr(self, "llm") and self.llm is not None:
            return self.llm.__class__.__name__
        return None

    def _get_request_observer_context(self) -> Dict[str, Any]:
        observer_ctx = getattr(self, "_request_observer_ctx", None)
        if observer_ctx is None:
            return {}
        return observer_ctx.get() or {}

    def _build_observer_event(self, phase: str, **event_fields: Any) -> Dict[str, Any]:
        context = self._get_request_observer_context()
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": context.get("request_id"),
            "owner_id": context.get("owner_id"),
            "chat_id": context.get("chat_id"),
            "agent_name": getattr(self, "name", self.__class__.__name__),
            "model": context.get("model") or self._get_model_name(),
            "flow": context.get("flow"),
            "phase": phase,
        }
        payload.update(event_fields)
        return payload

    async def _dispatch_observers(self, hook_name: str, event: Dict[str, Any]) -> None:
        observers = list(getattr(self, "observers", []) or [])
        if not observers:
            return

        for observer in observers:
            hook = getattr(observer, hook_name, None)
            if hook is None:
                continue
            try:
                result = hook(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.warning(
                    "[%s] Observer '%s' failed during '%s': %s",
                    getattr(self, "name", self.__class__.__name__),
                    observer.__class__.__name__,
                    hook_name,
                    exc,
                    exc_info=True,
                )

    async def _emit_observer_hook(self, hook_name: str, phase: str, **event_fields: Any) -> None:
        event = self._build_observer_event(phase=phase, **event_fields)
        await self._dispatch_observers(hook_name, event)

    async def _emit_observer_error(self, phase: str, error: Exception, **event_fields: Any) -> None:
        await self._emit_observer_hook(
            "on_error",
            phase=phase,
            error=str(error),
            error_type=type(error).__name__,
            success=False,
            **event_fields,
        )

    @staticmethod
    def _usage_from_response(response: Any) -> Optional[Dict[str, int]]:
        usage = {
            "prompt_tokens": getattr(response, "prompt_tokens", None),
            "completion_tokens": getattr(response, "completion_tokens", None),
            "total_tokens": getattr(response, "total_tokens", None),
            "thinking_tokens": getattr(response, "thinking_tokens", None),
        }
        filtered_usage = {key: value for key, value in usage.items() if value is not None}
        return filtered_usage or None

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((asyncio.get_running_loop().time() - started_at) * 1000))

    @staticmethod
    def _normalize_guardrail_limit(
        value: Optional[int],
        field_name: str,
        *,
        allow_zero: bool,
    ) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer or None")

        min_allowed = 0 if allow_zero else 1
        if value < min_allowed:
            comparator = ">="
            raise ValueError(f"{field_name} must be {comparator} {min_allowed} or None")

        return value

    @staticmethod
    def _build_tool_call_budget_error(
        max_total_tool_calls: int,
        dispatched_tool_calls: int,
        requested_tool_calls: int,
    ) -> str:
        return (
            "Error: Guardrail reached: max_total_tool_calls "
            f"({max_total_tool_calls}) exceeded. "
            f"Already dispatched={dispatched_tool_calls}, requested_batch={requested_tool_calls}."
        )

    def _resolve_tool_execution_policy(self, tool_instance: Any) -> Optional[ToolExecutionPolicy]:
        """Return normalized execution policy for a tool instance (if configured)."""
        policy_getter = getattr(tool_instance, "get_execution_policy", None)
        if callable(policy_getter):
            try:
                return policy_getter()
            except Exception as exc:
                logger.warning(
                    "[%s] Invalid execution policy on tool '%s': %s",
                    self.name,
                    getattr(tool_instance, "name", "unknown"),
                    exc,
                    exc_info=True,
                )
                return None

        raw_policy = getattr(tool_instance, "execution_policy", None)
        if raw_policy is None:
            return None

        try:
            return ToolExecutionPolicy.coerce(raw_policy)
        except Exception as exc:
            logger.warning(
                "[%s] Invalid execution policy on tool '%s': %s",
                self.name,
                getattr(tool_instance, "name", "unknown"),
                exc,
                exc_info=True,
            )
            return None

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

        policy = self._resolve_tool_execution_policy(tool_instance)
        max_attempts = policy.max_attempts if policy is not None else 1

        for attempt in range(1, max_attempts + 1):
            attempt_started_at = asyncio.get_running_loop().time()
            await self._emit_observer_hook(
                "on_tool_call_start",
                phase="tool",
                tool_name=tool_name,
                args=kwargs,
                attempt=attempt,
                max_attempts=max_attempts,
                timeout_ms=policy.timeout_ms if policy is not None else None,
            )

            try:
                if policy is not None and policy.timeout_seconds is not None:
                    result = await asyncio.wait_for(
                        tool_instance.run_async(**kwargs),
                        timeout=policy.timeout_seconds,
                    )
                else:
                    result = await tool_instance.run_async(**kwargs)

                latency_ms = self._elapsed_ms(attempt_started_at)
                await self._emit_observer_hook(
                    "on_tool_call_end",
                    phase="tool",
                    tool_name=tool_name,
                    args=kwargs,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=latency_ms,
                    success=True,
                )
                return {
                    "success": True,
                    "result": result,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                }
            except asyncio.CancelledError as exc:
                # Cancellation should propagate and must never be retried.
                latency_ms = self._elapsed_ms(attempt_started_at)
                await self._emit_observer_error(
                    phase="tool",
                    error=exc,
                    tool_name=tool_name,
                    args=kwargs,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=latency_ms,
                    cancelled=True,
                )
                await self._emit_observer_hook(
                    "on_tool_call_end",
                    phase="tool",
                    tool_name=tool_name,
                    args=kwargs,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=latency_ms,
                    cancelled=True,
                    success=False,
                )
                raise
            except Exception as exc:
                latency_ms = self._elapsed_ms(attempt_started_at)

                if (
                    isinstance(exc, asyncio.TimeoutError)
                    and policy is not None
                    and policy.timeout_ms is not None
                ):
                    error_message = f"Tool '{tool_name}' timed out after {policy.timeout_ms} ms"
                else:
                    error_message = str(exc)

                should_retry = (
                    attempt < max_attempts
                    and policy is not None
                    and policy.is_retryable_error(exc)
                )

                await self._emit_observer_error(
                    phase="tool",
                    error=exc,
                    tool_name=tool_name,
                    args=kwargs,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=latency_ms,
                    will_retry=should_retry,
                )
                await self._emit_observer_hook(
                    "on_tool_call_end",
                    phase="tool",
                    tool_name=tool_name,
                    args=kwargs,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=latency_ms,
                    success=False,
                    error=error_message,
                    error_type=type(exc).__name__,
                    will_retry=should_retry,
                )

                if not should_retry:
                    return {
                        "success": False,
                        "error": error_message,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    }

                # delay uses the current 1-based attempt number.
                delay_seconds = policy.backoff.get_delay_seconds(retry_number=attempt)
                if self.verbose:
                    logger.warning(
                        "[%s] Retrying tool '%s' (%s/%s) after error: %s",
                        self.name,
                        tool_name,
                        attempt,
                        max_attempts - 1,
                        error_message,
                    )
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)

        return {
            "success": False,
            "error": f"Tool '{tool_name}' failed without producing a result",
            "attempt": max_attempts,
            "max_attempts": max_attempts,
        }

    @staticmethod
    def _normalize_tool_result_envelope(tool_name: str, raw_result: Any) -> ToolResultEnvelope:
        """Normalize raw tool execution output into canonical envelope shape."""
        return ToolResultEnvelope.from_tool_execution_result(
            tool_name=tool_name,
            raw_result=raw_result,
        )

    async def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Any]:
        """Execute tool calls with optional concurrency guardrail.

        When ``max_concurrent_tool_calls`` is configured, execution is performed
        in explicit batches of that size. Any remainder is handled as the final
        partial batch.
        """
        if not tool_calls:
            return []

        max_concurrent_tool_calls = getattr(self, "max_concurrent_tool_calls", None)
        if max_concurrent_tool_calls is None:
            tool_tasks = [self.execute_tool(tc.name, **tc.arguments) for tc in tool_calls]
            return await asyncio.gather(*tool_tasks, return_exceptions=True)

        max_concurrent_tool_calls = max(1, max_concurrent_tool_calls)
        results: List[Any] = []

        for batch_start in range(0, len(tool_calls), max_concurrent_tool_calls):
            batch = tool_calls[batch_start: batch_start + max_concurrent_tool_calls]
            batch_tasks = [self.execute_tool(tc.name, **tc.arguments) for tc in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)

        return results
        
    async def _execute_tools_concurrently(self, tool_calls: List[ToolCall]) -> List[Message]:
        """
        Shared logic for both stream() and invoke().
        Takes a list of ToolCalls, runs them concurrently, and returns the exact
        List[Message] objects with role="tool" and matched tool_call_ids.
        """
        results = await self._execute_tool_calls(tool_calls)

        tool_messages = []
        for tool_call, result in zip(tool_calls, results):
            envelope = self._normalize_tool_result_envelope(tool_call.name, result)

            # Append a separate message for each tool with its specific ID
            tool_messages.append(
                Message(
                    role="tool",
                    content=envelope.to_json(),
                    tool_call_id=tool_call.id
                ))

        return tool_messages

    async def _orchestrate_stream(
        self,
        messages: List[Message],
        formatted_tools: Optional[List] = None,
        include_thinking = False,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Streaming orchestration engine.
        
        Implements the multi-turn tool calling loop with streaming support.
        Yields content chunks to the UI while accumulating tool calls.
        
        Args:
            messages: Current message history
            formatted_tools: Formatted tools for the LLM provider
            include_thinking
            **kwargs: Extra LLM parameters forwarded to the client
            
        Yields:
            StreamChunk objects with content chunks progressively
        """
        max_iterations = getattr(self, 'max_iterations', 5)
        iteration = 0
        total_tool_calls_dispatched = 0
        
        while iteration < max_iterations:
            accumulated_content = ""
            accumulated_thinking = ""
            accumulated_thinking_tokens = None
            accumulated_tool_calls = []
            model_call_started_at = asyncio.get_running_loop().time()
            await self._emit_observer_hook(
                "on_model_call_start",
                phase="model",
                iteration=iteration + 1,
                max_iterations=max_iterations,
                message_count=len(messages),
            )

            final_finish_reason = None
            try:
                # Call LLM with streaming
                active_tools = formatted_tools if formatted_tools is not None else self._get_request_formatted_tools()
                model_kwargs = dict(kwargs)
                if include_thinking and getattr(self, "provider", None) == "gemini":
                    model_kwargs.setdefault("include_thoughts", True)
                stream = self.llm.chat_completion_stream(
                    messages=messages,
                    system_message=Message(content=self._get_request_system_prompt(), role='system'),
                    tools=active_tools,
                    **model_kwargs
                )

                # Stream the response
                async for chunk in stream:
                    if chunk.finish_reason is not None:
                        final_finish_reason = chunk.finish_reason

                    chunk_thinking_tokens = chunk.thinking_tokens
                    if chunk_thinking_tokens is None and isinstance(chunk.usage, dict):
                        usage_thinking_tokens = chunk.usage.get("thinking_tokens")
                        if isinstance(usage_thinking_tokens, (int, float)) and not isinstance(usage_thinking_tokens, bool):
                            chunk_thinking_tokens = int(usage_thinking_tokens)

                    if chunk_thinking_tokens is not None:
                        accumulated_thinking_tokens = chunk_thinking_tokens

                    # Yield thinking contents to the UI
                    if chunk.thinking:
                        accumulated_thinking += chunk.thinking or ""
                        if include_thinking:
                            yield StreamChunk(
                                thinking=chunk.thinking,
                                thinking_tokens=chunk_thinking_tokens,
                                is_finished=chunk.is_finished,
                                finish_reason=chunk.finish_reason,
                                usage=chunk.usage,
                            )

                    # Yield content chunks to the UI
                    if chunk.content:
                        yield StreamChunk(
                            content=chunk.content,
                            thinking_tokens=chunk_thinking_tokens,
                            is_finished=chunk.is_finished,
                            finish_reason=chunk.finish_reason,
                            usage=chunk.usage,
                        )
                        accumulated_content += chunk.content
                    elif chunk.is_finished and not (chunk.thinking and include_thinking):
                        # Ensure callers waiting for terminal chunk get completion
                        # signal even when model emitted no textual content
                        # (e.g., tool-call-only turns).
                        yield StreamChunk(
                            is_finished=True,
                            finish_reason=chunk.finish_reason,
                            thinking_tokens=chunk_thinking_tokens,
                            usage=chunk.usage,
                        )

                    # Accumulate tool calls JSON (silently)
                    if chunk.tool_calls:
                        accumulated_tool_calls.extend(chunk.tool_calls)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._emit_observer_error(
                    phase="model",
                    error=exc,
                    iteration=iteration + 1,
                    max_iterations=max_iterations,
                    latency_ms=self._elapsed_ms(model_call_started_at),
                )
                raise

            await self._emit_observer_hook(
                "on_model_call_end",
                phase="model",
                iteration=iteration + 1,
                max_iterations=max_iterations,
                latency_ms=self._elapsed_ms(model_call_started_at),
                success=True,
                finish_reason=final_finish_reason,
                tool_call_count=len(accumulated_tool_calls),
            )
            
            # If there are tool calls, execute them and continue the loop
            if accumulated_tool_calls:
                max_total_tool_calls = getattr(self, "max_total_tool_calls", None)
                projected_tool_calls = total_tool_calls_dispatched + len(accumulated_tool_calls)

                if (
                    max_total_tool_calls is not None
                    and projected_tool_calls > max_total_tool_calls
                ):
                    guardrail_message = self._build_tool_call_budget_error(
                        max_total_tool_calls=max_total_tool_calls,
                        dispatched_tool_calls=total_tool_calls_dispatched,
                        requested_tool_calls=len(accumulated_tool_calls),
                    )
                    guardrail_error = RuntimeError(guardrail_message)
                    await self._emit_observer_error(
                        phase="tool",
                        error=guardrail_error,
                        guardrail="max_total_tool_calls",
                        max_total_tool_calls=max_total_tool_calls,
                        dispatched_tool_calls=total_tool_calls_dispatched,
                        requested_tool_calls=len(accumulated_tool_calls),
                    )

                    if accumulated_content or accumulated_thinking:
                        messages.append(Message(
                            role="ai",
                            content=accumulated_content,
                            thinking=accumulated_thinking or None,
                            thinking_tokens=accumulated_thinking_tokens,
                        ))
                    messages.append(Message(role="ai", content=guardrail_message))

                    yield StreamChunk(
                        content=guardrail_message,
                        is_finished=True,
                        finish_reason="guardrail_reached",
                    )
                    break

                total_tool_calls_dispatched = projected_tool_calls

                # After guardrail checks pass, append the AI tool-call message.
                messages.append(Message(
                    role="ai",
                    content=accumulated_content,
                    thinking=accumulated_thinking or None,
                    thinking_tokens=accumulated_thinking_tokens,
                    tool_calls=accumulated_tool_calls
                ))

                # Emit a "start" event for every pending tool call before
                # dispatching so consumers can show real-time step indicators.
                for tc in accumulated_tool_calls:
                    yield StreamChunk(
                        tool_call=ToolCallEvent(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            args=tc.arguments,
                            status="start",
                        )
                    )

                # Execute all tools concurrently.  We need per-call results to
                # emit individual success/error events, so we fan out manually
                # instead of going through _execute_tools_concurrently.
                raw_results = await self._execute_tool_calls(accumulated_tool_calls)

                tool_messages = []
                for tc, raw in zip(accumulated_tool_calls, raw_results):
                    envelope = self._normalize_tool_result_envelope(tc.name, raw)
                    if envelope.status == "success":
                        yield StreamChunk(
                            tool_call=ToolCallEvent(
                                tool_call_id=tc.id,
                                tool_name=tc.name,
                                args=tc.arguments,
                                result=envelope.result,
                                status="success",
                            )
                        )
                    else:
                        yield StreamChunk(
                            tool_call=ToolCallEvent(
                                tool_call_id=tc.id,
                                tool_name=tc.name,
                                args=tc.arguments,
                                error=envelope.error,
                                status="error",
                            )
                        )

                    tool_messages.append(
                        Message(
                            role="tool",
                            content=envelope.to_json(),
                            tool_call_id=tc.id,
                        )
                    )

                messages.extend(tool_messages)
                iteration += 1
                continue

            # No tool calls on this model turn, append regular AI response.
            messages.append(Message(
                role="ai",
                content=accumulated_content,
                thinking=accumulated_thinking or None,
                thinking_tokens=accumulated_thinking_tokens,
                tool_calls=accumulated_tool_calls
            ))
            
            # No tool calls - we're done
            break
        else:
            # while condition exhausted — max_iterations reached without resolving
            # all tool calls.  Yield a terminal error chunk so the consumer is never
            # left hanging waiting for is_finished=True.
            yield StreamChunk(
                content=f"Error: Maximum tool call iterations ({max_iterations}) reached.",
                is_finished=True,
                finish_reason="max_iterations",
            )

    async def _orchestrate_invoke(
        self,
        messages: List[Message],
        **kwargs
    ) -> str:
        """
        Blocking orchestration engine.
        
        Implements the multi-turn tool calling loop with standard (non-streaming) calls.
        
        Args:
            messages: Current message history
            formatted_tools: Formatted tools for the LLM provider
            **kwargs: Extra LLM parameters forwarded to the client
            
        Returns:
            Final response string after all tool calls are complete
        """

        # Get request-local config snapshot if available
        formatted_tools = self._get_request_formatted_tools()
        system_prompt = self._get_request_system_prompt()
        
        max_iterations = getattr(self, 'max_iterations', 5)
        iteration = 0
        total_tool_calls_dispatched = 0

        
        while iteration < max_iterations:
            model_call_started_at = asyncio.get_running_loop().time()
            await self._emit_observer_hook(
                "on_model_call_start",
                phase="model",
                iteration=iteration + 1,
                max_iterations=max_iterations,
                message_count=len(messages),
            )

            # Call the standard (non-streaming) LLM
            try:
                response = await self.llm.chat_completion_async(
                    messages=messages,
                    system_message=Message(content=system_prompt, role='system'),
                    tools=formatted_tools,
                    **kwargs
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._emit_observer_error(
                    phase="model",
                    error=exc,
                    iteration=iteration + 1,
                    max_iterations=max_iterations,
                    latency_ms=self._elapsed_ms(model_call_started_at),
                )
                raise

            await self._emit_observer_hook(
                "on_model_call_end",
                phase="model",
                iteration=iteration + 1,
                max_iterations=max_iterations,
                latency_ms=self._elapsed_ms(model_call_started_at),
                success=True,
                finish_reason=getattr(response, "finish_reason", None),
                usage=self._usage_from_response(response),
                tool_call_count=len(response.tool_calls or []),
            )
            
            # If there are tool calls, execute them and continue the loop
            if response.tool_calls:
                max_total_tool_calls = getattr(self, "max_total_tool_calls", None)
                projected_tool_calls = total_tool_calls_dispatched + len(response.tool_calls)

                if (
                    max_total_tool_calls is not None
                    and projected_tool_calls > max_total_tool_calls
                ):
                    guardrail_message = self._build_tool_call_budget_error(
                        max_total_tool_calls=max_total_tool_calls,
                        dispatched_tool_calls=total_tool_calls_dispatched,
                        requested_tool_calls=len(response.tool_calls),
                    )
                    guardrail_error = RuntimeError(guardrail_message)
                    await self._emit_observer_error(
                        phase="tool",
                        error=guardrail_error,
                        guardrail="max_total_tool_calls",
                        max_total_tool_calls=max_total_tool_calls,
                        dispatched_tool_calls=total_tool_calls_dispatched,
                        requested_tool_calls=len(response.tool_calls),
                    )

                    if response.content:
                        messages.append(Message(role="ai", content=response.content))
                    messages.append(Message(role="ai", content=guardrail_message))
                    return guardrail_message

                total_tool_calls_dispatched = projected_tool_calls

                # Append the response message to history only when dispatch proceeds.
                messages.append(response.to_message())
                tool_messages = await self._execute_tools_concurrently(response.tool_calls)
                messages.extend(tool_messages)
                iteration += 1
                continue

            # Append final text-only response.
            messages.append(response.to_message())
            
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
            "max_iterations": self.max_iterations,
            "max_total_tool_calls": self.max_total_tool_calls,
            "max_concurrent_tool_calls": self.max_concurrent_tool_calls,
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
        """Store conversation in memory if available.

        Rollover is checked ONCE before the batch is written, not per-message.
        This ensures that an agent turn (human + optional tool loop + ai reply)
        is always stored atomically within a single bucket.  Splitting a
        functionCall/functionResponse pair across a bucket boundary would corrupt
        conversation history and cause 400 errors on providers like Gemini that
        require strict turn ordering.
        """
        if not self.memory:
            return
        
        try:
            # Check rollover once for the whole batch upfront.
            # If the threshold is already reached before we write anything,
            # roll the bucket over now so that every message in this interaction
            # lands in the fresh bucket together.
            if (
                hasattr(self.memory, "rollover_enabled")
                and self.memory.rollover_enabled
                and hasattr(self.memory, "should_rollover")
                and await self.memory.should_rollover(owner_id, chat_id)
            ):
                await self.memory.rollover_history(owner_id, chat_id, summarize=True)

            # Write all messages with per-message rollover suppressed so the
            # batch cannot be split by a second threshold hit mid-loop.
            for msg in messages:
                await self.memory.add_message(
                    msg, owner_id=owner_id, chat_id=chat_id, defer_rollover=True
                )
            
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