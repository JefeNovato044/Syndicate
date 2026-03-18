"""
MCPSessionManager — Persistent MCP server lifecycle manager.

Manages long-lived connections to MCP servers. Each registered server gets
exactly one ClientSession that stays open for the app's lifetime. Discovery
runs once on start(), producing a proper BaseTool subclass per MCP sub-tool —
each with its own JSON schema, name, and description.

Usage (context manager — scripts / tests):

    async with MCPSessionManager() as mgr:
        mgr.register(
            "sora",
            command="/path/.venv/bin/python",
            args=["server.py", "--transport", "stdio"],
            env={"SORA_API_TOKEN": "...", "SORA_URL": "..."},
        )
        await mgr.start()

        agent = GenericAgent(
            llm_client=client,
            tools=mgr.get_tools("sora"),   # [sora_list_employees, sora_create_employee, ...]
        )
        response = await agent.invoke("List all employees")

Usage (explicit lifecycle — FastAPI / long-lived apps):

    mgr = MCPSessionManager()
    mgr.register("sora", command=..., args=..., env=...)

    # on startup:
    await mgr.start()

    # on shutdown:
    await mgr.close()
"""

import json
import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tools.base_tool import BaseTool, _clean_schema_for_gemini

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCPSubTool — one BaseTool per discovered MCP sub-tool
# ---------------------------------------------------------------------------

class MCPSubTool(BaseTool):
    """
    A fully-formed BaseTool backed by a single MCP server sub-tool.

    Created by MCPSessionManager.start() — do NOT instantiate directly.

    Each instance:
    - Has its own name  (e.g. "sora_create_employee")
    - Has its own description and JSON schema sourced directly from the MCP server
    - Shares the parent server's live ClientSession (no reconnect per call)
    - Flows through BaseTool.to_format() like any other Syndicate tool
    """

    # args_schema is not used — schema methods are overridden directly
    args_schema = None

    def __init__(
        self,
        tool_name: str,
        server_prefix: str,
        description: str,
        input_schema: Dict[str, Any],
        session: ClientSession,
    ) -> None:
        self._mcp_tool_name = tool_name
        self._session = session
        self._raw_schema = input_schema

        # Public attributes read by BaseTool and get_formatted_tools()
        self.name = f"{server_prefix}_{tool_name}"
        self.description = description or f"MCP tool '{tool_name}' on server '{server_prefix}'"

    # -- Abstract method implementation --

    def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "MCPSubTool is async-only. "
            "The framework calls run_async(); never call run() directly."
        )

    async def run_async(self, **kwargs: Any) -> str:
        """
        Execute the MCP tool call against the shared live session.

        kwargs are forwarded as-is to the MCP server (server validates them).
        Returns a plain string suitable for storing in conversation history.
        """
        try:
            result = await self._session.call_tool(
                self._mcp_tool_name,
                arguments=kwargs,
            )
            # Flatten MCP content blocks into a single string
            if hasattr(result, "content"):
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "\n".join(parts) if parts else ""
            return str(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def to_format(self, provider: str) -> Dict[str, Any]:
        """
        Override BaseTool's default formatter to pass raw MCP schemas.
        """
        if provider == 'gemini':
            return {
                "name": self.name,
                "description": self.description,
                "parameters": _clean_schema_for_gemini(self._raw_schema),
            }
        elif provider == 'openai':
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self._raw_schema,
                },
            }
        return super().to_format(provider)


# ---------------------------------------------------------------------------
# Internal server config
# ---------------------------------------------------------------------------

class _ServerConfig:
    def __init__(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]],
    ) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.env = env


# ---------------------------------------------------------------------------
# MCPSessionManager
# ---------------------------------------------------------------------------

class MCPSessionManager:
    """
    Manages persistent MCP server connections for agents.

    Servers are registered → sessions opened once in start() → tools vended
    to agents via get_tools(). All tool calls share the same live session.

    See module docstring for full usage examples.
    """

    def __init__(self) -> None:
        self._configs: Dict[str, _ServerConfig] = {}
        # Live ClientSession per server
        self._sessions: Dict[str, ClientSession] = {}
        # Per-server list of MCPSubTool instances (one per discovered sub-tool)
        self._tools: Dict[str, List[MCPSubTool]] = {}
        # Raw context-manager objects held open to keep sessions alive
        self._stdio_ctxs: Dict[str, Any] = {}
        self._session_ctxs: Dict[str, Any] = {}

    # -- Registration --

    def register(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "MCPSessionManager":
        """
        Register an MCP server. Must be called before start().

        Args:
            name:    Logical identifier used as prefix in tool names (e.g. "sora").
                     Agent tools will be named "{name}_{mcp_tool_name}".
            command: Executable that launches the server process
                     (e.g. "/path/to/.venv/bin/python").
            args:    Command-line arguments
                     (e.g. ["/path/to/server.py", "--transport", "stdio"]).
            env:     Extra environment variables forwarded to the subprocess
                     (e.g. {"SORA_API_TOKEN": "...", "SORA_URL": "..."}).

        Returns:
            self — chainable.
        """
        self._configs[name] = _ServerConfig(
            name=name, command=command, args=args or [], env=env
        )
        return self

    # -- Lifecycle --

    async def start(self) -> None:
        """
        Open sessions and discover tools for all registered servers concurrently.

        Call once after all register() calls (or on app startup).
        Already-started servers are skipped safely.

        Servers that fail to start are logged and skipped — healthy
        servers remain available.  Check ``list_started()`` after this
        call to see which servers came up.
        """
        servers_to_start = [
            (name, cfg)
            for name, cfg in self._configs.items()
            if name not in self._sessions
        ]
        results = await asyncio.gather(
            *(self._start_one(name, cfg) for name, cfg in servers_to_start),
            return_exceptions=True,
        )
        # Log failures and clean up their partial state, but leave
        # successfully-started servers running.
        for (name, _), result in zip(servers_to_start, results):
            if isinstance(result, BaseException):
                logger.error(
                    "MCP server '%s' failed to start: %s", name, result,
                    exc_info=result,
                )
                await self._close_one(name)

    async def _start_one(self, name: str, config: _ServerConfig) -> None:
        """Open one server's session and build its MCPSubTool list."""
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
        )

        # Manually enter context managers and store them so they stay alive
        # for the full lifetime of the manager (not just this coroutine).
        stdio_ctx = stdio_client(params)
        read, write = await stdio_ctx.__aenter__()
        self._stdio_ctxs[name] = stdio_ctx

        session_ctx = ClientSession(read, write)
        session: ClientSession = await session_ctx.__aenter__()
        self._session_ctxs[name] = session_ctx

        await session.initialize()
        self._sessions[name] = session

        # Discover available tools and build one MCPSubTool per sub-tool
        response = await session.list_tools()
        self._tools[name] = [
            MCPSubTool(
                tool_name=mcp_tool.name,
                server_prefix=name,
                description=mcp_tool.description or "",
                input_schema=mcp_tool.inputSchema,
                session=session,
            )
            for mcp_tool in response.tools
        ]

    async def close(self) -> None:
        """
        Close all open sessions gracefully. Call on app shutdown.

        Iterates all *registered* servers (not just fully-started ones) so
        that partially-opened connections from a failed start() are also
        cleaned up. _close_one() is safe to call on servers that never
        opened — it checks for None before tearing down.
        """
        for name in reversed(list(self._configs.keys())):
            await self._close_one(name)

    async def _close_one(self, name: str) -> None:
        try:
            ctx = self._session_ctxs.pop(name, None)
            if ctx:
                await ctx.__aexit__(None, None, None)
        except Exception:
            logger.warning("Error closing MCP session for '%s'", name, exc_info=True)
        try:
            ctx = self._stdio_ctxs.pop(name, None)
            if ctx:
                await ctx.__aexit__(None, None, None)
        except Exception:
            logger.warning("Error closing MCP stdio for '%s'", name, exc_info=True)
        self._sessions.pop(name, None)
        self._tools.pop(name, None)

    # -- Tool vending --

    def get_tools(self, server_name: str) -> List[MCPSubTool]:
        """
        Return all MCPSubTool instances for a started server.

        Pass the returned list directly into an agent's tools=[...].

        Args:
            server_name: Name passed to register().

        Returns:
            List of MCPSubTool instances — one per MCP sub-tool discovered.

        Raises:
            KeyError:     Server was never registered.
            RuntimeError: start() has not been called yet.
        """
        if server_name not in self._configs:
            raise KeyError(
                f"No server registered as '{server_name}'. "
                f"Registered servers: {list(self._configs)}"
            )
        if server_name not in self._tools:
            raise RuntimeError(
                f"Session for '{server_name}' not started. "
                "Call `await mgr.start()` before requesting tools."
            )
        return self._tools[server_name]

    def get_tool(self, server_name: str, tool_name: str) -> MCPSubTool:
        """
        Get a single MCPSubTool by its original MCP name (without server prefix).

        Useful when you want to add only specific sub-tools to an agent.

        Args:
            server_name: Name passed to register().
            tool_name:   MCP tool name exactly as the server exposes it
                         (e.g. "create_employee", not "sora_create_employee").

        Raises:
            KeyError: Tool not found on that server.
        """
        for tool in self.get_tools(server_name):
            if tool._mcp_tool_name == tool_name:
                return tool
        available = [t._mcp_tool_name for t in self.get_tools(server_name)]
        raise KeyError(
            f"Tool '{tool_name}' not found on server '{server_name}'. "
            f"Available: {available}"
        )

    # -- Introspection --

    def list_servers(self) -> List[str]:
        """Names of all registered servers."""
        return list(self._configs.keys())

    def list_started(self) -> List[str]:
        """Names of servers with active sessions."""
        return list(self._sessions.keys())

    def list_tools(self, server_name: str) -> List[str]:
        """
        MCP tool names discovered on a started server (without server prefix).

        Returns empty list if server not started yet.
        """
        return [t._mcp_tool_name for t in self._tools.get(server_name, [])]

    # -- Async context manager --

    async def __aenter__(self) -> "MCPSessionManager":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()