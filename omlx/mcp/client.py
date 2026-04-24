# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
MCP client for connecting to individual MCP servers.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .types import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)

logger = logging.getLogger(__name__)


def _is_auth_failure(exc: Exception) -> bool:
    """Return True when *exc* represents an HTTP 401 / unauthorized error."""
    msg = str(exc).lower()
    return "401" in msg or "unauthorized" in msg


class MCPClient:
    """
    Client for connecting to a single MCP server.

    Supports both stdio and SSE transports.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self._session = None
        self._read = None
        self._write = None
        self._tools: List[MCPTool] = []
        self._state = MCPServerState.DISCONNECTED
        self._error: Optional[str] = None
        self._last_connected: Optional[float] = None
        self._lock = asyncio.Lock()
        self._transport_tasks: List = []  # list of (task, disconnect_event)

    @property
    def name(self) -> str:
        """Get server name."""
        return self.config.name

    @property
    def state(self) -> MCPServerState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == MCPServerState.CONNECTED

    @property
    def tools(self) -> List[MCPTool]:
        """Get discovered tools."""
        return self._tools

    def get_status(self) -> MCPServerStatus:
        """Get server status."""
        auth_state: Optional[str] = None
        if self.config.auth:
            from .oauth import MCPOAuthManager  # lazy import
            manager = MCPOAuthManager()
            info = manager.get_token_info(self.name)
            if info is None:
                auth_state = "not_authenticated"
            elif info.get("is_expired"):
                auth_state = "token_expired"
            else:
                auth_state = "authenticated"

        return MCPServerStatus(
            name=self.name,
            state=self._state,
            transport=self.config.transport,
            tools_count=len(self._tools),
            error=self._error,
            last_connected=self._last_connected,
            auth_state=auth_state,
        )

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        For HTTP-based transports configured with OAuth, the client will
        attempt to acquire / refresh a token before connecting and retry
        once if the server returns a 401.

        Returns:
            True if connection successful, False otherwise
        """
        async with self._lock:
            if self._state == MCPServerState.CONNECTED:
                return True

            if not self.config.enabled:
                logger.info(f"MCP server '{self.name}' is disabled")
                return False

            self._state = MCPServerState.CONNECTING
            self._error = None

            try:
                # Eagerly fetch a stored / refreshed token for HTTP transports.
                if self.config.auth and self.config.transport in (
                    MCPTransport.SSE,
                    MCPTransport.STREAMABLE_HTTP,
                ):
                    token = await self._get_oauth_token()
                    if token:
                        self._inject_auth_header(token)

                await self._do_connect()

            except Exception as first_exc:
                # On auth failure, try to obtain a fresh token and retry once.
                if self.config.auth and _is_auth_failure(first_exc):
                    logger.info(
                        f"MCP server '{self.name}' returned 401; "
                        "attempting to refresh auth token and retry"
                    )
                    try:
                        token = await self._get_oauth_token(force_refresh=True)
                        if token:
                            self._inject_auth_header(token)
                            await self._cleanup_resources()
                            await self._do_connect()
                        else:
                            raise RuntimeError(
                                f"No valid OAuth token available for '{self.name}'. "
                                "Run 'omlx mcp login' to authenticate."
                            ) from first_exc
                    except Exception as retry_exc:
                        self._state = MCPServerState.ERROR
                        self._error = str(retry_exc)
                        logger.error(
                            f"Failed to connect to MCP server '{self.name}' "
                            f"after auth retry: {retry_exc}"
                        )
                        await self._cleanup_resources()
                        return False
                else:
                    self._state = MCPServerState.ERROR
                    self._error = str(first_exc)
                    logger.error(
                        f"Failed to connect to MCP server '{self.name}': {first_exc}"
                    )
                    await self._cleanup_resources()
                    return False

            self._state = MCPServerState.CONNECTED
            self._last_connected = time.time()
            auth_note = " (authenticated)" if self.config.auth else ""
            logger.info(
                f"Connected to MCP server '{self.name}'{auth_note} "
                f"({len(self._tools)} tools available)"
            )
            return True

    async def _do_connect(self) -> None:
        """Establish the transport connection and discover tools."""
        if self.config.transport == MCPTransport.STDIO:
            await self._connect_stdio()
        elif self.config.transport == MCPTransport.SSE:
            await self._connect_sse()
        elif self.config.transport == MCPTransport.STREAMABLE_HTTP:
            await self._connect_streamable_http()
        else:
            raise ValueError(f"Unknown transport: {self.config.transport}")

        await self._initialize_session()
        await self._discover_tools()

    async def _get_oauth_token(
        self, force_refresh: bool = False
    ) -> Optional[str]:
        """
        Retrieve an OAuth access token for this server.

        When *force_refresh* is True and a refresh token is available the
        stored token is refreshed before returning.
        """
        if self.config.auth is None:
            return None

        from .oauth import MCPOAuthManager  # lazy import

        manager = MCPOAuthManager()

        if force_refresh:
            from .token_store import TokenStore  # lazy import
            store = TokenStore(self.config.auth.token_store)
            existing = store.load(self.name)
            if existing and existing.refresh_token:
                try:
                    refreshed = await manager._do_refresh(
                        self.config.auth, existing.refresh_token, stored_token=existing
                    )
                    store.save(self.name, refreshed)
                    return refreshed.access_token
                except Exception as exc:
                    logger.warning(
                        f"Force-refresh failed for '{self.name}': {exc}"
                    )
                    return None

        return await manager.get_access_token(self.name, self.config.auth, server_url=self.config.url)

    def _inject_auth_header(self, token: str) -> None:
        """Add / replace the Bearer token in the server's headers dict."""
        if self.config.headers is None:
            self.config.headers = {}
        self.config.headers["Authorization"] = f"Bearer {token}"

    async def _run_transport_in_task(self, cm, streams_future: asyncio.Future, disconnect_event: asyncio.Event, nstreams: int = 2):
        """
        Enter *cm* (a transport context manager) inside this asyncio Task,
        hand the resulting streams back via *streams_future*, then hold the
        connection open until *disconnect_event* is set.

        Running the context manager entirely within a single Task keeps
        anyio cancel-scopes satisfied — they require __aexit__ to be called
        from the same task as __aenter__.
        """
        try:
            result = await cm.__aenter__()
            streams = result[:nstreams] if nstreams < len(result) else result
            streams_future.set_result(streams)
            await disconnect_event.wait()
        except Exception as exc:
            if not streams_future.done():
                streams_future.set_exception(exc)
        finally:
            try:
                await cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("Transport __aexit__ raised (expected on cancel): %s", exc)

    async def _start_transport_task(self, cm, nstreams: int = 2):
        """
        Spawn a background Task for *cm*, wait for streams to be ready, and
        return them.  Stores the task and disconnect event on *self* so that
        _cleanup_resources can tear it down gracefully.
        """
        loop = asyncio.get_event_loop()
        streams_future: asyncio.Future = loop.create_future()
        disconnect_event = asyncio.Event()

        task = asyncio.create_task(
            self._run_transport_in_task(cm, streams_future, disconnect_event, nstreams)
        )
        try:
            streams = await asyncio.wait_for(
                asyncio.shield(streams_future),
                timeout=self.config.timeout,
            )
        except Exception:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise

        # Store so _cleanup_resources can signal teardown
        if not hasattr(self, "_transport_tasks"):
            self._transport_tasks = []
        self._transport_tasks.append((task, disconnect_event))
        return streams

    async def _connect_stdio(self):
        """Connect via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env,
        )
        self._read, self._write = await self._start_transport_task(
            stdio_client(server_params), nstreams=2
        )
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

    async def _connect_sse(self):
        """Connect via SSE transport."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        headers = self.config.headers or {}
        self._read, self._write = await self._start_transport_task(
            sse_client(self.config.url, headers=headers), nstreams=2
        )
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

    async def _connect_streamable_http(self):
        """Connect via streamable_http transport."""
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            import httpx
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        headers = self.config.headers or {}
        http_client = httpx.AsyncClient(headers=headers)
        # nstreams=2: streamable_http_client yields (read, write, get_session_id)
        self._read, self._write = await self._start_transport_task(
            streamable_http_client(url=self.config.url, http_client=http_client),
            nstreams=2,
        )
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

    async def _initialize_session(self):
        """Initialize the MCP session."""
        if self._session is None:
            raise RuntimeError("Session not created")

        # Initialize with capabilities
        result = await self._session.initialize()
        logger.debug(
            f"MCP server '{self.name}' initialized: "
            f"protocol={result.protocolVersion}, "
            f"server={result.serverInfo.name if result.serverInfo else 'unknown'}"
        )

    async def _discover_tools(self):
        """Discover available tools from the server."""
        if self._session is None:
            raise RuntimeError("Session not initialized")

        try:
            result = await self._session.list_tools()
            self._tools = []

            for tool in result.tools:
                mcp_tool = MCPTool(
                    server_name=self.name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {},
                )
                self._tools.append(mcp_tool)
                logger.debug(f"Discovered tool: {mcp_tool.full_name}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from '{self.name}': {e}")
            self._tools = []

    async def _cleanup_resources(self):
        """Clean up connection resources without acquiring lock."""
        try:
            if self._session:
                try:
                    await self._session.__aexit__(None, None, None)
                except Exception as exc:
                    logger.debug("Session __aexit__ raised for '%s': %s", self.name, exc)
                self._session = None

            # Signal all background transport tasks to disconnect and wait for them
            for task, disconnect_event in getattr(self, "_transport_tasks", []):
                disconnect_event.set()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as exc:
                    logger.debug("Transport task teardown for '%s': %s", self.name, exc)
                    task.cancel()
            self._transport_tasks = []

        except Exception as e:
            logger.warning(f"Error cleaning up resources for '{self.name}': {e}")

    async def disconnect(self):
        """Disconnect from the MCP server."""
        async with self._lock:
            if self._state == MCPServerState.DISCONNECTED:
                return

            try:
                await self._cleanup_resources()
            finally:
                self._state = MCPServerState.DISCONNECTED
                self._tools = []
                logger.info(f"Disconnected from MCP server '{self.name}'")

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> MCPToolResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool (without server prefix)
            arguments: Tool arguments
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        if not self.is_connected:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Not connected to server '{self.name}'",
            )

        if self._session is None:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message="Session not initialized",
            )

        try:
            # Call with timeout
            timeout = timeout or self.config.timeout

            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            # Extract content from result
            content = self._extract_content(result)

            return MCPToolResult(
                tool_name=tool_name,
                content=content,
                is_error=result.isError if hasattr(result, "isError") else False,
            )

        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Tool call timed out after {timeout}s",
            )
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=str(e),
            )

    def _extract_content(self, result) -> Any:
        """Extract content from MCP tool result."""
        if not hasattr(result, "content") or not result.content:
            # Fall back to structuredContent if available
            if hasattr(result, "structuredContent") and result.structuredContent:
                return result.structuredContent
            return None

        # Handle list of content items
        contents = []
        for item in result.content:
            if hasattr(item, "text"):
                contents.append(item.text)
            elif hasattr(item, "data"):
                contents.append(item.data)
            else:
                contents.append(str(item))

        # Return single item or list
        if len(contents) == 1:
            return contents[0]
        return contents

    async def refresh_tools(self):
        """Refresh the list of available tools."""
        if not self.is_connected:
            return

        await self._discover_tools()
