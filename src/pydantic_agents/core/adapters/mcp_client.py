import abc
from typing import Any


class MCPClientPort(abc.ABC):
    """
    Port (interface) for interacting with Model Context Protocol (MCP) servers.
    Provides standard methods to list and call tools exposed by the MCP server.
    """

    @abc.abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List all available tools provided by the MCP server.

        Returns:
            A list of tool definitions (e.g., schemas).
        """
        pass

    @abc.abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a specific tool on the MCP server with the given arguments.

        Args:
            tool_name: The name of the tool to execute.
            arguments: The arguments to pass to the tool.

        Returns:
            The result of the tool execution.
        """
        pass


class StandardMCPClient(MCPClientPort):
    """
    Concrete implementation of the MCPClientPort.
    This class handles the communication with an MCP server.
    """

    def __init__(self, server_name: str, connection_config: dict[str, Any] | None = None):
        """
        Initialize the StandardMCPClient.

        Args:
            server_name: The name or identifier of the MCP server.
            connection_config: Configuration for connecting to the server.
        """
        self.server_name = server_name
        self.connection_config = connection_config or {}
        self._is_connected = False

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        # TODO: Implement actual connection logic (e.g., using mcp SDK or custom transport)
        self._is_connected = True

    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        # TODO: Implement actual disconnection logic
        self._is_connected = False

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List all available tools provided by the MCP server.
        """
        if not self._is_connected:
            await self.connect()

        # TODO: Delegate to the actual MCP transport layer to fetch tools
        return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:  # noqa: ARG002
        """
        Call a specific tool on the MCP server with the given arguments.
        """
        if not self._is_connected:
            await self.connect()

        # TODO: Delegate to the actual MCP transport layer to execute the tool
        raise NotImplementedError(f"call_tool not yet implemented for tool: {tool_name}")
