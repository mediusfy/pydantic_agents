from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.agy_bridge.server import _handle_command
from pydantic_agents.core.exceptions import AgentNotFoundError


@pytest.mark.asyncio
async def test_ping_command():
    """Tests the 'ping' command handler."""
    cmd = {"command": "ping"}
    response = await _handle_command(cmd)

    assert "response" in response
    assert response["response"] == "pong"
    assert "timestamp" in response


@pytest.mark.asyncio
@patch("pydantic_agents.core.agy_bridge.server.registry")
@patch("pydantic_agents.core.agy_bridge.server.BridgeAgent")
async def test_list_agents_command(mock_bridge_agent, mock_registry):
    """Tests the 'list-agents' command handler."""
    # Arrange
    mock_registry.list_agents.return_value = ["test_agent"]

    mock_bridge_instance = MagicMock()
    mock_bridge_instance.get_tool_names.return_value = ["tool1"]
    mock_bridge_instance.get_system_prompt.return_value = "System prompt."
    mock_bridge_agent.from_registry.return_value = mock_bridge_instance

    # Act
    cmd = {"command": "list-agents"}
    response = await _handle_command(cmd)

    # Assert
    assert "response" in response
    assert response["response"]["count"] == 1
    agent_meta = response["response"]["agents"][0]
    assert agent_meta["name"] == "test_agent"

    mock_bridge_agent.from_registry.assert_called_once_with("test_agent")


@pytest.mark.asyncio
async def test_get_agent_command_not_found():
    """Tests the 'get-agent' command for a non-existent agent."""
    with patch("pydantic_agents.core.agy_bridge.server.BridgeAgent.from_registry", side_effect=AgentNotFoundError):
        cmd = {"command": "get-agent", "agent_name": "unknown"}
        response = await _handle_command(cmd)
        assert "error" in response
        assert "not registered" in response["error"]


@pytest.mark.asyncio
async def test_unknown_command():
    """Tests the handler's response to an unknown command."""
    cmd = {"command": "non-existent-command"}
    response = await _handle_command(cmd)
    assert "error" in response
    assert "unknown command" in response["error"]


@pytest.mark.asyncio
@patch("pydantic_agents.core.agy_bridge.server.sync_agent_definitions", new_callable=AsyncMock)
async def test_sync_definitions_command(mock_sync):
    """Tests the 'sync-definitions' command."""
    # Arrange
    mock_sync.return_value = 1

    # Act
    response = await _handle_command({"command": "sync-definitions"})

    # Assert
    assert response == {"response": {"synced": 1}}
    mock_sync.assert_called_once()
