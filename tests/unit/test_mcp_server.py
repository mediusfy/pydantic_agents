from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.mcp_server import run_agent_by_name


@pytest.fixture
def mock_registry():
    with patch("pydantic_agents.mcp_server.registry") as mock_reg:
        yield mock_reg


@pytest.mark.asyncio
async def test_run_agent_by_name_local(mock_registry):
    # Setup mock agent
    mock_agent = AsyncMock()
    mock_result = MagicMock()
    mock_result.output = "local_result_data"
    mock_agent.run.return_value = mock_result

    mock_registry.get.return_value = mock_agent
    mock_registry._agents = {"test_agent": (None, None, lambda: {}, None)}

    with patch("pydantic_agents.mcp_server.retrieve_context", return_value="some_context"):
        result = await run_agent_by_name("test_agent", "do local work")

    assert result == "local_result_data"
    mock_agent.run.assert_called_once()
