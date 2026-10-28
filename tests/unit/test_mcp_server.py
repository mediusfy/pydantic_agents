from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.mcp_server import approve_cto_gate, run_agent_by_name


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

    mock_registry.is_registered.return_value = True
    mock_registry.run = AsyncMock(return_value=mock_result)

    with patch("pydantic_agents.mcp_server.retrieve_context", AsyncMock(return_value="some_context")):
        result = await run_agent_by_name("test_agent", "do local work")

    assert result == "local_result_data"
    mock_registry.run.assert_awaited_once()
    prompt = mock_registry.run.await_args.args[1]
    assert prompt.startswith("Relevant Architecture Context from Knowledge Base:\nsome_context")
    assert prompt.endswith("do local work")


@pytest.mark.asyncio
async def test_approve_cto_gate_requires_adr_id_to_approve():
    """Mirrors AgentAdvice.approved_to_proceed's code gate: no approval without a recorded decision."""
    with patch("pydantic_agents.core.cto_gate.write_approval") as mock_write:
        result = await approve_cto_gate("proj", approved=True)

    assert result.startswith("ERROR")
    assert "adr_id" in result
    mock_write.assert_not_called()


@pytest.mark.asyncio
async def test_approve_cto_gate_rejection_needs_no_adr_id():
    with patch("pydantic_agents.core.cto_gate.write_approval") as mock_write:
        mock_write.return_value.name = ".cto_approved"
        result = await approve_cto_gate("proj", approved=False)

    assert "rejected" in result
    mock_write.assert_called_once_with("proj", False)


@pytest.mark.asyncio
async def test_approve_cto_gate_saves_the_adr_when_content_is_given():
    with (
        patch("pydantic_agents.core.cto_gate.write_approval") as mock_write,
        patch("pydantic_agents.mcp_server.save_adr") as mock_save,
    ):
        mock_write.return_value.name = ".cto_approved"
        result = await approve_cto_gate("proj", approved=True, adr_id="0002", adr_content="# Decision")

    mock_save.assert_called_once_with("0002", "# Decision")
    mock_write.assert_called_once_with("proj", True)
    assert "approved" in result
    assert "ADR 0002 saved" in result


@pytest.mark.asyncio
async def test_approve_cto_gate_approves_without_saving_when_no_content_given():
    with (
        patch("pydantic_agents.core.cto_gate.write_approval") as mock_write,
        patch("pydantic_agents.mcp_server.save_adr") as mock_save,
    ):
        mock_write.return_value.name = ".cto_approved"
        result = await approve_cto_gate("proj", approved=True, adr_id="0002")

    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_approve_cto_gate_rejects_content_without_an_adr_id():
    """adr_content with approved=False (or any call with a blank adr_id) must not call save_adr("", ...)."""
    with (
        patch("pydantic_agents.core.cto_gate.write_approval") as mock_write,
        patch("pydantic_agents.mcp_server.save_adr") as mock_save,
    ):
        result = await approve_cto_gate("proj", approved=False, adr_content="# Decision")

    assert result.startswith("ERROR")
    assert "adr_id" in result
    mock_save.assert_not_called()
    mock_write.assert_not_called()


@pytest.mark.asyncio
async def test_approve_cto_gate_reports_a_save_adr_failure_without_approving():
    with (
        patch("pydantic_agents.core.cto_gate.write_approval") as mock_write,
        patch("pydantic_agents.mcp_server.save_adr", side_effect=RuntimeError("db down")) as mock_save,
    ):
        result = await approve_cto_gate("proj", approved=True, adr_id="0002", adr_content="# Decision")

    assert result.startswith("ERROR")
    assert "save_adr failed" in result
    mock_save.assert_called_once_with("0002", "# Decision")
    mock_write.assert_not_called()
    assert "approved" in result
