"""Regression tests for the bugs a local Sonar scan reported."""

import types
from unittest.mock import MagicMock

import pytest

from pydantic_agents.core.exceptions import AgentExecutionError
from pydantic_agents.core.session_recorder import current_run_context
from pydantic_agents.core.tools.docs import update_documentation


def test_run_context_default_is_immutable():
    default = current_run_context.get()
    assert isinstance(default, types.MappingProxyType)
    with pytest.raises(TypeError):
        default["leak"] = "state"  # type: ignore[index]


def test_run_context_default_is_still_usable_as_a_mapping():
    assert current_run_context.get().get("agent_name") is None
    assert {**current_run_context.get(), "agent_name": "x"} == {"agent_name": "x"}


async def test_bridge_agent_without_a_resolved_agent_raises_a_typed_error():
    from pydantic_agents.core.agy_bridge.bridge_agent import BridgeAgent

    agent = BridgeAgent.__new__(BridgeAgent)
    agent._name = "ghost"
    agent._agent = None
    agent._resolve = MagicMock()
    agent._hooks = MagicMock(pre_turn=[], post_turn=[], pre_tool_call=[], post_tool_call=[])
    with pytest.raises(AgentExecutionError, match="ghost"):
        await agent.run("hi")


@pytest.mark.parametrize(
    "mode, first, second, expected",
    [
        ("append", "one", "two", "\n\none\n\ntwo"),
        ("overwrite", "one", "two", "two"),
    ],
)
async def test_update_documentation_modes(tmp_path, monkeypatch, mode, first, second, expected):
    monkeypatch.chdir(tmp_path)
    await update_documentation(MagicMock(), "adr/001.md", first, mode="append")
    result = await update_documentation(MagicMock(), "adr/001.md", second, mode=mode)

    assert result == "Documentation updated: adr/001.md"
    assert (tmp_path / "docs" / "adr" / "001.md").read_text() == expected


async def test_update_documentation_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = await update_documentation(MagicMock(), "../../etc/evil.md", "x")
    assert result.startswith("Error")
    assert not (tmp_path / "etc").exists()
