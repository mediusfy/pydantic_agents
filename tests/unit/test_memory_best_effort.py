"""Memory tools are best-effort: an unreachable store must not abort the agent run."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.tools import memory

DB_ERRORS = [
    OSError("connection refused"),
    ConnectionError("reset by peer"),
    RuntimeError('password authentication failed for user "jocham"'),
    TimeoutError(),
]


@pytest.mark.parametrize("error", DB_ERRORS, ids=lambda e: type(e).__name__)
async def test_log_activity_survives_an_unreachable_store(error, caplog):
    with (
        patch.object(memory.vector_db, "async_add_documents", AsyncMock(side_effect=error)),
        caplog.at_level("WARNING", logger="pydantic_agents.core.tools.memory"),
    ):
        result = await memory.log_activity(MagicMock(deps=None), "Forecast", "BTC 60-70k")

    assert result.startswith("Activity not logged.")
    assert type(error).__name__ in result
    assert "memory store unavailable" in caplog.text


@pytest.mark.parametrize("error", DB_ERRORS, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("include_legacy", [False, True])
async def test_query_history_survives_an_unreachable_store(error, include_legacy):
    with patch.object(memory.vector_db, "async_search", AsyncMock(side_effect=error)):
        result = await memory.query_history(MagicMock(deps=None), "forecast", include_legacy=include_legacy)

    assert result.startswith("No history available.")
    assert type(error).__name__ in result


async def test_a_failing_legacy_lookup_also_degrades():
    search = AsyncMock(side_effect=[[], OSError("boom")])
    with patch.object(memory.vector_db, "async_search", search):
        result = await memory.query_history(MagicMock(deps=None), "q", include_legacy=True)
    assert result.startswith("No history available.")


async def test_error_text_does_not_echo_the_exception_message():
    secret_ish = RuntimeError("postgresql://user:hunter2@db:5432/x refused")
    with patch.object(memory.vector_db, "async_add_documents", AsyncMock(side_effect=secret_ish)):
        result = await memory.log_activity(MagicMock(deps=None), "a", "s")
    assert "hunter2" not in result
    assert "postgresql://" not in result


async def test_cancellation_still_propagates():
    with (
        patch.object(memory.vector_db, "async_add_documents", AsyncMock(side_effect=asyncio.CancelledError)),
        pytest.raises(asyncio.CancelledError),
    ):
        await memory.log_activity(MagicMock(deps=None), "a", "s")


async def test_healthy_store_behaviour_is_unchanged():
    with patch.object(memory.vector_db, "async_add_documents", AsyncMock()) as add:
        assert await memory.log_activity(MagicMock(deps=None), "Act", "s") == "Activity logged successfully: Act"
    add.assert_awaited_once()

    with patch.object(memory.vector_db, "async_search", AsyncMock(return_value=[])):
        assert await memory.query_history(MagicMock(deps=None), "q") == "No relevant history found."


async def test_an_agent_run_completes_when_the_store_is_down():
    """Reproduces the eval failure: the model calls log_activity and Postgres rejects the connection."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from pydantic_agents.agents.registration import register_all_agents
    from pydantic_agents.core.registry import registry

    register_all_agents()
    agent = registry.get("review")
    assert "log_activity" in agent._function_toolset.tools

    calls = []

    async def model(messages, info: AgentInfo):
        calls.append(len(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("log_activity", {"action": "Review", "summary": "looked at it"})])
        verdict = {"approved": True, "feedback": [], "security_score": 90, "risk_level": "LOW"}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, verdict)])

    down = AsyncMock(side_effect=RuntimeError('password authentication failed for user "jocham"'))
    with patch.object(memory.vector_db, "async_add_documents", down), agent.override(model=FunctionModel(model)):
        result = await agent.run("review this")

    assert result.output.approved is True
    assert len(calls) == 2
    down.assert_awaited_once()
