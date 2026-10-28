"""JSON results, scoped agent memory and non-technical prompt guardrails."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.models.core import Artifact, UserStory
from pydantic_agents.core.registry import AgentRegistry, registry
from pydantic_agents.core.session_recorder import current_run_context
from pydantic_agents.core.tools import memory
from pydantic_agents.core.utils import serialize_output
from pydantic_agents.core.vector_db import VectorDBManager


@pytest.mark.parametrize(
    "output, check",
    [
        ("plain text", lambda s: s == "plain text"),
        (
            Artifact(agent_name="x", files={"a.go": "package a"}),
            lambda s: json.loads(s)["files"] == {"a.go": "package a"},
        ),
        ([UserStory(id="U1", title="t", gherkin="g")], lambda s: json.loads(s)[0]["id"] == "U1"),
        ({"k": 1}, lambda s: json.loads(s) == {"k": 1}),
    ],
)
def test_serialize_output_is_json_for_structured_results(output, check):
    assert check(serialize_output(MagicMock(output=output)))


async def test_registry_query_returns_json_for_model_output():
    reg = AgentRegistry()
    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output=Artifact(agent_name="x", files={"a": "b"})))
    reg.register("a", agent, ["cap"])
    assert json.loads(await reg.query("cap", "go"))["agent_name"] == "x"


# ── scoped memory ──────────────────────────────────────────────────────────


@pytest.fixture
def run_ctx():
    token = current_run_context.set({"agent_name": "crypto_oracle", "run_id": "run-1"})
    yield
    current_run_context.reset(token)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, {"agent_name": "crypto_oracle", "run_id": "run-1"}),
        ({"pass_num": 2}, {"agent_name": "crypto_oracle", "run_id": "run-1", "pass_num": "2"}),
        ({"all_agents": True}, {"run_id": "run-1"}),
    ],
)
async def test_query_history_is_scoped(run_ctx, kwargs, expected):
    with patch.object(memory.vector_db, "async_search", AsyncMock(return_value=[])) as search:
        await memory.query_history(MagicMock(deps=None), "forecast BTC", **kwargs)
    assert search.await_args.kwargs["filters"] == expected


async def test_log_activity_tags_agent_and_run(run_ctx):
    with patch.object(memory.vector_db, "async_add_documents", AsyncMock()) as add:
        await memory.log_activity(MagicMock(deps=None), "Crypto Forecast", "BTC base 60-70k")
    meta = add.await_args.args[2][0]
    assert (meta["agent_name"], meta["run_id"], meta["pass_num"]) == ("crypto_oracle", "run-1", "0")


async def test_vector_search_builds_parameterised_filters():
    calls = []

    class Conn:
        async def fetch(self, sql, *args):
            calls.append((sql, args))
            return [{"content": "row"}]

    class Ctx:
        async def __aenter__(self):
            return Conn()

        async def __aexit__(self, *_):
            return False

    class Pool:
        def acquire(self):
            return Ctx()

    async def noop(*_a, **_k):
        return None

    async def pool():
        return Pool()

    from pydantic_agents.core import vector_db as vdb

    manager = VectorDBManager()
    with (
        patch.object(vdb, "_require_vector_deps"),
        patch.object(vdb, "_ensure_table", noop),
        patch.object(vdb, "_to_embedding", return_value=[0.1]),
        patch.object(vdb, "_get_pool", pool),
    ):
        await manager._search("activity_logs", "q", 5, None, {"agent_name": "a", "run_id": "r"})
    sql, args = calls[0]
    assert "metadata->>'agent_name' = $3" in sql
    assert "metadata->>'run_id' = $4" in sql
    assert args[2:] == ("a", "r")


@pytest.mark.parametrize("key", ["a'; DROP TABLE x;--", "a b", "1abc", ""])
async def test_vector_search_rejects_unsafe_filter_keys(key):
    manager = VectorDBManager()
    with pytest.raises(ValueError):
        await manager._search("activity_logs", "q", 5, None, {key: "v"})


# ── non-technical prompts ──────────────────────────────────────────────────

NON_TECH = [
    "accountant",
    "asymmetric_reviewer",
    "crypto_oracle",
    "current_events",
    "holiday_planner",
    "market_shark",
    "price_analyst",
    "region_analyzer",
    "weather",
]
INVESTING = ["crypto_oracle", "market_shark", "price_analyst"]


@pytest.fixture(scope="module")
def prompts():
    register_all_agents()
    return {n: "\n".join(str(p) for p in registry.get(n)._system_prompts) for n in NON_TECH}


@pytest.mark.parametrize("name", NON_TECH)
def test_prompts_require_dates_sources_and_clarification(prompts, name):
    text = prompts[name]
    assert "get_current_datetime" in text
    assert "tool result" in text
    assert "clarifying question" in text


@pytest.mark.parametrize("name", INVESTING)
def test_investment_agents_state_uncertainty_and_disclaimer(prompts, name):
    text = prompts[name]
    assert "not financial advice" in text
    assert "never as certainties" in text
    assert "definitive" not in text.lower()


def test_weather_agent_is_told_about_its_forecast_tool(prompts):
    assert "get_weather_forecast" in prompts["weather"]
    assert "get_weather_forecast" in registry.get("weather")._function_toolset.tools


def test_reviewer_is_framed_around_consumer_rights(prompts):
    text = prompts["asymmetric_reviewer"].lower()
    assert "loophole" not in text
    assert "exploit" not in text
    assert "consumer" in text


async def test_query_history_can_opt_in_to_untagged_legacy_rows(run_ctx):
    legacy_row = json.dumps({"pass_num": 1, "agent_name": "old", "action": "A", "summary": "before tagging"})
    calls = []

    async def async_search(collection, query, n_results=3, project=None, filters=None):  # noqa: ARG001
        calls.append(filters)
        return [] if "agent_name" in (filters or {}) and filters["agent_name"] else [legacy_row]

    with patch.object(memory.vector_db, "async_search", async_search):
        without = await memory.query_history(MagicMock(deps=None), "q")
        with_legacy = await memory.query_history(MagicMock(deps=None), "q", include_legacy=True)

    assert without == "No relevant history found."
    assert "before tagging" in with_legacy
    assert {"type": "activity", "agent_name": None} in calls


async def test_vector_search_matches_untagged_rows_with_is_null():
    captured = []

    class Conn:
        async def fetch(self, sql, *args):
            captured.append((sql, args))
            return []

    class Ctx:
        async def __aenter__(self):
            return Conn()

        async def __aexit__(self, *_):
            return False

    class Pool:
        def acquire(self):
            return Ctx()

    async def noop(*_a, **_k):
        return None

    async def pool():
        return Pool()

    from pydantic_agents.core import vector_db as vdb

    manager = VectorDBManager()
    with (
        patch.object(vdb, "_require_vector_deps"),
        patch.object(vdb, "_ensure_table", noop),
        patch.object(vdb, "_to_embedding", return_value=[0.1]),
        patch.object(vdb, "_get_pool", pool),
    ):
        await manager._search("activity_logs", "q", 5, None, {"type": "activity", "agent_name": None})
    sql, args = captured[0]
    assert "metadata->>'type' = $3" in sql
    assert "metadata->>'agent_name' IS NULL" in sql
    assert args[2:] == ("activity",)


async def test_mcp_run_agent_by_name_returns_parseable_json_for_structured_output():
    from pydantic_agents.mcp_server import run_agent_by_name

    artifact = Artifact(agent_name="backend", files={"main.go": "package main"})
    agent = AsyncMock()
    agent.run.return_value = MagicMock(output=artifact)
    with (
        patch("pydantic_agents.mcp_server.registry") as reg,
        patch("pydantic_agents.mcp_server.retrieve_context", AsyncMock(return_value="")),
    ):
        reg.is_registered.return_value = True
        reg.run = AsyncMock(return_value=agent.run.return_value)
        reg._agents = {"backend": (None, None, lambda: None, None)}
        result = await run_agent_by_name("backend", "build it")

    parsed = json.loads(result)
    assert parsed["agent_name"] == "backend"
    assert parsed["files"] == {"main.go": "package main"}


def test_mcp_agent_tools_describe_the_result_format():
    from pydantic_agents.mcp_server import RESULT_FORMAT_NOTE

    assert "JSON" in RESULT_FORMAT_NOTE
