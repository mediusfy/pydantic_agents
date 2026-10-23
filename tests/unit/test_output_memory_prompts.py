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

    with (
        patch.object(vdb, "_require_vector_deps"),
        patch.object(vdb, "_ensure_table", noop),
        patch.object(vdb, "_to_embedding", return_value=[0.1]),
        patch.object(vdb, "_get_pool", pool),
    ):
        await VectorDBManager()._search("activity_logs", "q", 5, None, {"agent_name": "a", "run_id": "r"})
    sql, args = calls[0]
    assert "metadata->>'agent_name' = $3" in sql and "metadata->>'run_id' = $4" in sql
    assert args[2:] == ("a", "r")


@pytest.mark.parametrize("key", ["a'; DROP TABLE x;--", "a b", "1abc", ""])
async def test_vector_search_rejects_unsafe_filter_keys(key):
    with pytest.raises(ValueError):
        await VectorDBManager()._search("activity_logs", "q", 5, None, {key: "v"})


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
    assert "get_current_datetime" in text and "tool result" in text and "clarifying question" in text


@pytest.mark.parametrize("name", INVESTING)
def test_investment_agents_state_uncertainty_and_disclaimer(prompts, name):
    text = prompts[name]
    assert "not financial advice" in text and "never as certainties" in text
    assert "definitive" not in text.lower()


def test_weather_agent_is_told_about_its_forecast_tool(prompts):
    assert "get_weather_forecast" in prompts["weather"]
    assert "get_weather_forecast" in registry.get("weather")._function_toolset.tools


def test_reviewer_is_framed_around_consumer_rights(prompts):
    text = prompts["asymmetric_reviewer"].lower()
    assert "loophole" not in text and "exploit" not in text and "consumer" in text
