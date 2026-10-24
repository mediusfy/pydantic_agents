"""Typed results for the investing agents: the prompts' rules are enforced by validation."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.models.analysis import (
    DISCLAIMER,
    Forecast,
    MarketAnalysis,
    PurchaseAdvice,
    Scenario,
)
from pydantic_agents.core.registry import AgentRegistry, registry
from pydantic_agents.core.utils import serialize_output

SOURCE = {"tool": "get_historical_prices", "reference": "BTC-USD 1mo"}


def _scenario(name="base", low=60_000.0, high=70_000.0, probability=0.5):
    return {"name": name, "low": low, "high": high, "probability": probability, "drivers": ["ETF inflows"]}


def _forecast(**overrides):
    data = {
        "as_of": "2026-10-24",
        "summary": "range-bound",
        "asset": "BTC",
        "horizon": "30 days",
        "scenarios": [
            _scenario("base", probability=0.5),
            _scenario("bull", 70_000, 80_000, 0.3),
            _scenario("bear", 50_000, 60_000, 0.2),
        ],
        "sources": [SOURCE],
    }
    data.update(overrides)
    return data


def _market(**overrides):
    data = {
        "as_of": "2026-10-24",
        "summary": "s",
        "asset": "AAPL",
        "thesis": "undervalued vs peers",
        "metrics": [{"name": "P/E", "value": "28.1", "source": "get_stock_ticker_data"}],
        "bull_case": ["services growth"],
        "bear_case": ["China exposure"],
        "sources": [SOURCE],
    }
    data.update(overrides)
    return data


def _purchase(**overrides):
    both = [
        {"action": "buy_now", "conditions": ["price near 52w low"], "rationale": "r"},
        {"action": "wait", "conditions": ["sale season"], "expected_price_change_pct": -8.0, "rationale": "r"},
    ]
    data = {
        "as_of": "2026-10-24",
        "summary": "s",
        "item": "laptop",
        "leaning": "wait",
        "trend": "-3%/mo",
        "scenarios": both,
        "sources": [SOURCE],
    }
    data.update(overrides)
    return data


# ── rules enforced by the models ──────────────────────────────────────────


@pytest.mark.parametrize(
    "probs",
    [(0.5, 0.3, 0.2), (0.49, 0.3, 0.2), (0.33, 0.33, 0.33), (0.5, 0.3, 0.18), (0.52, 0.3, 0.2)],
    ids=["exact", "-0.01", "rounded thirds", "exactly -0.02", "exactly +0.02"],
)
def test_forecast_probabilities_within_tolerance_are_accepted(probs):
    scenarios = [
        _scenario(n, probability=p) for n, p in zip(("base", "bull", "bear")[: len(probs)], probs, strict=True)
    ]
    assert Forecast(**_forecast(scenarios=scenarios))


@pytest.mark.parametrize(
    "probs",
    [(0.5, 0.3, 0.17), (0.53, 0.3, 0.2), (0.5, 0.45), (0.9, 0.9, 0.9)],
    ids=["just under 0.98", "just over 1.02", "0.95 total", "2.7 total"],
)
def test_forecast_probabilities_outside_tolerance_are_rejected(probs):
    scenarios = [
        _scenario(n, probability=p) for n, p in zip(("base", "bull", "bear")[: len(probs)], probs, strict=True)
    ]
    kwargs = _forecast(scenarios=scenarios)
    with pytest.raises(ValidationError, match="add up"):
        Forecast(**kwargs)


def test_scenario_range_must_be_ordered():
    kwargs = _scenario(low=80_000, high=60_000)
    with pytest.raises(ValidationError, match="above high"):
        Scenario(**kwargs)


@pytest.mark.parametrize("model, data", [(Forecast, _forecast), (MarketAnalysis, _market), (PurchaseAdvice, _purchase)])
def test_every_result_must_cite_a_source(model, data):
    kwargs = data(sources=[])
    with pytest.raises(ValidationError, match="sources"):
        model(**kwargs)


@pytest.mark.parametrize("model, data", [(Forecast, _forecast), (MarketAnalysis, _market), (PurchaseAdvice, _purchase)])
def test_disclaimer_is_set_by_code_not_the_model(model, data):
    assert model(**data()).disclaimer == DISCLAIMER
    assert model(**data(disclaimer="trust me, this is advice")).disclaimer == DISCLAIMER


def test_forecast_needs_at_least_two_scenarios():
    kwargs = _forecast(scenarios=[_scenario(probability=1.0)])
    with pytest.raises(ValidationError):
        Forecast(**kwargs)


def test_previous_forecast_is_optional_but_typed():
    assert Forecast(**_forecast()).previous_forecast is None
    prev = {"prediction": "BTC 60-70k", "actual": "BTC 64k", "error_pct": -2.1, "lesson": "ETF flows dominated"}
    assert Forecast(**_forecast(previous_forecast=prev)).previous_forecast.error_pct == -2.1


@pytest.mark.parametrize("field", ["metrics", "bull_case", "bear_case"])
def test_market_analysis_needs_metrics_and_both_cases(field):
    kwargs = _market(**{field: []})
    with pytest.raises(ValidationError):
        MarketAnalysis(**kwargs)


@pytest.mark.parametrize("actions", [["buy_now", "buy_now"], ["wait", "wait"]])
def test_purchase_advice_must_cover_both_choices(actions):
    scenarios = [{"action": a, "conditions": ["c"], "rationale": "r"} for a in actions]
    kwargs = _purchase(scenarios=scenarios)
    with pytest.raises(ValidationError, match="both buy_now and wait"):
        PurchaseAdvice(**kwargs)


def test_purchase_leaning_allows_unclear_and_rejects_other_values():
    assert PurchaseAdvice(**_purchase(leaning="unclear")).leaning == "unclear"
    kwargs = _purchase(leaning="definitely buy")
    with pytest.raises(ValidationError):
        PurchaseAdvice(**kwargs)


@pytest.mark.parametrize("price", [0.0, -5.0])
def test_current_price_must_be_positive_when_given(price):
    kwargs = _purchase(current_price=price)
    with pytest.raises(ValidationError):
        PurchaseAdvice(**kwargs)


def test_current_price_may_be_absent():
    assert PurchaseAdvice(**_purchase()).current_price is None
    assert PurchaseAdvice(**_purchase(current_price=1299.99)).current_price == 1299.99


# ── agents return the typed models, and a rule violation makes the model retry ──

AGENTS = [
    ("crypto_oracle", Forecast, _forecast),
    ("market_shark", MarketAnalysis, _market),
    ("price_analyst", PurchaseAdvice, _purchase),
]


@pytest.fixture(scope="module", autouse=True)
def _agents():
    register_all_agents()


@pytest.mark.parametrize("name, model, data", AGENTS, ids=[a[0] for a in AGENTS])
async def test_agent_returns_the_typed_result_as_json(name, model, data):
    agent = registry.get(name)
    with agent.override(model=TestModel(call_tools=[], custom_output_args=data())):
        result = await agent.run("analyse", deps=MagicMock(agent_name=name))

    assert isinstance(result.output, model)
    parsed = json.loads(serialize_output(result))
    assert parsed["disclaimer"] == DISCLAIMER
    assert parsed["sources"][0]["tool"] == "get_historical_prices"


async def test_invalid_forecast_is_sent_back_to_the_model_to_fix():
    bad = _forecast(scenarios=[_scenario("base", probability=0.5), _scenario("bear", 50_000, 60_000, 0.1)])
    good = _forecast()
    attempts = []

    async def model(messages, info: AgentInfo):
        attempts.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, bad if len(attempts) == 1 else good)])

    agent = registry.get("crypto_oracle")
    with agent.override(model=FunctionModel(model)):
        result = await agent.run("analyse BTC", deps=MagicMock(agent_name="crypto_oracle"))

    assert len(attempts) == 2  # the first answer broke the probability rule and was retried
    assert result.output.scenarios[0].probability == 0.5
    assert abs(sum(s.probability for s in result.output.scenarios) - 1.0) <= 0.05


async def test_registry_query_returns_the_json_for_these_agents():
    reg = AgentRegistry()
    agent = MagicMock()
    forecast = Forecast(**_forecast())

    async def run(*_a, **_k):
        return MagicMock(output=forecast)

    agent.run = run
    reg.register("crypto_oracle", agent, ["finance.crypto"])
    parsed = json.loads(await reg.query("finance.crypto", "BTC?"))
    assert parsed["asset"] == "BTC"
    assert len(parsed["scenarios"]) == 3


@pytest.mark.parametrize("name", [a[0] for a in AGENTS])
def test_prompts_describe_the_typed_result_and_keep_the_guardrails(name):
    text = "\n".join(str(p) for p in registry.get(name)._system_prompts)
    assert "not financial advice" in text
    assert "never as certainties" in text
    assert "`as_of`" in text
    assert "## Output" in text


def test_the_mcp_tool_descriptions_name_the_result_types():
    from pydantic_agents.mcp_server import AGENT_DOCSTRINGS

    assert "Forecast" in AGENT_DOCSTRINGS["crypto_oracle"]
    assert "MarketAnalysis" in AGENT_DOCSTRINGS["market_shark"]
    assert "PurchaseAdvice" in AGENT_DOCSTRINGS["price_analyst"]


async def test_the_mcp_server_returns_parseable_json_for_a_typed_investing_result():
    from unittest.mock import AsyncMock, patch

    from pydantic_agents.mcp_server import run_agent_by_name

    agent = AsyncMock()
    agent.run.return_value = MagicMock(output=Forecast(**_forecast()))
    with (
        patch("pydantic_agents.mcp_server.registry") as reg,
        patch("pydantic_agents.mcp_server.retrieve_context", return_value=""),
    ):
        reg.get.return_value = agent
        reg._agents = {"crypto_oracle": (None, None, lambda: None, None)}
        result = await run_agent_by_name("crypto_oracle", "BTC outlook?")

    parsed = json.loads(result)
    assert parsed["asset"] == "BTC"
    assert parsed["disclaimer"] == DISCLAIMER
