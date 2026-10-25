"""Typed results for holiday_planner (Itinerary) and accountant (CostBreakdown): the prompts' rules are enforced."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.models.personal import (
    ACCOUNTING_DISCLAIMER,
    CostBreakdown,
    CostLine,
    Itinerary,
    ItineraryItem,
)
from pydantic_agents.core.registry import registry
from pydantic_agents.core.utils import serialize_output

SOURCE = {"tool": "get_weather_forecast", "reference": "Lisbon 2026-11-02..04"}


def _item(**overrides):
    data = {"time": "09:30", "activity": "Belem Tower", "status": "confirmed", "cost": 12.0}
    data.update(overrides)
    return data


def _day(day=1, date="2026-11-02", **overrides):
    data = {"day": day, "date": date, "items": [_item()]}
    data.update(overrides)
    return data


def _itinerary(**overrides):
    data = {
        "destination": "Lisbon",
        "start_date": "2026-11-02",
        "end_date": "2026-11-04",
        "days": [_day(1, "2026-11-02"), _day(2, "2026-11-03"), _day(3, "2026-11-04")],
        "sources": [SOURCE],
    }
    data.update(overrides)
    return data


def _line(**overrides):
    data = {"name": "Hotel", "kind": "fixed", "low": 360.0, "high": 360.0, "basis": "3 x 120", "source": "user"}
    data.update(overrides)
    return data


def _breakdown(**overrides):
    data = {
        "as_of": "2026-10-24",
        "currency": "EUR",
        "lines": [_line(), _line(name="Food", kind="estimate", low=150.0, high=240.0, basis="3 x 50-80")],
        "sources": [{"tool": "user"}],
    }
    data.update(overrides)
    return data


# ── Itinerary ─────────────────────────────────────────────────────────────


def test_a_complete_itinerary_is_accepted_and_totals_are_derived():
    plan = Itinerary(**_itinerary(budget=100.0, estimated_total=1.0, within_budget=False))
    assert plan.estimated_total == 36.0  # supplied values are ignored
    assert plan.within_budget is True


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(36.0, True), (35.99, False), (36.004, True), (35.996, True), (35.994, False), (None, None)],
    ids=["exactly at budget", "over", "rounds up to 36.00", "rounds up (35.996)", "rounds down (35.994)", "no budget"],
)
def test_within_budget_follows_the_costs(budget, expected):
    assert Itinerary(**_itinerary(budget=budget)).within_budget is expected


def test_costs_that_are_unknown_leave_the_total_unset():
    days = [_day(items=[_item(cost=None)])]
    plan = Itinerary(**_itinerary(days=days, end_date="2026-11-02", budget=50.0))
    assert plan.estimated_total is None
    assert plan.within_budget is None


def test_an_itinerary_may_be_only_questions():
    plan = Itinerary(destination="Lisbon", open_questions=["Which dates?", "How many travellers?"])
    assert plan.days == []


def test_an_itinerary_without_days_or_questions_is_rejected():
    with pytest.raises(ValidationError, match="open_questions"):
        Itinerary(destination="Lisbon")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sources": []}, "cite at least one source"),
        ({"start_date": "2026-11-09"}, "after end_date"),
        ({"start_date": "next week"}, "ISO 8601"),
        ({"days": [_day(1, "2026-11-02"), _day(3, "2026-11-03")]}, "numbered 1"),
        ({"days": [_day(1, "2026-11-03"), _day(2, "2026-11-02")]}, "unique and in order"),
        ({"days": [_day(1, "2026-11-02"), _day(2, "2026-11-02")]}, "unique and in order"),
        ({"days": [_day(1, "2026-11-01")]}, "fall outside"),
        ({"days": [_day(1, "2026-11-02"), _day(2, "2026-11-05")]}, "fall outside"),
        ({"days": [_day(1, "02/11/2026")]}, "ISO 8601"),
        ({"start_date": "2026-02-31", "end_date": "2026-03-02"}, "ISO 8601"),
    ],
    ids=[
        "no sources",
        "reversed range",
        "bad start date",
        "gap in day numbers",
        "dates out of order",
        "duplicate dates",
        "before start",
        "after end",
        "bad day date",
        "impossible date",
    ],
)
def test_itinerary_rules(overrides, message):
    kwargs = _itinerary(**overrides)
    with pytest.raises(ValidationError, match=message):
        Itinerary(**kwargs)


def test_an_unverified_item_must_say_where_to_confirm():
    kwargs = _item(status="to_confirm")
    with pytest.raises(ValidationError, match="where to confirm"):
        ItineraryItem(**kwargs)
    assert ItineraryItem(**_item(status="to_confirm", confirm_where="the museum's website"))


@pytest.mark.parametrize(
    "field, value", [("duration_minutes", 0), ("cost", -1.0), ("travel_minutes_from_previous", -5)]
)
def test_item_numbers_must_be_sane(field, value):
    kwargs = _item(**{field: value})
    with pytest.raises(ValidationError):
        ItineraryItem(**kwargs)


def test_a_day_needs_at_least_one_item():
    itinerary = _itinerary(days=[_day(items=[])])
    with pytest.raises(ValidationError):
        Itinerary(**itinerary)


# ── CostBreakdown ─────────────────────────────────────────────────────────


def test_totals_are_added_up_in_code_and_the_disclaimer_is_fixed():
    result = CostBreakdown(**_breakdown(total_low=1.0, total_high=2.0, disclaimer="this is tax advice"))
    assert (result.total_low, result.total_high) == (510.0, 600.0)
    assert result.disclaimer == ACCOUNTING_DISCLAIMER


def test_totals_do_not_accumulate_float_error():
    lines = [_line(name=f"l{i}", low=0.1, high=0.1) for i in range(3)]
    assert CostBreakdown(**_breakdown(lines=lines)).total_low == 0.3


def test_a_breakdown_may_be_only_questions():
    result = CostBreakdown(as_of="2026-10-24", currency="EUR", open_questions=["Which currency is the quote in?"])
    assert result.lines == []
    assert result.total_low is None


def test_a_breakdown_without_lines_or_questions_is_rejected():
    with pytest.raises(ValidationError, match="open_questions"):
        CostBreakdown(as_of="2026-10-24", currency="EUR")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sources": []}, "cite at least one source"),
        ({"as_of": "today"}, "ISO 8601"),
        ({"lines": [_line(kind="fixed", low=100.0, high=120.0)]}, "fixed but has a range"),
        ({"lines": [_line(kind="estimate", low=120.0, high=100.0)]}, "above high"),
        ({"lines": [_line(low=-1.0, high=-1.0)]}, "greater than or equal to 0"),
    ],
    ids=["no sources", "bad as_of", "fixed with a range", "reversed range", "negative cost"],
)
def test_breakdown_rules(overrides, message):
    kwargs = _breakdown(**overrides)
    with pytest.raises(ValidationError, match=message):
        CostBreakdown(**kwargs)


def test_a_fixed_cost_must_be_exact_to_the_cent():
    kwargs = _line(low=100.0, high=100.01)
    with pytest.raises(ValidationError, match="fixed but has a range"):
        CostLine(**kwargs)


def test_an_estimate_may_have_equal_bounds():
    assert CostLine(**_line(kind="estimate"))


@pytest.mark.parametrize(
    ("rate", "as_of", "message"),
    [(0.0, "2026-10-24", "greater than 0"), (1.08, "yesterday", "ISO 8601")],
    ids=["zero rate", "undated rate"],
)
def test_exchange_rates_need_a_positive_rate_and_a_date(rate, as_of, message):
    rates = [{"from_currency": "EUR", "to_currency": "USD", "rate": rate, "as_of": as_of, "source": "ecb"}]
    kwargs = _breakdown(exchange_rates=rates)
    with pytest.raises(ValidationError, match=message):
        CostBreakdown(**kwargs)


def test_a_dated_exchange_rate_is_kept():
    rates = [{"from_currency": "EUR", "to_currency": "USD", "rate": 1.08, "as_of": "2026-10-24", "source": "ecb"}]
    assert CostBreakdown(**_breakdown(exchange_rates=rates)).exchange_rates[0].rate == 1.08


# ── agents ────────────────────────────────────────────────────────────────

AGENTS = [
    ("holiday_planner", Itinerary, _itinerary),
    ("accountant", CostBreakdown, _breakdown),
]


@pytest.fixture(scope="module", autouse=True)
def _agents():
    register_all_agents()


@pytest.mark.parametrize(("name", "model", "data"), AGENTS, ids=[a[0] for a in AGENTS])
async def test_agent_returns_the_typed_result_as_json(name, model, data):
    agent = registry.get(name)
    with agent.override(model=TestModel(call_tools=[], custom_output_args=data())):
        result = await agent.run("go", deps=MagicMock(agent_name=name))

    assert isinstance(result.output, model)
    parsed = json.loads(serialize_output(result))
    assert parsed["sources"]


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("holiday_planner", {"destination": "Lisbon", "open_questions": ["Which dates?"]}),
        ("accountant", {"as_of": "2026-10-24", "currency": "EUR", "open_questions": ["Which currency?"]}),
    ],
)
async def test_agents_can_answer_with_questions_only(name, args):
    agent = registry.get(name)
    with agent.override(model=TestModel(call_tools=[], custom_output_args=args)):
        result = await agent.run("plan something", deps=MagicMock(agent_name=name))
    assert json.loads(serialize_output(result))["open_questions"]


async def test_an_invalid_itinerary_is_sent_back_to_the_model_to_fix():
    bad = _itinerary(days=[_day(1, "2026-11-02"), _day(3, "2026-11-03")])
    good = _itinerary()
    attempts = []

    async def model(messages, info: AgentInfo):
        attempts.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, bad if len(attempts) == 1 else good)])

    agent = registry.get("holiday_planner")
    with agent.override(model=FunctionModel(model)):
        result = await agent.run("3 days in Lisbon", deps=MagicMock(agent_name="holiday_planner"))

    assert len(attempts) == 2  # the first answer skipped a day number and was retried
    assert [d.day for d in result.output.days] == [1, 2, 3]


@pytest.mark.parametrize("name", [a[0] for a in AGENTS])
def test_prompts_describe_the_typed_result_and_keep_the_ground_rules(name):
    text = "\n".join(str(p) for p in registry.get(name)._system_prompts)
    assert "## Output" in text
    assert "open_questions" in text
    assert "## Ground rules" in text


def test_the_accountant_prompt_still_refuses_tax_and_legal_advice():
    text = "\n".join(str(p) for p in registry.get("accountant")._system_prompts)
    assert "not a tax or legal adviser" in text
    assert "`disclaimer` is set for you" in text


def test_the_mcp_tool_descriptions_name_the_result_types():
    from pydantic_agents.mcp_server import AGENT_DOCSTRINGS

    assert "Itinerary" in AGENT_DOCSTRINGS["holiday_planner"]
    assert "CostBreakdown" in AGENT_DOCSTRINGS["accountant"]


@pytest.mark.parametrize(
    ("name", "model", "data"),
    [("holiday_planner", Itinerary, _itinerary), ("accountant", CostBreakdown, _breakdown)],
)
async def test_the_mcp_server_returns_the_model_json(name, model, data):
    from unittest.mock import AsyncMock, patch

    from pydantic_agents.mcp_server import run_agent_by_name

    agent = AsyncMock()
    agent.run.return_value = MagicMock(output=model(**data()))
    with (
        patch("pydantic_agents.mcp_server.registry") as reg,
        patch("pydantic_agents.mcp_server.retrieve_context", return_value=""),
    ):
        reg.get.return_value = agent
        reg._agents = {name: (None, None, lambda: None, None)}
        result = await run_agent_by_name(name, "go")

    assert model.model_validate_json(result) == model(**data())
