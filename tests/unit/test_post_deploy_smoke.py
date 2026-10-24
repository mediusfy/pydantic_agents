import json

import pytest

from pydantic_agents.core.models.analysis import Forecast
from scripts.post_deploy_smoke import (
    CHECKS,
    check_forecast,
    check_review,
    check_security,
    find_tool,
    run_checks,
)

REVIEW = json.dumps({"approved": True, "feedback": [], "security_score": 90, "risk_level": "LOW"})
SECURITY = json.dumps({"passed": True, "score": 95, "feedback": []})
FORECAST = Forecast(
    as_of="2026-10-24",
    summary="range-bound",
    asset="BTC",
    horizon="30 days",
    scenarios=[
        {"name": "base", "low": 60_000, "high": 70_000, "probability": 0.6},
        {"name": "bear", "low": 50_000, "high": 60_000, "probability": 0.4},
    ],
    sources=[{"tool": "get_historical_prices", "reference": "BTC-USD 1mo"}],
).model_dump_json()
NAMES = ["pydantic_agents-run_review", "pydantic_agents-run_security", "pydantic_agents-run_crypto_oracle", "other"]


@pytest.mark.parametrize(
    ("names", "suffix", "expected"),
    [
        (NAMES, "run_review", "pydantic_agents-run_review"),
        (["run_review"], "run_review", "run_review"),
        (["pydantic_agents_run_review"], "run_review", "pydantic_agents_run_review"),
        (["rerun_review"], "run_review", None),
        ([], "run_review", None),
    ],
    ids=["gateway prefix", "bare", "underscore prefix", "not a suffix match", "empty"],
)
def test_find_tool(names, suffix, expected):
    assert find_tool(names, suffix) == expected


@pytest.mark.parametrize(
    ("check", "text"),
    [(check_review, REVIEW), (check_security, SECURITY), (check_forecast, FORECAST)],
    ids=["review", "security", "forecast"],
)
def test_checks_accept_valid_output(check, text):
    assert check(text)


@pytest.mark.parametrize(
    ("check", "text"),
    [
        (check_review, "Error: model unavailable"),
        (check_review, json.dumps({"approved": True})),
        (check_security, "not json"),
        (check_forecast, "plain prose forecast"),
        (check_forecast, json.dumps({"asset": "BTC"})),
    ],
    ids=["review prose", "review missing fields", "security prose", "forecast prose", "forecast partial"],
)
def test_checks_reject_unparseable_output(check, text):
    with pytest.raises(ValueError):
        check(text)


def test_security_check_uses_the_score_not_the_reported_verdict():
    # SecurityReport recomputes `passed` from the score, so a contradictory payload is reported as failed.
    assert "passed=False" in check_security(json.dumps({"passed": True, "score": 10}))


async def test_run_checks_passes_when_every_output_parses():
    outputs = {"run_review": REVIEW, "run_security": SECURITY, "run_crypto_oracle": FORECAST}

    async def call(tool, request):
        return next(v for k, v in outputs.items() if tool.endswith(k))

    assert await run_checks(NAMES, call) == []


async def test_run_checks_reports_each_failure_and_missing_tool():
    async def call(tool, request):
        if tool.endswith("run_review"):
            return "Error: Event loop is closed"
        raise ConnectionError("gateway down")

    failures = dict(await run_checks(["pydantic_agents-run_review", "x-run_security"], call))
    assert set(failures) == {c.tool for c in CHECKS}
    assert failures["run_crypto_oracle"] == "tool not exposed by the endpoint"
    assert failures["run_security"].startswith("ConnectionError")
    assert "ValidationError" in failures["run_review"]
