import json
from contextlib import asynccontextmanager

import httpx
import pytest

from pydantic_agents.core.models.analysis import Forecast
from scripts.post_deploy_smoke import (
    CHECKS,
    check_forecast,
    check_review,
    check_security,
    describe_error,
    find_tool,
    main,
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
        (["x-run_review_v2"], "run_review", None),
        ([], "run_review", None),
    ],
    ids=["gateway prefix", "bare", "underscore prefix", "not a suffix match", "versioned name", "empty"],
)
def test_find_tool(names, suffix, expected):
    assert find_tool(names, suffix) == expected


def test_find_tool_refuses_an_ambiguous_match():
    with pytest.raises(ValueError, match="ambiguous"):
        find_tool(["a-run_review", "b-run_review"], "run_review")


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


async def test_an_ambiguous_tool_name_is_a_failure_not_a_crash():
    async def call(tool, request):
        raise AssertionError("must not be called")

    failures = dict(await run_checks(["a-run_review", "b-run_review"], call, CHECKS[:1]))
    assert "ambiguous" in failures["run_review"]


# ── the client wrapper, against the in-process MCP server ─────────────────


async def test_the_client_wrapper_lists_tools_and_returns_their_text():
    from unittest.mock import AsyncMock, patch

    from mcp.shared.memory import create_connected_server_and_client_session

    from pydantic_agents.mcp_server import mcp
    from scripts.post_deploy_smoke import _caller, _tool_names

    with patch("pydantic_agents.mcp_server.run_agent_by_name", new=AsyncMock(return_value=REVIEW)):
        async with create_connected_server_and_client_session(mcp._mcp_server) as session:
            names = await _tool_names(session)
            assert find_tool(names, "run_review") == "run_review"
            assert await _caller(session)("run_review", "review this") == REVIEW
            assert await run_checks(names, _caller(session), CHECKS[:1]) == []


async def test_a_tool_error_from_the_server_is_a_failure():
    from unittest.mock import AsyncMock, patch

    from mcp.shared.memory import create_connected_server_and_client_session

    from pydantic_agents.mcp_server import mcp
    from scripts.post_deploy_smoke import _caller, _tool_names

    with patch("pydantic_agents.mcp_server.run_agent_by_name", new=AsyncMock(side_effect=RuntimeError("pool closed"))):
        async with create_connected_server_and_client_session(mcp._mcp_server) as session:
            failures = dict(await run_checks(await _tool_names(session), _caller(session), CHECKS[:1]))

    assert "pool closed" in failures["run_review"]


async def test_an_empty_tool_result_is_reported_as_such():
    from unittest.mock import AsyncMock, patch

    from mcp.shared.memory import create_connected_server_and_client_session

    from pydantic_agents.mcp_server import mcp
    from scripts.post_deploy_smoke import _caller, _tool_names

    # mcp._mcp_server is private SDK API; this in-process connection is the only way to test the wrapper offline.
    with patch("pydantic_agents.mcp_server.run_agent_by_name", new=AsyncMock(return_value="")):
        async with create_connected_server_and_client_session(mcp._mcp_server) as session:
            failures = dict(await run_checks(await _tool_names(session), _caller(session), CHECKS[:1]))

    assert "no text content" in failures["run_review"]


def _status_error(code: int, reason: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://gateway.example/mcp")
    return httpx.HTTPStatusError(
        "boom",
        request=request,
        response=httpx.Response(code, request=request, extensions={"reason_phrase": reason.encode()}),
    )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_status_error(503, "Service Unavailable"), "the endpoint answered HTTP 503 (Service Unavailable)"),
        (
            ExceptionGroup("outer", [ExceptionGroup("inner", [_status_error(502, "Bad Gateway")])]),
            "the endpoint answered HTTP 502 (Bad Gateway)",
        ),
        (ConnectionError("refused"), "ConnectionError: refused"),
        (ExceptionGroup("outer", [ValueError("x" * 400)]), "ValueError: " + "x" * 300),
    ],
    ids=["status", "nested-group", "plain", "long-message-truncated"],
)
def test_describe_error_gives_one_line_without_the_group_noise(exc, expected):
    assert describe_error(exc) == expected


async def test_main_reports_an_unreachable_endpoint_in_one_line(monkeypatch, capsys):
    @asynccontextmanager
    async def refuse(*args, **kwargs):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [_status_error(503, "Service Unavailable")])
        yield

    monkeypatch.setattr("mcp.client.streamable_http.streamablehttp_client", refuse)
    monkeypatch.setenv("MCP_URL", "https://gateway.example/mcp")

    assert await main() == 1
    err = capsys.readouterr().err
    assert (
        err.strip()
        == "FAIL could not use https://gateway.example/mcp: the endpoint answered HTTP 503 (Service Unavailable)"
    )
