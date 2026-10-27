"""Pins the critic, api_contract, observability_engineer, finops and release_manager agents.

Covers their registration (capabilities, output type, tool grants), the derived fields on their typed outputs, the
LiteLLM spend tool's request shape and failure mode, and the release manager's PR-body contract.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.agents.release_manager import ATTRIBUTION_LINE, WORKSPACE_MARKER_LINE
from pydantic_agents.core.models.core import Artifact, ContractReport, CostReport, CriticReport
from pydantic_agents.core.registry import registry
from pydantic_agents.core.tools import finops_tools


@pytest.fixture(scope="module", autouse=True)
def _agents():
    register_all_agents()


def _toolset(name: str) -> set[str]:
    return set(registry.get(name)._function_toolset.tools)


def _capabilities(name: str) -> list[str]:
    return dict(registry.items())[name][1]


# ── registration ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "capabilities", "output_type"),
    [
        ("critic", ["critic", "verification"], CriticReport),
        ("api_contract", ["api_contract", "contract_drift"], ContractReport),
        ("observability_engineer", ["observability", "alerting"], Artifact),
        ("finops", ["finops", "cost"], CostReport),
        ("release_manager", ["release", "changelog"], Artifact),
    ],
)
def test_agent_is_registered_with_capabilities_and_output_type(name, capabilities, output_type):
    assert _capabilities(name) == capabilities
    assert registry.get(name).output_type is output_type
    for capability in capabilities:
        assert registry.owners_of(capability) == [name]


@pytest.mark.parametrize(
    ("name", "present", "absent"),
    [
        ("critic", {"read_file", "search_codebase"}, {"delegate_to_agent", "write_file", "fetch_url"}),
        ("api_contract", {"read_file", "validate_protobuf", "validate_json"}, {"delegate_to_agent", "write_file"}),
        ("observability_engineer", {"read_file", "validate_yaml", "delegate_to_agent"}, {"write_file"}),
        ("finops", {"litellm_spend_report", "fetch_url"}, {"read_file", "delegate_to_agent"}),
        ("release_manager", {"read_file", "update_documentation"}, {"delegate_to_agent", "write_file"}),
    ],
)
def test_agent_tool_grants(name, present, absent):
    tools = _toolset(name)
    assert present <= tools, f"{name} lacks {sorted(present - tools)}"
    assert not absent & tools, f"{name} must not have {sorted(absent & tools)}"


# ── derived output fields ─────────────────────────────────────────────────────


def _claims(*supported: bool) -> list[dict[str, Any]]:
    return [
        {"claim": f"claim {i}", "supported": ok, "evidence": "log line" if ok else ""} for i, ok in enumerate(supported)
    ]


@pytest.mark.parametrize(
    ("payload", "unsupported_count", "passed"),
    [
        ({"claims": _claims(True, True), "unsupported_count": 5, "passed": False}, 0, True),
        ({"claims": _claims(True, False, False), "unsupported_count": 0, "passed": True}, 2, False),
        ({"claims": [], "passed": False}, 0, True),
    ],
    ids=["all supported, lies ignored", "two unsupported, lies ignored", "no claims passes"],
)
def test_critic_report_derives_count_and_verdict(payload, unsupported_count, passed):
    report = CriticReport.model_validate(payload)
    assert (report.unsupported_count, report.passed) == (unsupported_count, passed)


@pytest.mark.parametrize(
    ("kinds", "supplied", "passed"),
    [
        ([], False, True),
        (["missing_endpoint", "extra_endpoint"], False, True),
        (["missing_endpoint", "type_mismatch"], True, False),
        (["breaking_change"], True, False),
    ],
    ids=["no drift", "non-blocking kinds only", "type mismatch blocks", "breaking change blocks"],
)
def test_contract_report_derives_passed_from_drift_kinds(kinds, supplied, passed):
    drifts = [{"kind": k, "location": "api/handler.go:12", "detail": "d"} for k in kinds]
    report = ContractReport.model_validate({"drifts": drifts, "passed": supplied})
    assert report.passed is passed


def test_contract_drift_rejects_unknown_kind():
    with pytest.raises(ValueError):
        ContractReport.model_validate({"drifts": [{"kind": "typo", "location": "a.go:1", "detail": "d"}]})


def _lines(*amounts: float) -> list[dict[str, Any]]:
    return [{"item": f"item {i}", "amount_usd": a, "period": "7d", "source": "test"} for i, a in enumerate(amounts)]


@pytest.mark.parametrize(
    ("payload", "total_usd", "over_budget"),
    [
        ({"lines": _lines(1.5, 2.5), "total_usd": 999.0, "over_budget": True}, 4.0, False),
        ({"lines": _lines(1.5, 2.5), "budget_usd": 3.0, "over_budget": False}, 4.0, True),
        ({"lines": _lines(1.5, 2.5), "budget_usd": 4.0, "over_budget": True}, 4.0, False),
        ({"lines": [], "budget_usd": 0.0, "over_budget": True}, 0.0, False),
    ],
    ids=["no budget never over", "over budget", "at budget is not over", "empty report"],
)
def test_cost_report_derives_total_and_budget_verdict(payload, total_usd, over_budget):
    report = CostReport.model_validate(payload)
    assert (report.total_usd, report.over_budget) == (total_usd, over_budget)


def test_cost_line_rejects_negative_amount():
    with pytest.raises(ValueError):
        CostReport.model_validate({"lines": _lines(-1.0)})


# ── litellm_spend_report ──────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status: int, payload: Any):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://proxy.test/spend/logs")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Stands in for httpx.AsyncClient; records the one GET it serves."""

    calls: list[dict[str, Any]] = []
    response: Any = None
    error: Exception | None = None

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url: str, **kwargs: Any) -> Any:
        type(self).calls.append({"url": url, "timeout": self.kwargs.get("timeout"), **kwargs})
        if type(self).error is not None:
            raise type(self).error
        return type(self).response


@pytest.fixture
def fake_proxy(monkeypatch):
    from pydantic_agents.core.models import factory

    _FakeClient.calls = []
    _FakeClient.response = None
    _FakeClient.error = None
    monkeypatch.setattr(factory, "_litellm_connection", lambda _spec: ("http://proxy.test/v1", "sk-test-key"))
    monkeypatch.setattr(finops_tools.httpx, "AsyncClient", _FakeClient)
    return _FakeClient


ROWS = [
    {"model": "deepseek-v4-pro", "key_alias": "backend", "spend": 1.25},
    {"model": "subagent", "key_alias": "finops", "spend": 0.25},
    {"model": "deepseek-v4-pro", "key_alias": "finops", "spend": 0.5},
    "not a row",
]


@pytest.mark.parametrize(
    ("days", "key_alias", "expected_days", "total", "per_model"),
    [
        (7, "", 7, 2.0, {"deepseek-v4-pro": 1.75, "subagent": 0.25}),
        (3, "finops", 3, 0.75, {"deepseek-v4-pro": 0.5, "subagent": 0.25}),
        (500, "", 90, 2.0, {"deepseek-v4-pro": 1.75, "subagent": 0.25}),
        (0, "", 1, 2.0, {"deepseek-v4-pro": 1.75, "subagent": 0.25}),
    ],
    ids=["all keys", "one alias", "window capped at 90", "window floored at 1"],
)
async def test_spend_report_builds_request_and_summarises(fake_proxy, days, key_alias, expected_days, total, per_model):
    fake_proxy.response = _FakeResponse(200, ROWS)
    out = json.loads(await finops_tools.litellm_spend_report(None, days=days, key_alias=key_alias))

    (call,) = fake_proxy.calls
    end = datetime.now(UTC).date()
    assert call["url"] == "http://proxy.test/spend/logs"
    assert call["params"] == {
        "start_date": (end - timedelta(days=expected_days)).isoformat(),
        "end_date": end.isoformat(),
    }
    assert call["headers"] == {"Authorization": "Bearer sk-test-key"}
    assert call["timeout"] == 15.0
    assert out["days"] == expected_days
    assert out["total_usd"] == total
    assert out["per_model"] == per_model
    assert sum(out["per_key_alias"].values()) == total


@pytest.mark.parametrize(
    ("response", "error", "fragment"),
    [
        (_FakeResponse(500, None), None, "Error: HTTP 500"),
        (None, httpx.ConnectError("refused"), "Error: request for spend logs"),
        (None, RuntimeError("bad json"), "Error: could not read spend logs"),
    ],
    ids=["http status", "transport error", "unexpected error"],
)
async def test_spend_report_returns_error_string_on_failure(fake_proxy, response, error, fragment):
    fake_proxy.response = response
    fake_proxy.error = error
    out = await finops_tools.litellm_spend_report(None, days=7)
    assert out.startswith(fragment)
    assert "sk-test-key" not in out


async def test_spend_report_reports_missing_proxy_config(monkeypatch):
    from pydantic_agents.core.models import factory

    def _unconfigured(spec):
        raise RuntimeError("no key")

    monkeypatch.setattr(factory, "_litellm_connection", _unconfigured)
    out = await finops_tools.litellm_spend_report(None)
    assert out.startswith("Error: LiteLLM proxy is not configured")


@pytest.mark.parametrize(
    ("payload", "total"),
    [({"data": ROWS[:2]}, 1.5), ({"logs": ROWS[:1]}, 1.25), ({"unexpected": 1}, 0.0), (ROWS[:0], 0.0)],
    ids=["data wrapper", "logs wrapper", "unknown dict", "empty list"],
)
async def test_spend_report_tolerates_payload_shapes(fake_proxy, payload, total):
    fake_proxy.response = _FakeResponse(200, payload)
    out = json.loads(await finops_tools.litellm_spend_report(None))
    assert out["total_usd"] == total


# ── release manager prompt contract ───────────────────────────────────────────


def _prompt(name: str) -> str:
    return "\n".join(str(p) for p in registry.get(name)._system_prompts)


@pytest.mark.parametrize(
    "line",
    [
        WORKSPACE_MARKER_LINE,
        ATTRIBUTION_LINE,
        "agent-ws: <id>",
        "🤖 Generated with [Claude Code](https://claude.com/claude-code)",
    ],
)
def test_release_manager_prompt_carries_the_pr_body_lines(line):
    assert line in _prompt("release_manager")
