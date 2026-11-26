"""Unit tests for PR 3: OTLP content redaction (no instrument_pydantic, no unredacted fallback), opt-in redacted
SessionRecorder, conftest ALLOW_MODEL_REQUESTS=False + trace isolation, gen_ai.agent.name, and run_id + cost_usd +
verdicts in platform_metrics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic_ai.models
import pytest

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core import observability, platform_metrics, session_recorder
from pydantic_agents.core.adapters import agents as agents_adapter
from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.models.core import AgentAdvice, CostLine, CostReport, CriterionResult, TestRunReport
from pydantic_agents.core.registry import registry
from pydantic_agents.core.session_recorder import SessionRecorder, current_run_context, recorder, redact_secrets
from pydantic_agents.core.settings import settings
from pydantic_agents.core.telemetry import telemetry
from pydantic_agents.core.tools import a2a_tools
from pydantic_agents.core.tools import registry as tools_registry
from tests.unit.db_stub import stub_database

# ── 1. Observability OTLP Redaction & No Unredacted Fallback ─────────────────


def test_instrument_never_calls_instrument_pydantic_or_unredacted_fallback():
    fake = MagicMock()
    # Both (include_content=False, include_binary_content=False) and (include_content=False) raise TypeError:
    # must record a problem rather than calling unredacted logfire.instrument_pydantic_ai()!
    fake.instrument_pydantic_ai.side_effect = TypeError("unsupported signature")
    with patch.object(observability, "logfire", fake):
        problems = observability._instrument()

    fake.instrument_pydantic.assert_not_called()
    assert all(call.kwargs.get("include_content") is False for call in fake.instrument_pydantic_ai.call_args_list)
    assert any("pydantic-ai:" in p for p in problems)


def test_span_attrs_exclude_raw_pattern_and_context_preview():
    sample_pat = "sensitive_regex_pattern_value"
    attrs = tools_registry._build_span_attrs("search_codebase", {"pattern": sample_pat, "capability": "ba"})
    assert "arg_pattern" not in attrs
    assert attrs["arg_pattern_len"] == len(sample_pat)
    assert attrs["arg_capability"] == "ba"


@pytest.mark.asyncio
async def test_a2a_delegate_span_excludes_context_summary_preview(monkeypatch):
    captured: dict = {}

    async def fake_instrumented_query(**kwargs):
        captured.update(kwargs["attributes"])
        return "ok"

    monkeypatch.setattr(a2a_tools, "instrumented_query", fake_instrumented_query)
    res = await a2a_tools.delegate_to_agent(MagicMock(usage=None), "ba", "write story", "secret context body")
    assert res == "ok"
    assert "context_summary_preview" not in captured
    assert captured["context_summary_length"] == len("secret context body")


# ── 2. Opt-In & Secret-Redacted SessionRecorder + Tool-Call Recording ────────


def test_default_session_recorder_is_opt_in_until_enabled(tmp_path, monkeypatch):
    default_rec = SessionRecorder()  # constructed with DEFAULT_TRACE_FILE
    monkeypatch.setattr(default_rec, "filepath", str(tmp_path / "trace.jsonl"))
    monkeypatch.delenv("SESSION_TRACE_ENABLED", raising=False)
    monkeypatch.setattr(settings, "session_trace_enabled", False)

    assert default_rec.is_enabled() is False
    rec_obj = default_rec._new_record("r1", "ba", "ba", "prompt")
    default_rec._finish(rec_obj, success=True, output="done")
    assert not (tmp_path / "trace.jsonl").exists()

    # Enabling via env var writes redacted records
    monkeypatch.setenv("SESSION_TRACE_ENABLED", "true")
    assert default_rec.is_enabled() is True
    synthetic_key = "sk-" + "proj-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"
    rec_obj2 = default_rec._new_record("r2", "ba", "ba", f"key is {synthetic_key}")
    default_rec._finish(rec_obj2, success=True, output="done")
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert "[REDACTED]" in parsed["prompt"]
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234" not in parsed["prompt"]

    # Explicit enabled=False overrides custom filepath
    explicit_off = SessionRecorder(str(tmp_path / "off.jsonl"), enabled=False)
    assert explicit_off.is_enabled() is False


def test_redact_secrets_scrubs_keys_dsns_headers_and_nested_structures():
    # Construct synthetic test shapes dynamically so static secret scanners never trip on test fixtures
    fake_sk = "sk-" + "live-" + ("a1B2" * 6)
    fake_ghp = "ghp_" + ("X9y8Z7" * 6)
    fake_dsn = "postgresql://admin:SuperSecret99@pgvector.local:5432/agents"
    fake_bearer = "Bearer " + ("tok9876543210" * 2)

    payload = {
        "prompt": f"Connect to {fake_dsn} with {fake_sk}",
        "headers": [fake_bearer, ("gh", fake_ghp)],
        "count": 42,
    }
    scrubbed = redact_secrets(payload)
    dumped = json.dumps(scrubbed)

    assert "SuperSecret99" not in dumped
    assert fake_sk not in dumped
    assert fake_ghp not in dumped
    assert "tok9876543210" not in dumped
    assert "postgresql://[REDACTED]@pgvector.local:5432/agents" in scrubbed["prompt"]
    assert scrubbed["count"] == 42


@pytest.mark.asyncio
async def test_wrapped_tools_record_redacted_calls_inside_record_span(tmp_path):
    rec = SessionRecorder(str(tmp_path / "spanned.jsonl"))
    fake_tok = "sk-" + "test-" + ("9z8Y" * 6)

    async def sample_async_tool(secret_arg: str = "") -> str:
        return f"used {secret_arg}"

    def sample_sync_tool(name: str = "") -> str:
        return f"hello {name}"

    wrapped_async = tools_registry._wrap_tool_with_tracing(sample_async_tool, "sample_async_tool")
    wrapped_sync = tools_registry._wrap_tool_with_tracing(sample_sync_tool, "sample_sync_tool")

    # Swap module-level recorder temporarily so _wrap_tool_with_tracing writes into our explicit `rec`
    orig_rec = tools_registry.recorder
    tools_registry.recorder = rec
    try:
        async with rec.record_span("run_42", "backend", "implementation", "build service") as span:
            await wrapped_async(secret_arg=fake_tok)
            wrapped_sync(name="world")
            span.set_output({"token": fake_tok})
    finally:
        tools_registry.recorder = orig_rec

    (entry,) = session_recorder.extract_traces(str(tmp_path / "spanned.jsonl"))
    assert len(entry["tool_calls"]) == 2
    assert entry["tool_calls"][0]["tool_name"] == "sample_async_tool"
    assert fake_tok not in json.dumps(entry)
    assert entry["output_json"] == {"token": "[REDACTED]"}


# ── 3. conftest.py Autouse Guardrails ────────────────────────────────────────


def test_conftest_blocks_live_model_requests_and_isolates_trace_files(tmp_path):
    assert pydantic_ai.models.ALLOW_MODEL_REQUESTS is False
    with pytest.raises(RuntimeError, match="ALLOW_MODEL_REQUESTS is False"):
        pydantic_ai.models.check_allow_model_requests()

    repo_root = Path(__file__).resolve().parents[2]
    assert Path(telemetry.log_file).parent == tmp_path
    assert Path(recorder.filepath).parent == tmp_path
    assert Path(telemetry.log_file).resolve() != (repo_root / "telemetry.jsonl").resolve()


# ── 4. gen_ai.agent.name, run_id Propagation, cost_usd & Verdicts ────────────


@pytest.mark.asyncio
async def test_all_registered_agents_have_agent_name_and_propagate_run_id(monkeypatch):
    register_all_agents()
    names = registry.list_agents()
    assert names
    for name in names:
        agent = registry.get(name)
        assert agent.name == name

    observed_ctx: dict = {}

    async def fake_run_agent(**kwargs):
        observed_ctx.update(current_run_context.get())
        return SimpleNamespace(output="ok")

    monkeypatch.setattr(registry._provider, "run_agent", fake_run_agent)
    tok = run_id_var.set("call_abc123def456")
    try:
        await registry.run("ba", "write a user story")
    finally:
        run_id_var.reset(tok)

    assert observed_ctx["agent_name"] == "ba"
    assert observed_ctx["run_id"] == "call_abc123def456"


def test_verdict_of_and_estimate_cost_usd_cover_all_gate_models(monkeypatch):
    monkeypatch.setattr(settings, "cost_default_input_per_1m_usd", 1.0)
    monkeypatch.setattr(settings, "cost_default_output_per_1m_usd", 2.0)
    monkeypatch.setattr(settings, "cost_subagent_input_per_1m_usd", 0.1)
    monkeypatch.setattr(settings, "cost_subagent_output_per_1m_usd", 0.2)

    # None when neither token count is reported
    assert platform_metrics.estimate_cost_usd("cto", None, None) is None
    # Reasoning model tier (cto_model = deepseek-v4-pro): 100k in ($0.10) + 50k out ($0.10) = $0.20
    assert platform_metrics.estimate_cost_usd("cto", 100_000, 50_000) == pytest.approx(0.20)
    # Subagent model tier (docs_model = subagent): 100k in ($0.01) + 50k out ($0.01) = $0.02
    assert platform_metrics.estimate_cost_usd("docs", 100_000, 50_000) == pytest.approx(0.02)
    # test_eng maps to settings.test_model
    assert platform_metrics.estimate_cost_usd("test_eng", 1_000_000, 0) == pytest.approx(1.0)

    # Verdicts for AgentAdvice, TestRunReport, and CostReport
    assert platform_metrics.verdict_of(AgentAdvice(approved_to_proceed=True, adr_id="ADR-0001")) == "passed"
    assert platform_metrics.verdict_of(AgentAdvice(approved_to_proceed=False)) == "failed"

    passing_test_report = TestRunReport(
        criterion_results=[CriterionResult(criterion_id="C1", passed=True)],
        coverage_percent=90.0,
    )
    assert platform_metrics.verdict_of(passing_test_report) == "passed"
    assert platform_metrics.verdict_of(TestRunReport(criterion_results=[])) == "failed"

    under_budget = CostReport(
        lines=[CostLine(item="llm", amount_usd=10.0, period="monthly", source="litellm")],
        budget_usd=20.0,
    )
    over_budget = CostReport(
        lines=[CostLine(item="llm", amount_usd=30.0, period="monthly", source="litellm")],
        budget_usd=20.0,
    )
    assert platform_metrics.verdict_of(under_budget) == "passed"
    assert platform_metrics.verdict_of(over_budget) == "failed"


@pytest.mark.asyncio
async def test_provider_records_run_id_cost_usd_verdict_and_telemetry_run_id(monkeypatch):
    class _Conn:
        def __init__(self, db):
            self.db = db

        async def execute(self, sql, *args):
            if "INSERT INTO platform_metrics" in sql:
                row_id, agent, run_id, duration, success, in_tok, out_tok, cost_usd, verdict = args
                self.db[row_id] = {
                    "id": row_id,
                    "agent": agent,
                    "run_id": run_id,
                    "duration": duration,
                    "success": success,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost_usd,
                    "verdict": verdict,
                    "created_at": None,
                }

        async def fetch(self, _sql, *_args):
            return list(self.db.values())

    stub_database(monkeypatch, platform_metrics, _Conn)

    fake_result = MagicMock()
    fake_result.usage.return_value = MagicMock(input_tokens=10_000, output_tokens=2_000)
    fake_result.output = AgentAdvice(approved_to_proceed=True, adr_id="ADR-0001")
    fake_agent = MagicMock(model_settings=None, run=AsyncMock(return_value=fake_result))

    tok = run_id_var.set("pipe_123456789abc")
    try:
        await agents_adapter.PydanticAIAgentProvider().run_agent(
            asyncio.Semaphore(1), "cto", fake_agent, "review design"
        )
    finally:
        run_id_var.reset(tok)

    rows = await platform_metrics.recent_calls()
    assert len(rows) == 1
    row = rows[0]
    assert row["agent"] == "cto"
    assert row["run_id"] == "pipe_123456789abc"
    assert row["cost_usd"] is not None and row["cost_usd"] > 0
    assert row["verdict"] == "passed"

    # Also verify telemetry.jsonl received run_id in our isolated tmp_path file
    telemetry_lines = Path(telemetry.log_file).read_text(encoding="utf-8").splitlines()
    assert json.loads(telemetry_lines[-1])["run_id"] == "pipe_123456789abc"

    # Verify format_usage renders cost when cost_usd is present
    formatted = platform_metrics.format_usage(
        [
            {
                "agent": "cto",
                "runs": 1,
                "successes": 1,
                "p50_seconds": 1.2,
                "p95_seconds": 1.2,
                "runs_with_tokens": 1,
                "input_tokens": 10_000,
                "output_tokens": 2_000,
                "cost_usd": 0.0049,
                "verdict_passed": 1,
                "verdict_failed": 0,
                "last_failure": None,
            }
        ],
        7,
        "cto",
    )
    assert "cost=$0.0049" in formatted
