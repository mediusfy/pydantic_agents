"""Unit & ASGI tests for PR 1: MCP Bearer auth + /mcp/admin split, atomic run-bound CTO gate, cto_agent ModelRetry,
and gating artifact disk writes on pipeline approval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic_ai.models
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from starlette.testclient import TestClient

from pydantic_agents import mcp_server
from pydantic_agents.agents.cto import OrgContext, cto_agent, cto_prompt
from pydantic_agents.core import cto_gate, existing_code_handler, pipeline_jobs
from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.models.core import AgentAdvice, Artifact, FailureReport, Finding, PipelineState
from pydantic_agents.core.pipeline.config import STAGES, PipelineConfig
from pydantic_agents.core.pipeline.factory import _GraphPipelineWrapper, build_default_pipeline
from pydantic_agents.core.settings import settings

# ── 1. MCP Auth, /mcp/admin Split, Streamable HTTP & DNS Rebinding ──────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" yes ", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("", False),
    ],
)
def test_dns_rebinding_protection_parses_truthy_strings(monkeypatch, raw, expected):
    monkeypatch.setenv("MCP_DNS_REBINDING_PROTECTION", raw)
    assert mcp_server._dns_rebinding_protection_enabled() is expected
    ts = mcp_server._build_transport_security()
    assert ts.enable_dns_rebinding_protection is expected


@pytest.mark.asyncio
async def test_privileged_tools_are_isolated_on_mcp_admin():
    worker_tools = {t.name for t in await mcp_server.mcp.list_tools()}
    admin_tools = {t.name for t in await mcp_server.mcp_admin.list_tools()}

    privileged = {"approve_cto_gate", "save_adr", "cancel_pipeline_run"}
    assert privileged.issubset(admin_tools)
    assert worker_tools.isdisjoint(privileged)
    assert "search_adrs" in worker_tools


def test_http_app_auth_and_admin_isolation(monkeypatch):
    monkeypatch.setattr(settings, "mcp_auth_token", None)
    monkeypatch.setattr(settings, "mcp_admin_token", None)
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MCP_ADMIN_TOKEN", raising=False)

    app = mcp_server.create_http_app()
    mcp_headers = {
        "Host": "127.0.0.1:8080",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-suite", "version": "1.0"},
        },
    }

    with TestClient(app, base_url="http://127.0.0.1:8080") as client:
        # 1. /healthz is always open even when tokens are configured
        monkeypatch.setenv("MCP_AUTH_TOKEN", "dummy-worker-tok")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "dummy-admin-tok")
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        # 2. /mcp/admin fails closed (503) when MCP_ADMIN_TOKEN is unset
        monkeypatch.delenv("MCP_ADMIN_TOKEN", raising=False)
        res_disabled = client.post("/mcp/admin", json=init_payload, headers=mcp_headers)
        assert res_disabled.status_code == 503
        assert "MCP_ADMIN_TOKEN" in res_disabled.json()["error"]

        # 3. Restore MCP_ADMIN_TOKEN: missing / malformed / wrong tokens get 401 with WWW-Authenticate
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "dummy-admin-tok")
        for bad_headers in (
            {},
            {"Authorization": "Basic dXNlcjpwYXNz"},
            {"Authorization": "Bearer    "},
            {"Authorization": "Bearer wrong-tok"},
        ):
            res_unauth = client.post("/mcp/admin", json=init_payload, headers={**mcp_headers, **bad_headers})
            assert res_unauth.status_code == 401
            assert res_unauth.headers.get("www-authenticate") == "Bearer"

        # 4. Worker token on /mcp/admin gets 403 Forbidden (cannot escalate to approve_cto_gate)
        res_forbidden = client.post(
            "/mcp/admin",
            json=init_payload,
            headers={**mcp_headers, "Authorization": "Bearer dummy-worker-tok"},
        )
        assert res_forbidden.status_code == 403
        assert "www-authenticate" not in {k.lower() for k in res_forbidden.headers}

        # 5. Admin token on /mcp/admin succeeds via Streamable HTTP
        res_admin_ok = client.post(
            "/mcp/admin",
            json=init_payload,
            headers={**mcp_headers, "Authorization": "Bearer dummy-admin-tok"},
        )
        assert res_admin_ok.status_code == 200

        # 6. Worker route (/mcp): missing / wrong token -> 401; worker OR admin token -> 200
        res_worker_missing = client.post("/mcp", json=init_payload, headers=mcp_headers)
        assert res_worker_missing.status_code == 401
        assert res_worker_missing.headers.get("www-authenticate") == "Bearer"

        res_worker_wrong = client.post(
            "/mcp",
            json=init_payload,
            headers={**mcp_headers, "Authorization": "Bearer wrong-tok"},
        )
        assert res_worker_wrong.status_code == 401

        for valid_tok in ("dummy-worker-tok", "dummy-admin-tok"):
            res_worker_ok = client.post(
                "/mcp",
                json=init_payload,
                headers={**mcp_headers, "Authorization": f"Bearer {valid_tok}"},
            )
            assert res_worker_ok.status_code == 200

        # 7. Unset MCP_AUTH_TOKEN leaves worker route open (local dev compatibility) while /mcp/admin stays protected
        monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
        res_worker_open = client.post("/mcp", json=init_payload, headers=mcp_headers)
        assert res_worker_open.status_code == 200


@pytest.mark.asyncio
async def test_bearer_middleware_passes_non_http_scopes_through():
    inner = AsyncMock()
    mw = mcp_server._MCPBearerAuthMiddleware(inner)
    scope = {"type": "websocket", "path": "/mcp/admin"}
    await mw(scope, AsyncMock(), AsyncMock())
    inner.assert_awaited_once()


# ── 2. Atomic, run_id-Bound .cto_approved & approve_cto_gate ─────────────────


def test_write_approval_is_atomic_and_cleans_up_on_replace_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    marker = cto_gate.write_approval("svc", True, run_id="pipe_000000000001", adr_id="ADR-0001")
    assert marker.name == ".cto_approved.pipe_000000000001"
    assert json.loads(marker.read_text()) == {
        "approved": True,
        "run_id": "pipe_000000000001",
        "adr_id": "ADR-0001",
    }
    assert list(marker.parent.glob("*.tmp")) == []

    # Simulate os.replace failing: temporary file must be unlinked
    with (
        patch.object(os, "replace", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        cto_gate.write_approval("svc", False, run_id="pipe_000000000002")
    assert list((tmp_path / "svc").glob("*.tmp")) == []


@pytest.mark.parametrize("bad_run_id", ["../evil", "a/b", "..", ".hidden", "x" * 70])
def test_write_approval_rejects_traversal_run_id(monkeypatch, tmp_path, bad_run_id):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    with pytest.raises(PathTraversalError):
        cto_gate.write_approval("svc", True, run_id=bad_run_id, adr_id="ADR-0001")


@pytest.mark.parametrize(
    ("raw", "expected_run_id", "want"),
    [
        ("", "run_1", None),
        ("   \n\t  ", "run_1", None),
        ("{not-json", "run_1", None),
        ('["list"]', "run_1", None),
        ('{"approved": true, "run_id": 123}', "run_1", None),
        ('{"approved": "yes", "run_id": "run_1"}', "run_1", None),
        ('{"approved": true, "run_id": "run_other", "adr_id": "0001"}', "run_1", None),
        ('{"approved": true, "run_id": "run_1", "adr_id": "0001"}', "run_1", True),
        ('{"approved": false, "run_id": "run_1", "adr_id": ""}', "run_1", False),
        ('{"approved": true, "run_id": "", "adr_id": "0001"}', "run_1", True),
    ],
)
def test_read_marker_json_and_empty_edge_cases(tmp_path, raw, expected_run_id, want):
    marker = tmp_path / ".cto_approved"
    marker.write_text(raw, encoding="utf-8")
    assert cto_gate.read_marker(marker, expected_run_id=expected_run_id) is want


def test_read_marker_handles_oserror(tmp_path, monkeypatch):
    marker = tmp_path / ".cto_approved"
    marker.write_text("true", encoding="utf-8")
    monkeypatch.setattr(Path, "read_text", MagicMock(side_effect=OSError("io error")))
    assert cto_gate.read_marker(marker) is None


@pytest.mark.asyncio
async def test_concurrent_runs_on_same_project_do_not_steal_each_others_markers(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.delenv("ORCHESTRATION_CTO_OVERRIDE", raising=False)
    orig_poll = cto_gate._poll_for_file_approval
    monkeypatch.setattr(
        cto_gate,
        "_poll_for_file_approval",
        lambda path, timeout_seconds=3600.0: orig_poll(path, poll_interval=0.01, timeout_seconds=timeout_seconds),
    )

    marker_b = cto_gate.write_approval("svc", True, run_id="pipe_00000000000b", adr_id="ADR-0001")

    # run_a must NOT consume run_b's marker, nor even touch it
    approved_a = await cto_gate.apply(
        PipelineState(run_id="pipe_00000000000a", stage="advisory", cto_approved=False),
        project="svc",
        timeout_seconds=0.04,
    )
    assert approved_a is False
    assert marker_b.exists()

    # Also verify a JSON marker written at the legacy .cto_approved path with run_b's run_id is left intact by run_a
    legacy_bound = tmp_path / "svc" / ".cto_approved"
    legacy_bound.write_text(json.dumps({"approved": True, "run_id": "pipe_00000000000b", "adr_id": "ADR-0001"}))
    assert (
        await cto_gate.apply(
            PipelineState(run_id="pipe_00000000000a", stage="advisory", cto_approved=False),
            project="svc",
            timeout_seconds=0.03,
        )
        is False
    )
    assert legacy_bound.exists()
    legacy_bound.unlink()

    # run_b consumes its own marker cleanly
    approved_b = await cto_gate.apply(
        PipelineState(run_id="pipe_00000000000b", stage="advisory", cto_approved=False),
        project="svc",
        timeout_seconds=0.05,
    )
    assert approved_b is True
    assert not marker_b.exists()


@pytest.mark.asyncio
async def test_approve_cto_gate_validates_project_and_run_state(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(settings, "database_url", None)

    # 1. Invalid project name
    res = await mcp_server.approve_cto_gate("../bad", approved=False)
    assert res.startswith("ERROR: Invalid project_name")

    # 2. No active runs at all
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {})
    res = await mcp_server.approve_cto_gate("svc", approved=True, adr_id="ADR-0001")
    assert res.startswith("ERROR: No active pipeline run")

    # 3. Unknown explicit run_id
    res = await mcp_server.approve_cto_gate("svc", approved=True, adr_id="ADR-0001", run_id="pipe_000000000999")
    assert res == "ERROR: Unknown run_id 'pipe_000000000999'."

    # 4. Explicit run_id belongs to another project
    monkeypatch.setattr(
        pipeline_jobs,
        "PIPELINE_RUNS",
        {"pipe_000000000001": {"project": "other-svc", "status": "waiting_cto_approval"}},
    )
    res = await mcp_server.approve_cto_gate("svc", approved=True, adr_id="ADR-0001", run_id="pipe_000000000001")
    assert "belongs to project 'other-svc'" in res

    # 5. Explicit run_id already finished
    monkeypatch.setattr(
        pipeline_jobs,
        "PIPELINE_RUNS",
        {"pipe_000000000001": {"project": "svc", "status": "completed"}},
    )
    res = await mcp_server.approve_cto_gate("svc", approved=True, adr_id="ADR-0001", run_id="pipe_000000000001")
    assert "already completed" in res

    # 6. Multiple waiting runs without explicit run_id
    monkeypatch.setattr(
        pipeline_jobs,
        "PIPELINE_RUNS",
        {
            "pipe_000000000001": {"project": "svc", "status": "waiting_cto_approval"},
            "pipe_000000000002": {"project": "svc", "status": "waiting_cto_approval"},
        },
    )
    res = await mcp_server.approve_cto_gate("svc", approved=True, adr_id="ADR-0001")
    assert res.startswith("ERROR: Multiple active runs for project 'svc'")

    # 7. One waiting run + one running run -> selects the waiting run automatically
    monkeypatch.setattr(
        pipeline_jobs,
        "PIPELINE_RUNS",
        {
            "pipe_000000000001": {"project": "svc", "status": "waiting_cto_approval"},
            "pipe_000000000002": {"project": "svc", "status": "running"},
        },
    )
    res = await mcp_server.approve_cto_gate("svc", approved=True, adr_id="ADR-0001")
    assert "run 'pipe_000000000001'" in res
    assert (tmp_path / "svc" / ".cto_approved.pipe_000000000001").exists()

    # 8. Explicit valid run_id targets the specified active run directly
    res_explicit = await mcp_server.approve_cto_gate(
        "svc", approved=True, adr_id="ADR-0002", run_id="pipe_000000000002"
    )
    assert "run 'pipe_000000000002'" in res_explicit
    assert (tmp_path / "svc" / ".cto_approved.pipe_000000000002").exists()


@pytest.mark.asyncio
async def test_admin_threaded_save_adr_tool_runs_off_event_loop():
    tool = mcp_server.mcp_admin._tool_manager.get_tool("save_adr")
    assert tool is not None
    with patch.object(mcp_server.vector_db, "add_documents") as mock_add:
        out = await tool.fn("ADR-0099", "# Decision body")
    assert out == "Saved ADR ADR-0099 successfully."
    mock_add.assert_called_once()


# ── 3. cto_prompt & @cto_agent.output_validator (ModelRetry on missing adr_id) ─


def test_cto_prompt_documents_adr_id_and_findings():
    assert "`adr_id`" in cto_prompt
    assert "`findings`" in cto_prompt
    assert "CRITICAL" in cto_prompt


def test_direct_agent_advice_construction_stays_fail_closed():
    missing_adr = AgentAdvice(approved_to_proceed=True, adr_id="")
    assert missing_adr.approved_to_proceed is False
    assert missing_adr._approval_withdrawn_reason == "missing_adr_id"

    crit = AgentAdvice(
        approved_to_proceed=True,
        adr_id="ADR-0001",
        findings=[
            Finding(
                id="F-1",
                source_agent="cto",
                severity="CRITICAL",
                description="Plaintext secret",
                location="config.yaml:1",
            )
        ],
    )
    assert crit.approved_to_proceed is False
    assert crit._approval_withdrawn_reason == "critical_finding"


@pytest.mark.asyncio
async def test_cto_agent_output_validator_retries_when_adr_id_is_missing(monkeypatch):
    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)
    calls: list[list[ModelMessage]] = []

    def model_logic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(list(messages))
        output_tool = info.output_tools[0].name
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=output_tool,
                        args={"approved_to_proceed": True, "adr_id": "", "recommendations": ["use hexagonal"]},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=output_tool,
                    args={
                        "approved_to_proceed": True,
                        "adr_id": "ADR-0001",
                        "recommendations": ["use hexagonal"],
                    },
                )
            ]
        )

    deps = OrgContext(tech_stack=["Go"], compliance_tier="standard", monthly_cloud_budget=1000.0)
    with cto_agent.override(model=FunctionModel(model_logic)):
        result = await cto_agent.run("Review health check service", deps=deps)

    assert len(calls) == 2
    retry_parts = [p for m in calls[1] for p in getattr(m, "parts", []) if isinstance(p, RetryPromptPart)]
    assert retry_parts and "adr_id" in str(retry_parts[-1].content)
    assert result.output.approved_to_proceed is True
    assert result.output.adr_id == "ADR-0001"


@pytest.mark.asyncio
async def test_cto_agent_critical_finding_blocks_without_missing_adr_retry(monkeypatch):
    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)
    call_count = 0

    def model_logic(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "approved_to_proceed": True,
                        "adr_id": "",
                        "findings": [
                            {
                                "id": "F-CRIT",
                                "source_agent": "cto",
                                "severity": "CRITICAL",
                                "description": "RCE exposure",
                                "location": "cmd/server/main.go:10",
                            }
                        ],
                    },
                )
            ]
        )

    deps = OrgContext(tech_stack=["Go"], compliance_tier="standard", monthly_cloud_budget=1000.0)
    with cto_agent.override(model=FunctionModel(model_logic)):
        result = await cto_agent.run("Review unsafe service", deps=deps)

    assert call_count == 1
    assert result.output.approved_to_proceed is False


# ── 4. Artifact Disk Writes Gated on Pipeline Approval (BP-R3) ───────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_orgitration_pipeline_writes_artifacts_only_when_approved(tmp_path, monkeypatch, approved):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "cp")
    config = PipelineConfig(project="svc", blueprint="security_review", requirement="audit")
    pipeline = build_default_pipeline(config)

    async def fake_gate(state: PipelineState, _cfg: PipelineConfig) -> PipelineState:
        state.artifacts = [Artifact(agent_name="backend", files={"main.go": "package main"})]
        state.approved = approved
        return state

    pipeline.stages[STAGES.QUALITY_GATE].run = fake_gate
    write_spy = MagicMock()
    monkeypatch.setattr(pipeline.artifact_writer, "write_artifacts", write_spy)

    final = await pipeline.execute()
    assert final.approved is approved
    assert write_spy.call_count == (1 if approved else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_langgraph_wrapper_writes_artifacts_only_when_approved(tmp_path, monkeypatch, approved):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    config = PipelineConfig(project="svc", blueprint="langgraph_quick", requirement="build")

    final_state = PipelineState(
        run_id="pipe_000000000010",
        stage="quality_gate",
        approved=approved,
        artifacts=[Artifact(agent_name="backend", files={"main.go": "package main"})],
    )
    fake_graph = MagicMock(ainvoke=AsyncMock(return_value=final_state))
    wrapper = _GraphPipelineWrapper(fake_graph, config)

    with patch("pydantic_agents.core.adapters.workspace.ArtifactStorageAdapter.write_artifacts") as write_spy:
        res = await wrapper.execute()

    assert res.approved is approved
    assert write_spy.call_count == (1 if approved else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_crewai_pipeline_writes_artifacts_only_when_approved(tmp_path, monkeypatch, approved):
    import importlib
    import sys
    import types

    fake_state = PipelineState(
        run_id="pipe_000000000011",
        stage="review",
        approved=approved,
        artifacts=[Artifact(agent_name="backend", files={"main.go": "package main"})],
    )
    stub_bridge = types.ModuleType("pydantic_agents.core.pipeline.crewai.bridge")
    stub_bridge.run_crewai_pipeline = AsyncMock(return_value=fake_state)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pydantic_agents.core.pipeline.crewai.bridge", stub_bridge)
    monkeypatch.delitem(sys.modules, "pydantic_agents.core.pipeline.crewai.orchestrator", raising=False)

    crew_orch_mod = importlib.import_module("pydantic_agents.core.pipeline.crewai.orchestrator")
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    config = PipelineConfig(project="svc", blueprint="crew_audit", requirement="audit")
    pipeline = crew_orch_mod.CrewAIPipeline(config)
    write_spy = MagicMock()
    monkeypatch.setattr(pipeline.artifact_writer, "write_artifacts", write_spy)

    res = await pipeline.execute()

    assert res.approved is approved
    assert write_spy.call_count == (1 if approved else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_passes", [False, True])
async def test_existing_code_pipeline_writes_result_only_when_passed(tmp_path, monkeypatch, attempt_passes):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(settings, "memory_bridge_sync", False)

    async def fake_attempt(result, _recorder, _attempt, _options):
        result.artifacts = [Artifact(agent_name="service_engineer", files={"svc.go": "package svc"})]
        result.changed_files = ["svc.go"]
        result.passed = attempt_passes
        if not attempt_passes:
            result.errors.append("quality gate rejected")
        return attempt_passes

    write_spy = MagicMock()
    monkeypatch.setattr(existing_code_handler, "_run_attempt", fake_attempt)
    monkeypatch.setattr(existing_code_handler, "_write_result", write_spy)
    monkeypatch.setattr(existing_code_handler, "build_stage_event_recorder", lambda _p: MagicMock())

    res = await existing_code_handler.run_existing_code_pipeline("add metric", str(tmp_path))
    assert res.passed is attempt_passes
    assert write_spy.call_count == (1 if attempt_passes else 0)


@pytest.mark.asyncio
async def test_scaffold_and_build_service_reports_unapproved_pipeline_honestly():
    unapproved_with_report = PipelineState(
        run_id="r1",
        stage="advisory",
        approved=False,
        failure_report=FailureReport(source="advisory", error_type="blocked", trace="The CTO gate did not approve."),
    )
    fake_pipeline = MagicMock(execute=AsyncMock(return_value=unapproved_with_report))
    with patch("pydantic_agents.core.pipeline.factory.build_pipeline", return_value=fake_pipeline):
        out = await mcp_server.scaffold_and_build_service("billing-svc", "billing")
    assert out.startswith("Error: Service 'billing-svc' scaffolding pipeline did not pass gates")
    assert "The CTO gate did not approve." in out
    assert "Unapproved artifacts were not written." in out

    unapproved_no_report = PipelineState(run_id="r2", stage="quality_gate", approved=False)
    fake_pipeline_2 = MagicMock(execute=AsyncMock(return_value=unapproved_no_report))
    with patch("pydantic_agents.core.pipeline.factory.build_pipeline", return_value=fake_pipeline_2):
        out2 = await mcp_server.scaffold_and_build_service("billing-svc", "billing")
    assert "pipeline finished without gate approval" in out2
