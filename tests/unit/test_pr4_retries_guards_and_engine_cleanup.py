"""Unit tests for PR 4: AsyncOpenAI(max_retries=0), pipeline-wide RunUsage ceilings across stages & checkpoints,
A2A cycle detection + bounded child concurrency, routing private sub-agents & BridgeAgent through guarded provider
execution, and retirement of CrewAI/AutoGen/composite engines."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_agents.agents import frontend, test_eng
from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core import existing_code_handler, limits
from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
from pydantic_agents.core.models.core import Artifact, PipelineState
from pydantic_agents.core.models.factory import _make_chat_model
from pydantic_agents.core.pipeline.config import BLUEPRINTS, STAGES, Framework, PipelineConfig
from pydantic_agents.core.pipeline.factory import _GraphPipelineWrapper, build_default_pipeline
from pydantic_agents.core.registry import registry
from pydantic_agents.core.session_recorder import current_run_context
from pydantic_agents.core.settings import settings
from pydantic_agents.core.tools import a2a_tools

# ── 1. Single Retry Layer: AsyncOpenAI(max_retries=0) ────────────────────────


@pytest.mark.parametrize("provider", ["openai", "deepseek"])
def test_make_chat_model_disables_openai_sdk_internal_retries(provider):
    model = _make_chat_model(provider, "deepseek-v4-pro", "http://localhost:4000/v1", "dummy-key")
    # Verify underlying AsyncOpenAI client has max_retries == 0
    client = model._provider.client  # type: ignore[attr-defined]
    assert client.max_retries == 0


# ── 2. Pipeline-Wide RunUsage Cap Across Stages, Checkpoints & Engines ──────


def test_usage_limits_clamps_to_remaining_pipeline_budget_and_halts_when_exhausted(monkeypatch):
    monkeypatch.setattr(settings, "agent_total_tokens_limit", 2_000_000)
    monkeypatch.setattr(settings, "agent_request_limit", 40)
    monkeypatch.setattr(settings, "pipeline_total_tokens_limit", 500_000)
    monkeypatch.setattr(settings, "pipeline_request_limit", 10)

    pipe_usage = RunUsage(requests=7, input_tokens=300_000, output_tokens=150_000)
    clamped = limits.usage_limits("backend", pipeline_usage=pipe_usage)
    assert clamped.request_limit == 3
    assert clamped.total_tokens_limit == 50_000

    exhausted_toks = RunUsage(requests=2, input_tokens=400_000, output_tokens=100_000)
    with pytest.raises(UsageLimitExceeded, match="Pipeline token budget exhausted"):
        limits.usage_limits("backend", pipeline_usage=exhausted_toks)

    exhausted_reqs = RunUsage(requests=10, input_tokens=1_000, output_tokens=1_000)
    with pytest.raises(UsageLimitExceeded, match="Pipeline request budget exhausted"):
        limits.usage_limits("backend", pipeline_usage=exhausted_reqs)


@pytest.mark.asyncio
async def test_orgitration_pipeline_enforces_cumulative_budget_and_persists_across_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "ckpt")
    monkeypatch.setattr(settings, "pipeline_total_tokens_limit", 1_000)

    cfg = PipelineConfig(project="svc", blueprint="quick_backend", requirement="build api", run_id="pipe_000000000042")
    pipeline = build_default_pipeline(cfg)
    monkeypatch.setattr(pipeline.artifact_writer, "write_artifacts", MagicMock())

    fake_agent = MagicMock(model_settings=None)
    usage_res = MagicMock()
    usage_res.usage.return_value = MagicMock(input_tokens=600, output_tokens=500, requests=2)
    usage_res.output = Artifact(agent_name="backend", files={"main.go": "package main"})
    fake_agent.run = AsyncMock(return_value=usage_res)

    async def impl_stage(state: PipelineState, _c: PipelineConfig) -> PipelineState:
        # Runs 1 agent call consuming 1,100 tokens (> 1,000 pipeline ceiling)
        await pipeline.agent_provider.run_agent(asyncio.Semaphore(1), "backend", fake_agent, "implement")
        state.artifacts = [usage_res.output]
        return state

    async def qg_stage(state: PipelineState, _c: PipelineConfig) -> PipelineState:
        # Must fail immediately before making another LLM call because 1,100 >= 1,000!
        await pipeline.agent_provider.run_agent(asyncio.Semaphore(1), "review", fake_agent, "review")
        state.approved = True
        return state

    pipeline.stages[STAGES.IMPLEMENTATION].run = impl_stage
    pipeline.stages[STAGES.QUALITY_GATE].run = qg_stage

    final = await pipeline.execute()
    assert final.approved is False
    assert final.failure_report is not None
    assert final.failure_report.error_type == "usage_limit"
    assert final.total_input_tokens == 600
    assert final.total_output_tokens == 500
    assert final.total_requests == 2
    assert fake_agent.run.await_count == 1  # second stage's LLM call was blocked before firing!
    assert limits.pipeline_usage_var.get() is None
    assert limits.run_id_var.get() is None


@pytest.mark.asyncio
async def test_langgraph_and_existing_code_pipelines_catch_usage_limit_exceeded(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "memory_bridge_sync", False)

    fake_graph = MagicMock(ainvoke=AsyncMock(side_effect=UsageLimitExceeded("budget blown")))
    wrapper = _GraphPipelineWrapper(fake_graph, PipelineConfig(project="svc", blueprint="langgraph_quick"))
    lg_state = await wrapper.execute()
    assert lg_state.approved is False
    assert lg_state.failure_report is not None
    assert lg_state.failure_report.error_type == "usage_limit"

    monkeypatch.setattr(
        existing_code_handler,
        "_run_attempt",
        AsyncMock(side_effect=UsageLimitExceeded("existing-code budget blown")),
    )
    monkeypatch.setattr(existing_code_handler, "build_stage_event_recorder", lambda _p: MagicMock())
    ec_res = await existing_code_handler.run_existing_code_pipeline("add log", str(tmp_path))
    assert ec_res.passed is False
    assert any("existing-code budget blown" in e for e in ec_res.errors)


# ── 3. A2A Cycle Detection & Per-Parent Bounded Child Concurrency ───────────


@pytest.mark.asyncio
async def test_a2a_detects_direct_and_indirect_cycles():
    register_all_agents()

    # Direct self-cycle: backend -> backend
    tok = current_run_context.set(
        {
            "agent_name": "backend",
            "visited_agents": frozenset({"backend"}),
            "delegation_chain": ("backend",),
        }
    )
    try:
        self_res = await a2a_tools.delegate_to_agent(MagicMock(usage=None), "backend", "do work", "ctx")
    finally:
        current_run_context.reset(tok)
    assert self_res.startswith("ERROR: Delegation cycle detected (backend -> backend)")

    # Indirect cycle: ba -> cto -> ba
    tok2 = current_run_context.set(
        {
            "agent_name": "cto",
            "visited_agents": frozenset({"ba", "cto"}),
            "delegation_chain": ("ba", "cto"),
        }
    )
    try:
        indirect_res = await a2a_tools.delegate_to_agent(MagicMock(usage=None), "ba", "re-check stories", "ctx")
    finally:
        current_run_context.reset(tok2)
    assert indirect_res.startswith("ERROR: Delegation cycle detected (ba -> cto -> ba)")


@pytest.mark.asyncio
async def test_a2a_bounds_parallel_child_fanout_per_parent(monkeypatch):
    register_all_agents()
    active = 0
    peak_active = 0

    async def slow_query(**_kwargs):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.03)
        active -= 1
        return "child done"

    monkeypatch.setattr(a2a_tools, "instrumented_query", slow_query)
    parent_sem = asyncio.Semaphore(2)
    tok = current_run_context.set(
        {
            "agent_name": "planner",
            "visited_agents": frozenset({"planner"}),
            "delegation_chain": ("planner",),
            "child_semaphore": parent_sem,
        }
    )
    try:
        outputs = await asyncio.gather(
            *(
                a2a_tools.delegate_to_agent(MagicMock(usage=None), "backend", f"task {i}", "ctx")
                for i in range(6)
            )
        )
    finally:
        current_run_context.reset(tok)

    assert outputs == ["child done"] * 6
    assert peak_active == 2


# ── 4. Private Sub-Agents Route Through registry.run_subagent ───────────────


@pytest.mark.asyncio
async def test_private_subagents_execute_through_guarded_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATION_CODEBASE_ROOT", str(tmp_path))
    (tmp_path / "svc.go").write_text("package svc\nfunc Add(a, b int) int { return a + b }\n", encoding="utf-8")

    seen_agents: list[str] = []
    real_run_agent = PydanticAIAgentProvider.run_agent

    async def spy_run_agent(self, semaphore, agent_name, agent, prompt, **kwargs):
        seen_agents.append(agent_name)
        return await real_run_agent(self, semaphore, agent_name, agent, prompt, **kwargs)

    monkeypatch.setattr(PydanticAIAgentProvider, "run_agent", spy_run_agent)

    with frontend._component_writer.override(model=TestModel(custom_output_text="```vue\n<template/>\n```")):
        comp = await frontend.create_component(MagicMock(usage=None), "Card", "show card", "card-1")
    assert comp == "<template/>"

    with test_eng._test_writer.override(model=TestModel(custom_output_args={"code": "test('ok', () => {})"})):
        pw = await test_eng.generate_test_suite(MagicMock(usage=None), "Given x When y Then z")
    assert "test('ok'" in pw

    with test_eng._go_test_writer.override(
        model=TestModel(
            custom_output_args={"filepath": "svc_test.go", "code": "package svc", "description": "table test"}
        )
    ):
        go_t = await test_eng.generate_go_table_driven_tests(MagicMock(usage=None), "svc.go", "Add")
    assert "svc_test.go" in go_t

    assert seen_agents == [
        "frontend._component_writer",
        "test_eng._test_writer",
        "test_eng._go_test_writer",
    ]
    # Still private: none of the three helper writers are exposed in registry.list_agents()
    assert not (set(seen_agents) & set(registry.list_agents()))


# ── 5. CrewAI, AutoGen & composite Retirement Verification ──────────────────


def test_only_legacy_and_langgraph_frameworks_remain():
    assert {f.value for f in Framework} == {"legacy", "langgraph"}
    assert {bp["framework"] for bp in BLUEPRINTS.values()} <= {"legacy", "langgraph"}
