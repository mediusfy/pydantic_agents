"""Run-level LiteLLM/Langfuse tagging, per-agent model defaults and pipeline resume."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
from pydantic_agents.core.limits import run_id_var, trace_model_settings
from pydantic_agents.core.pipeline import orchestrator as orch
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.settings import Settings, settings


@pytest.mark.parametrize(
    ("run_id", "provider", "want"),
    [
        (
            "pipe_abc",
            "litellm",
            {"extra_body": {"metadata": {"session_id": "pipe_abc", "trace_name": "pydantic-agents/ba"}}},
        ),
        (None, "litellm", None),
        ("pipe_abc", "openai", None),
        ("pipe_abc", None, None),
    ],
)
def test_trace_model_settings(monkeypatch, run_id, provider, want):
    monkeypatch.setattr(settings, "model_provider", provider)
    token = run_id_var.set(run_id)
    try:
        assert trace_model_settings("ba") == want
    finally:
        run_id_var.reset(token)


@pytest.mark.parametrize("run_id", ["pipe_abc", None])
async def test_run_agent_passes_session_metadata(monkeypatch, run_id):
    monkeypatch.setattr(settings, "model_provider", "litellm")
    agent = MagicMock()
    agent.run = AsyncMock(return_value="ok")
    token = run_id_var.set(run_id)
    try:
        await PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "ba", agent, "hi")
    finally:
        run_id_var.reset(token)
    sent = agent.run.await_args.kwargs.get("model_settings")
    assert (sent is not None) == (run_id is not None)
    if sent is not None:
        assert sent["extra_body"]["metadata"]["session_id"] == run_id


@pytest.mark.parametrize("field", sorted(f for f in Settings.model_fields if f.endswith("_model")))
def test_agent_model_defaults_are_real_litellm_groups(field):
    """claude-* names are aliases that LiteLLM silently maps to DeepSeek; defaults must name the real group."""
    assert Settings.model_fields[field].default in {"deepseek-v4-pro", "subagent"}


def test_transport_retries_default_to_one_layer():
    assert Settings.model_fields["agent_transport_attempts"].default == 1


async def test_run_id_reaches_spawned_tasks_but_not_a_fresh_context(monkeypatch):
    monkeypatch.setattr(settings, "model_provider", "litellm")
    token = run_id_var.set("pipe_abc")
    try:
        inherited = await asyncio.create_task(_settings_in_task())
        import contextvars

        detached = contextvars.Context().run(trace_model_settings, "ba")
    finally:
        run_id_var.reset(token)
    assert inherited["extra_body"]["metadata"]["session_id"] == "pipe_abc"
    assert detached is None


async def _settings_in_task():
    return trace_model_settings("ba")


class _FakeRepo:
    def __init__(self, latest):
        self.latest = latest

    def find_latest_run_id(self):
        return self.latest


@pytest.mark.parametrize(
    ("resume", "latest", "want_latest"),
    [(True, "run-earlier", True), (True, None, False), (False, "run-earlier", False)],
)
def test_resume_reuses_the_latest_run_id(monkeypatch, resume, latest, want_latest):
    monkeypatch.setattr(orch, "build_state_repository", lambda _project: _FakeRepo(latest))
    config = PipelineConfig(
        project="p", step="all", blueprint="new_feature", resume=resume, auto_yes=True, requirement="r"
    )
    pipeline = orch.OrgitrationPipeline(config)
    assert (pipeline.checkpoint_mgr.run_id == "run-earlier") is want_latest
    assert pipeline.checkpoint_mgr.run_id


def test_resume_without_a_checkpoint_warns(monkeypatch, caplog):
    monkeypatch.setattr(orch, "build_state_repository", lambda _project: _FakeRepo(None))
    config = PipelineConfig(
        project="p", step="all", blueprint="new_feature", resume=True, auto_yes=True, requirement="r"
    )
    with caplog.at_level("WARNING"):
        orch.OrgitrationPipeline(config)
    assert "no checkpoint exists" in caplog.text
