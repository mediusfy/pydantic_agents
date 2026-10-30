"""Regression tests for the Medium/Low findings of the 2026-10-29 full-repository review."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core import memory_bridge
from pydantic_agents.core.settings import Settings

# --- memory bridge: all_runtimes / max_entries / symlinks / status resilience ----------------------------------


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    claude = tmp_path / "claude" / "agents"
    implicit = tmp_path / "gemini" / "implicit"
    claude.mkdir(parents=True)
    implicit.mkdir(parents=True)
    monkeypatch.setattr(memory_bridge, "CLAUDE_AGENTS_DIR", claude)
    monkeypatch.setattr(memory_bridge, "ANTIGRAVITY_IMPLICIT_DIR", implicit)
    return claude, implicit


@pytest.fixture
def db(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(memory_bridge, "vector_db", fake)
    return fake


@pytest.mark.parametrize(("include", "expected_files"), [(True, 1), (False, 0)])
def test_sync_memory_to_claude_honours_include_antigravity(dirs, db, include, expected_files):
    claude, implicit = dirs
    (claude / "ba").mkdir()
    db.search.return_value = ["entry"]

    assert memory_bridge.sync_memory_to_claude(include_antigravity=include) == 1
    assert len(list(implicit.iterdir())) == expected_files


def test_sync_claude_to_pgvector_stops_at_max_entries(dirs, db):
    claude, _ = dirs
    for agent in ("a", "b"):
        mem = claude / agent / "agent-memory"
        mem.mkdir(parents=True)
        for i in range(3):
            (mem / f"n{i}.md").write_text(f"{agent}{i}")

    assert memory_bridge.sync_claude_to_pgvector(4) == 4
    assert db.add_documents.call_count == 4


def test_memory_files_skip_symlinks(dirs, db, tmp_path):
    claude, _ = dirs
    mem = claude / "ba" / "agent-memory"
    mem.mkdir(parents=True)
    outside = tmp_path / "secret.md"
    outside.write_text("do not ingest")
    (mem / "link.md").symlink_to(outside)
    (mem / "real.md").write_text("ok")

    assert [p.name for p in memory_bridge._memory_files(mem)] == ["real.md"]
    assert memory_bridge.sync_claude_to_pgvector() == 1


def test_get_bridge_status_survives_a_failing_vector_db(dirs, db):
    db.search.side_effect = RuntimeError("db down")
    status = memory_bridge.get_bridge_status()
    assert status["pgvector_tables"]["activity_logs"]["count"] == 0


def test_mcp_sync_memory_passes_both_arguments_through(monkeypatch):
    from pydantic_agents import mcp_server

    push, pull = MagicMock(return_value=2), MagicMock(return_value=3)
    monkeypatch.setattr(mcp_server, "sync_memory_to_claude", push)
    monkeypatch.setattr(mcp_server, "sync_claude_to_pgvector", pull)
    monkeypatch.setattr(mcp_server, "logfire", MagicMock())

    out = mcp_server.sync_memory(all_runtimes=False, max_entries=7)

    push.assert_called_once_with(max_entries=7, include_antigravity=False)
    pull.assert_called_once_with(7)
    assert "2 entries" in out
    assert "3 entries" in out


# --- healthcheck key sources -------------------------------------------------------------------------------------


def _hc_settings(**overrides):
    base = {
        "model_provider": "openai",
        "openai_api_key": None,
        "anthropic_api_key": None,
        "gemini_api_key": None,
        "deepseek_api_key": None,
        "litellm_virtual_key": None,
        "litellm_master_key": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    ("overrides", "ok"),
    [
        ({"model_provider": "litellm", "litellm_virtual_key": "sk-real-virtual"}, True),
        ({"model_provider": "litellm", "litellm_master_key": "sk-real-master"}, True),
        ({"model_provider": "litellm"}, False),
        ({"model_provider": "litellm", "litellm_virtual_key": "sk-..."}, False),
        ({"deepseek_api_key": "sk-real-deepseek"}, True),
        ({}, False),
    ],
)
def test_validate_api_keys_reads_settings_and_deepseek(monkeypatch, overrides, ok):
    from pydantic_agents.core import healthcheck

    for name in ("LITELLM_VIRTUAL_KEY", "LITELLM_MASTER_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(healthcheck, "settings", _hc_settings(**overrides))
    assert (healthcheck.validate_api_keys() == []) is ok


# --- model resolution --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model_name", "expected"),
    [
        ("deepseek", "deepseek-reasoner", "deepseek:deepseek-reasoner"),
        ("deepseek", "deepseek-v4-pro", "deepseek:deepseek-chat"),
        ("test", "deepseek-v4-pro", "test"),
    ],
)
def test_resolve_model_maps_deepseek_and_test(provider, model_name, expected):
    assert Settings(_env_file=None, model_provider=provider).resolve_model(model_name) == expected


# --- provider keeps an agent's own max_tokens --------------------------------------------------------------------


@pytest.mark.parametrize(("agent_ms", "expected"), [({"max_tokens": 8192}, 8192), (None, None)])
async def test_run_agent_keeps_the_agent_s_own_max_tokens(agent_ms, expected):
    import asyncio

    from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
    from pydantic_agents.core.settings import settings

    agent = MagicMock()
    agent.model_settings = agent_ms
    agent.run = AsyncMock(return_value="ok")
    await PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "database", agent, "hi")
    sent = agent.run.await_args.kwargs["model_settings"]
    assert sent["max_tokens"] == (expected or settings.agent_max_output_tokens)


async def test_the_real_database_agent_keeps_its_8192_cap(monkeypatch):
    """Against the real pydantic_ai Agent, so the attribute the provider reads is the one Agent actually sets."""
    import asyncio

    from pydantic_agents.agents.database import database_agent
    from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider

    assert database_agent.model_settings == {"max_tokens": 8192}
    run = AsyncMock(return_value="ok")
    monkeypatch.setattr(database_agent, "run", run)
    await PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "database", database_agent, "hi")
    assert run.await_args.kwargs["model_settings"]["max_tokens"] == 8192


# --- registration: default deps factories and the CTO's ADR tool -------------------------------------------------


@pytest.mark.parametrize(
    ("agent", "deps_cls_path"),
    [
        ("cto", "pydantic_agents.agents.cto.OrgContext"),
        ("backend", "pydantic_agents.agents.backend.ProjectState"),
        ("review", "pydantic_agents.agents.review.ReviewCriteria"),
        ("database", "pydantic_agents.agents.database.DatabaseDeps"),
        ("devops", "pydantic_agents.agents.devops.DevopsDeps"),
        ("security", "pydantic_agents.agents.security.SecurityDeps"),
    ],
)
def test_sdlc_agents_with_deps_type_have_a_default_deps_factory(agent, deps_cls_path):
    import importlib

    from pydantic_agents.agents.registration import register_all_agents
    from pydantic_agents.core.registry import registry

    register_all_agents()
    module, cls = deps_cls_path.rsplit(".", 1)
    factory = registry._agents[agent][2]
    assert callable(factory)
    assert isinstance(factory(), getattr(importlib.import_module(module), cls))


async def test_cto_search_existing_adrs(monkeypatch):
    from pydantic_agents.agents import cto

    search = AsyncMock(return_value=["ADR-1: use NATS"])
    monkeypatch.setattr(cto.vector_db, "async_search", search)
    assert "ADR-1" in await cto.search_existing_adrs(MagicMock(), "messaging", n_results=99)
    search.assert_awaited_once_with("adrs", "messaging", n_results=20)

    search.return_value = []
    assert "No ADRs found" in await cto.search_existing_adrs(MagicMock(), "x")

    search.side_effect = OSError("password authentication failed")
    out = await cto.search_existing_adrs(MagicMock(), "x")
    assert out.startswith("ADR search unavailable (OSError)")
    assert "password" not in out


# --- state repository, recorder close, cancel fallback -----------------------------------------------------------


async def test_cancel_pipeline_run_falls_back_to_the_run_store(monkeypatch):
    from pydantic_agents import mcp_server
    from pydantic_agents.core import pipeline_jobs

    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {})
    monkeypatch.setattr(pipeline_jobs, "get_run", AsyncMock(return_value={"status": "interrupted"}))
    out = await mcp_server.cancel_pipeline_run("pipe_aaaaaaaaaaaa")
    assert "already interrupted" in out

    monkeypatch.setattr(pipeline_jobs, "get_run", AsyncMock(return_value=None))
    assert "No pipeline run found" in await mcp_server.cancel_pipeline_run("pipe_bbbbbbbbbbbb")


async def test_existing_code_pipeline_closes_its_recorder(monkeypatch, tmp_path):
    from pydantic_agents.core import existing_code_handler as ech

    recorder = MagicMock()
    monkeypatch.setattr(ech, "build_stage_event_recorder", lambda _p: recorder)
    monkeypatch.setattr(ech, "_run_attempt", AsyncMock(return_value=True))
    monkeypatch.setattr(ech, "_write_result", lambda *_a: None)

    await ech.run_existing_code_pipeline("change", str(tmp_path))
    recorder.close.assert_called_once()


# --- session recorder truncation, observability content ---------------------------------------------------------


def test_session_record_truncates_long_prompts():
    from pydantic_agents.core.session_recorder import AgentRunRecord

    rec = AgentRunRecord("r", 0.0, "a", "s", "x" * 100_050, "", None, "")
    d = rec.to_dict()
    assert d["prompt_truncated"] is True
    assert len(d["prompt"]) == 100_000

    short = AgentRunRecord("r", 0.0, "a", "s", "short", "", None, "").to_dict()
    assert (short["prompt"], short["prompt_truncated"]) == ("short", False)


@pytest.mark.parametrize("older_logfire", [False, True])
def test_pydantic_ai_spans_exclude_content(older_logfire):
    from pydantic_agents.core import observability

    fake = MagicMock()
    if older_logfire:
        fake.instrument_pydantic_ai.side_effect = [TypeError("no include_content"), None]
    with patch.object(observability, "logfire", fake):
        observability._instrument()
    first = fake.instrument_pydantic_ai.call_args_list[0]
    assert first.kwargs == {"include_content": False}
    assert fake.instrument_pydantic_ai.call_count == (2 if older_logfire else 1)


# --- agy bridge: decorator hooks are live, server rejects non-object JSON ---------------------------------------


def test_bridge_agent_uses_hooks_registered_with_the_decorators():
    from pydantic_agents.agents.registration import register_all_agents
    from pydantic_agents.core.agy_bridge import hooks as hooks_mod
    from pydantic_agents.core.agy_bridge.bridge_agent import BridgeAgent

    register_all_agents()
    hooks_mod.clear_hooks("ux")
    try:

        @hooks_mod.pre_turn("ux")
        async def deny(_prompt):
            return None

        agent = BridgeAgent("ux")
        assert agent.hooks is hooks_mod.get_hooks("ux")
        assert len(agent.hooks.pre_turn) == 1
    finally:
        hooks_mod.clear_hooks("ux")


@pytest.mark.parametrize("payload", ["[1, 2]", '"ping"', "42", "null"])
async def test_bridge_server_rejects_non_object_commands(payload, capsys):
    from pydantic_agents.core.agy_bridge.server import _respond_to

    await _respond_to(payload)
    assert json.loads(capsys.readouterr().out) == {"error": "command must be a JSON object"}


# --- merge stage with a relative target root, sandbox_root resolution -------------------------------------------


async def test_merge_stage_handles_a_relative_target_root(tmp_path, monkeypatch):
    from pydantic_agents.core.models.core import Artifact, PipelineState
    from pydantic_agents.core.pipeline.config import PipelineConfig
    from pydantic_agents.core.stages.merge_stage import MergeStage

    project = tmp_path / "proj"
    (project / "backend").mkdir(parents=True)
    (project / "backend" / "main.go").write_text("old")
    monkeypatch.chdir(tmp_path)

    state = PipelineState(run_id="r", stage="quality_gate", requirement="x")
    state.artifacts = [Artifact(agent_name="backend", files={"main.go": "new"})]
    config = PipelineConfig(dry_run=False)
    config.target_root = "proj"  # relative on purpose
    config.project = "proj"

    with patch("pydantic_agents.core.context_engine.store_artifact_context"):
        stage = MergeStage()
        await stage.run(state, config)

    assert stage.result.errors == []
    assert (project / "backend" / "main.go").read_text() == "new"
    assert stage.result.backups_created == 1


def test_sandbox_root_resolves_a_relative_context_root(tmp_path, monkeypatch):
    from pydantic_agents.core.utils import resolve_sandboxed, sandbox_root, sandbox_root_var

    (tmp_path / "code").mkdir()
    monkeypatch.chdir(tmp_path)
    token = sandbox_root_var.set(Path("code"))
    try:
        assert sandbox_root() == (tmp_path / "code").resolve()
        assert resolve_sandboxed("a.go") == (tmp_path / "code" / "a.go").resolve()
    finally:
        sandbox_root_var.reset(token)


# --- seed_projects: name validation, safe join, timeouts --------------------------------------------------------


@pytest.mark.parametrize("bad", ["../etc", "a/b", "..", ".", "", "x y"])
def test_seed_projects_rejects_unsafe_repo_names(bad, tmp_path):
    from scripts import seed_projects

    with pytest.raises(ValueError):
        seed_projects.sync_clone("mediusfy", bad, tmp_path)


def test_seed_projects_sync_clone_uses_timeouts(tmp_path, monkeypatch):
    from scripts import seed_projects

    run = MagicMock()
    monkeypatch.setattr(seed_projects.subprocess, "run", run)
    dest = seed_projects.sync_clone("mediusfy", "repo", tmp_path)
    assert dest == (tmp_path / "repo").resolve()
    assert run.call_args.kwargs["timeout"] == 120

    (dest / ".git").mkdir(parents=True)
    seed_projects.sync_clone("mediusfy", "repo", tmp_path)
    assert run.call_args.args[0][:3] == ["git", "-C", str(dest)]
    assert run.call_args.kwargs["timeout"] == 120


def test_seed_projects_active_repos_validates_names_and_times_out(monkeypatch):
    from scripts import seed_projects

    out = json.dumps(
        [
            {"name": "good", "pushedAt": "2999-01-01T00:00:00Z", "isArchived": False},
            {"name": "../evil", "pushedAt": "2999-01-01T00:00:00Z", "isArchived": False},
        ]
    )
    check_output = MagicMock(return_value=out)
    monkeypatch.setattr(seed_projects.subprocess, "check_output", check_output)
    with pytest.raises(ValueError):
        seed_projects.active_repos("mediusfy", 31, set())
    assert check_output.call_args.kwargs["timeout"] == 60

    with pytest.raises(ValueError):
        seed_projects.active_repos("bad org", 31, set())


def test_check_changelog_git_show_has_a_timeout(monkeypatch):
    from scripts import check_changelog

    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="x"))
    monkeypatch.setattr(check_changelog.subprocess, "run", run)
    assert check_changelog._base_changelog("origin/main") == "x"
    assert run.call_args.kwargs["timeout"] == 30


# --- vector_db ingest ids are deterministic ---------------------------------------------------------------------


def test_ingest_pdf_ids_are_deterministic(monkeypatch):
    pytest.importorskip("langchain_community")
    pytest.importorskip("langchain_text_splitters")
    from pydantic_agents.core import vector_db as vdb

    class _Doc:
        def __init__(self, text):
            self.page_content = text
            self.metadata = {}

    monkeypatch.setattr("langchain_community.document_loaders.PyPDFLoader", lambda _p: SimpleNamespace(load=lambda: []))

    class _Splitter:
        def __init__(self, **_k):
            pass

        def split_documents(self, _docs):
            return [_Doc("alpha"), _Doc("beta")]

    monkeypatch.setattr("langchain_text_splitters.RecursiveCharacterTextSplitter", _Splitter)
    monkeypatch.setattr(vdb, "_require_vector_deps", lambda: None)
    seen: list[list[str]] = []

    async def fake_add(_self, _c, _t, _m, ids):
        seen.append(ids)

    monkeypatch.setattr(vdb.VectorDBManager, "_add_documents", fake_add)
    manager = vdb.VectorDBManager()
    manager.ingest_pdf("docs", "/tmp/a.pdf")
    manager.ingest_pdf("docs", "/tmp/a.pdf")

    assert seen[0] == seen[1]
    assert len(set(seen[0])) == 2
    assert all(i.startswith("pdf_") and len(i) == len("pdf_") + 16 for i in seen[0])


# --- vector_db.ingest_url: SSRF guard and deterministic ids ------------------------------------------------------


def _stub_ingest(monkeypatch, vdb, texts):
    class _Doc:
        def __init__(self, text):
            self.page_content = text

    class _Splitter:
        def __init__(self, **_k):
            pass

        def split_documents(self, _docs):
            return [_Doc(t) for t in texts]

    loaders = []

    def loader(url, requests_kwargs=None):
        loaders.append((url, requests_kwargs))
        return SimpleNamespace(load=lambda: [])

    monkeypatch.setattr("langchain_community.document_loaders.WebBaseLoader", loader)
    monkeypatch.setattr("langchain_text_splitters.RecursiveCharacterTextSplitter", _Splitter)
    monkeypatch.setattr(vdb, "_require_vector_deps", lambda: None)
    seen: list[list[str]] = []

    async def fake_add(_self, _c, _t, _m, ids):
        seen.append(ids)

    monkeypatch.setattr(vdb.VectorDBManager, "_add_documents", fake_add)
    return loaders, seen


async def test_ingest_url_validates_the_url_and_uses_deterministic_ids(monkeypatch):
    pytest.importorskip("langchain_community")
    pytest.importorskip("langchain_text_splitters")
    from pydantic_agents.core import vector_db as vdb
    from pydantic_agents.core.tools import web_fetch

    loaders, seen = _stub_ingest(monkeypatch, vdb, ["alpha", "beta"])
    monkeypatch.setattr(web_fetch, "validate_public_http_url", lambda url, **_k: "blocked" if "10." in url else None)
    manager = vdb.VectorDBManager()

    with pytest.raises(ValueError, match="blocked"):
        manager.ingest_url("docs", "http://10.0.0.1/")
    with pytest.raises(ValueError, match="blocked"):
        await manager.async_ingest_url("docs", "http://10.0.0.1/")
    assert loaders == []

    manager.ingest_url("docs", "https://example.com/a")
    await manager.async_ingest_url("docs", "https://example.com/a")
    assert seen[0] == seen[1]
    assert all(i.startswith("url_") for i in seen[0])
    assert loaders[0][1] == {"timeout": 15, "allow_redirects": False}


# --- pydantic_ai_compat fails loudly on private attribute drift --------------------------------------------------


def test_pydantic_ai_compat_fails_loudly_when_agent_internals_change(monkeypatch):
    from pydantic_ai import Agent

    from pydantic_agents.core.exceptions import PydanticAICompatibilityError
    from pydantic_agents.core.pydantic_ai_compat import (
        agent_tools,
        check_pydantic_ai_compatibility,
        tool_function,
    )

    check_pydantic_ai_compatibility.cache_clear()
    check_pydantic_ai_compatibility()

    real_agent = Agent("test", defer_model_check=True)
    monkeypatch.delattr(real_agent, "_function_toolset")
    with pytest.raises(PydanticAICompatibilityError, match="_function_toolset"):
        agent_tools(real_agent)

    no_function = SimpleNamespace(function=None)
    with pytest.raises(PydanticAICompatibilityError, match="callable"):
        tool_function(no_function)
    plain_object = object()
    with pytest.raises(PydanticAICompatibilityError, match="callable"):
        tool_function(plain_object)
