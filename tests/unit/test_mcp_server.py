from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.mcp_server import (
    _check_codebase_root,
    mcp,
    modify_existing_code,
    query_context,
    run_agent_by_name,
    scaffold_and_build_service,
)


@pytest.fixture
def mock_registry():
    with patch("pydantic_agents.mcp_server.registry") as mock_reg:
        yield mock_reg


@pytest.mark.asyncio
async def test_run_agent_by_name_local(mock_registry):
    # Setup mock agent
    mock_agent = AsyncMock()
    mock_result = MagicMock()
    mock_result.output = "local_result_data"
    mock_agent.run.return_value = mock_result

    mock_registry.is_registered.return_value = True
    mock_registry.run = AsyncMock(return_value=mock_result)

    with patch("pydantic_agents.mcp_server.retrieve_context", AsyncMock(return_value="some_context")):
        result = await run_agent_by_name("test_agent", "do local work")

    assert result == "local_result_data"
    mock_registry.run.assert_awaited_once()
    prompt = mock_registry.run.await_args.args[1]
    assert prompt.startswith("Relevant Architecture Context from Knowledge Base:\nsome_context")
    assert prompt.endswith("do local work")


def test_mcp_transport_security_populates_default_allowed_hosts():
    ts = mcp.settings.transport_security
    assert ts is not None
    assert "127.0.0.1:*" in ts.allowed_hosts
    assert "localhost:*" in ts.allowed_hosts
    assert "[::1]:*" in ts.allowed_hosts


def test_query_context_rejects_unknown_or_injected_domain():
    out = query_context("auth", domain="adrs; DROP TABLE pipeline_runs; --")
    assert out.startswith("Invalid domain")


@pytest.mark.asyncio
async def test_modify_existing_code_rejects_path_outside_allowed_roots(tmp_path, monkeypatch):
    from pydantic_agents.core.settings import settings

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(settings, "seed_roots", str(allowed))
    monkeypatch.delenv("ORCHESTRATION_CODEBASE_ROOT", raising=False)

    out = await modify_existing_code("add logging", codebase_path=str(outside))
    assert out.startswith("Error: codebase_path is outside")


def test_check_codebase_root_honours_orchestration_codebase_root_env(tmp_path, monkeypatch):
    from pydantic_agents.core.exceptions import PathTraversalError
    from pydantic_agents.core.settings import settings

    env_root = tmp_path / "env_allowed"
    sub = env_root / "service_a"
    sub.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setattr(settings, "seed_roots", "")
    monkeypatch.setenv("ORCHESTRATION_CODEBASE_ROOT", str(env_root))

    assert _check_codebase_root(str(sub)) == sub.resolve()

    with pytest.raises(PathTraversalError, match="codebase_path is outside"):
        _check_codebase_root(str(outside))


@pytest.mark.asyncio
async def test_scaffold_and_build_service_validates_name_and_uses_build_pipeline():
    bad = await scaffold_and_build_service("../escape", "billing")
    assert bad.startswith("Error: Invalid project_name")

    fake_pipeline = MagicMock(execute=AsyncMock())
    with patch("pydantic_agents.core.pipeline.factory.build_pipeline", return_value=fake_pipeline) as build:
        ok = await scaffold_and_build_service("billing-svc", "billing")
    build.assert_called_once()
    fake_pipeline.execute.assert_awaited_once()
    assert "billing-svc" in ok
