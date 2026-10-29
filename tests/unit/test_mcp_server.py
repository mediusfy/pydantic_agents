from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.tools.secrets import scan_for_secrets
from pydantic_agents.core.tools.web_fetch import fetch_url, validate_public_http_url
from pydantic_agents.core.vector_db import _ensure_table, vector_db
from pydantic_agents.mcp_server import (
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


@pytest.mark.parametrize(
    "bad_collection",
    [
        "adrs; DROP TABLE pipeline_runs; --",
        "adrs--",
        "123bad",
        "../escape",
        "",
        "a" * 65,
    ],
)
@pytest.mark.asyncio
async def test_vector_db_rejects_invalid_collection_names(bad_collection):
    with pytest.raises(ValueError, match="Invalid collection name"):
        await _ensure_table(bad_collection)
    with pytest.raises(ValueError, match="Invalid collection name"):
        await vector_db._search(bad_collection, "query")
    with pytest.raises(ValueError, match="Invalid collection name"):
        await vector_db._add_documents(bad_collection, ["doc"], [{}], ["id1"])


@pytest.mark.parametrize(
    "blocked_url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1:8080/admin",
        "http://localhost:4000/v1/models",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/status",
        "http://[::1]:8000/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://service.internal/api",
    ],
)
def test_validate_public_http_url_blocks_ssrf_targets(blocked_url):
    err = validate_public_http_url(blocked_url, resolve_dns=False)
    assert err is not None
    assert err.startswith("Error:")


def test_validate_public_http_url_allows_public_ip():
    assert validate_public_http_url("https://93.184.216.34/index.html", resolve_dns=False) is None


@pytest.mark.asyncio
async def test_fetch_url_blocks_redirect_to_loopback(monkeypatch):
    import httpx

    monkeypatch.setattr(
        "pydantic_agents.core.tools.web_fetch.validate_public_http_url",
        lambda u, **_kw: validate_public_http_url(u, resolve_dns=False),
    )

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, **_kwargs):
            req = httpx.Request("GET", url)
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"}, request=req)

    monkeypatch.setattr("pydantic_agents.core.tools.web_fetch.httpx.AsyncClient", _FakeClient)
    out = await fetch_url(MagicMock(), "https://93.184.216.34/start")
    assert out.startswith("Error: Access to non-public IP address")


def test_scan_for_secrets_redacts_matches_and_catches_expanded_patterns():
    sample = "\n".join(
        [
            "OPENAI_API_KEY = sk-proj-abcdefghijklmnopqrstuvwxyz123456",
            "api_key = my_super_secret_token_value_12345",
            "DATABASE_URL=postgresql://admin:s3cr3tP@ss@db.prod.internal:5432/app-db",
            "REDIS_URL=rediss://default:r3d1sPass@cache-01.internal:6379",
        ]
    )
    findings = scan_for_secrets(MagicMock(), sample)
    types = {f["type"] for f in findings}
    assert {"llm_api_key", "api_key_generic", "pg_connection_string", "redis_connection_string"} <= types
    for finding in findings:
        assert "s3cr3tP@ss" not in finding["snippet"]
        assert "r3d1sPass" not in finding["snippet"]
        assert "abcdefghijklmnopqrstuvwxyz123456" not in finding["snippet"]
        assert "my_super_secret_token_value_12345" not in finding["snippet"]
        assert "[REDACTED]" in finding["snippet"]


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
