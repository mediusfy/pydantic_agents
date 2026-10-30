"""Tests for the consolidated observability module.

Verifies:
- configure_observability() is idempotent (safe to call from multiple entrypoints
  — main.py, mcp_server.py, and a child invocation via run_orchestration_pipeline).
- When no credentials are present, it sets the disabled flag instead of crashing.
- pydantic-ai instrumentation is attempted (when the helper exists on the
  installed Logfire version).
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_observability_state(monkeypatch):
    """Each test starts with a fresh module-level state."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    from pydantic_agents.core import observability

    observability._initialized = False
    observability.LOGFIRE_ENABLED = False
    observability.LOGFIRE_CONFIGURE_ERROR = None
    yield
    observability._initialized = False


def _clear_logfire_env(monkeypatch):
    """Clear all Logfire credential sources for deterministic test setup.

    The observability module imports a pydantic-settings singleton that loads
    from .env at import time. We must also clear that singleton's value so
    configure_observability() sees no credentials.
    """
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    monkeypatch.setattr("os.path.exists", lambda _: False)
    import pydantic_agents.core.observability as _obs_mod

    _obs_mod.settings.logfire_token = ""


def test_configure_is_a_noop_when_no_credentials(tmp_path, monkeypatch):
    _clear_logfire_env(monkeypatch)

    from pydantic_agents.core import observability

    observability.configure_observability()

    assert observability.LOGFIRE_ENABLED is False
    assert observability.LOGFIRE_CONFIGURE_ERROR is not None
    assert "telemetry disabled" in observability.LOGFIRE_CONFIGURE_ERROR


def test_configure_is_idempotent(monkeypatch):
    _clear_logfire_env(monkeypatch)

    from pydantic_agents.core import observability

    observability.configure_observability()
    first_state = observability._initialized
    # Second invocation must be a no-op (the early-return on _initialized=True
    # is the post-fix idempotency contract).
    observability.configure_observability()
    assert first_state is True
    assert observability._initialized is True


def test_configure_calls_instrument_pydantic_ai_when_credentials_present(monkeypatch):
    monkeypatch.setenv("LOGFIRE_TOKEN", "fake-token")

    fake_logfire = MagicMock()
    with patch("pydantic_agents.core.observability.logfire", fake_logfire):
        from pydantic_agents.core import observability

        observability.configure_observability()

    fake_logfire.configure.assert_called_once()
    fake_logfire.instrument_pydantic_ai.assert_called_once()
    fake_logfire.instrument_pydantic.assert_called_once()
    fake_logfire.instrument_httpx.assert_called_once()


def test_configure_tolerates_missing_instrument_pydantic_ai(monkeypatch):
    """Older Logfire versions lack instrument_pydantic_ai(); that must not
    fail the boot."""
    monkeypatch.setenv("LOGFIRE_TOKEN", "fake-token")

    fake_logfire = MagicMock()
    # Make the helper raise AttributeError to simulate an older Logfire.
    fake_logfire.instrument_pydantic_ai.side_effect = AttributeError("no such attribute")
    with patch("pydantic_agents.core.observability.logfire", fake_logfire):
        from pydantic_agents.core import observability

        observability.configure_observability()

    assert observability.LOGFIRE_ENABLED is True


@pytest.mark.parametrize(
    ("env", "expect_problem"),
    [
        ({}, True),
        ({"LOGFIRE_TOKEN": "t"}, False),
        ({"LANGFUSE_HOST": "https://l", "LANGFUSE_PUBLIC_KEY": "p", "LANGFUSE_SECRET_KEY": "s"}, False),
        ({"LANGFUSE_HOST": "https://l", "LANGFUSE_PUBLIC_KEY": "p"}, True),
        ({"OTEL_SDK_DISABLED": "true"}, False),
    ],
)
def test_telemetry_backend_problem(monkeypatch, env, expect_problem):
    for key in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    _clear_logfire_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from pydantic_agents.core import observability

    problem = observability.telemetry_backend_problem()
    assert (problem is not None) is expect_problem
    if expect_problem:
        assert "no telemetry backend configured" in problem


def test_no_backend_logs_a_warning(monkeypatch, caplog):
    for key in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    _clear_logfire_env(monkeypatch)
    from pydantic_agents.core import observability

    with caplog.at_level("INFO", logger=observability.logger.name):
        observability.configure_observability()

    records = [r for r in caplog.records if "telemetry disabled" in r.getMessage()]
    assert records
    assert all(r.levelname == "WARNING" for r in records)
