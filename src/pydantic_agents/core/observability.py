"""Centralised Logfire / observability setup.

Both `main.py` and `mcp_server.py` previously contained near-identical
configure-Logfire blocks. They are unified here so that fixes (e.g. enabling
pydantic-ai instrumentation, tightening httpx body capture) only need to be
made once, and so that a second call from a child entrypoint is a no-op.
"""

from __future__ import annotations

import contextlib
import logging
import os

import logfire

from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

_initialized = False

LOGFIRE_ENABLED = False
LOGFIRE_CONFIGURE_ERROR: str | None = None


def configure_observability() -> None:
    """Configure Logfire once per process.

    - Reads LOGFIRE_TOKEN from pydantic-settings (which loads .env at import
      time via SettingsConfigDict). Falls back to os.environ for direct usage.
    - Skips silently when no credentials are present (Logfire token or cached
      credentials file). The pipeline still runs; spans become no-ops.
    - Instruments pydantic-ai (per-model-call spans, token counts, tool calls),
      pydantic validators, and httpx — with request/response bodies suppressed
      so generated source code (read via tools) does not leave the host.
    - Idempotent: subsequent calls return immediately.
    """
    global _initialized, LOGFIRE_ENABLED, LOGFIRE_CONFIGURE_ERROR
    if _initialized:
        return
    _initialized = True

    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        LOGFIRE_CONFIGURE_ERROR = "OTEL_SDK_DISABLED is true — telemetry disabled"
        logger.info(LOGFIRE_CONFIGURE_ERROR)
        return

    logfire_token = os.environ.get("LOGFIRE_TOKEN", "") or getattr(settings, "logfire_token", "") or ""
    logfire_credentials = os.path.expanduser("~/.logfire/credentials.json")
    has_credentials = bool(logfire_token) or os.path.exists(logfire_credentials)

    if not has_credentials:
        LOGFIRE_CONFIGURE_ERROR = (
            "No Logfire token or credentials found — telemetry disabled (set LOGFIRE_TOKEN or run 'logfire auth')"
        )
        logger.info(LOGFIRE_CONFIGURE_ERROR)
        return

    try:
        logfire.configure(service_name="orchestration", inspect_arguments=False)
        logfire.instrument_pydantic()
        logfire.instrument_httpx(
            capture_request_headers=False,
            capture_response_headers=False,
            capture_request_body=False,
            capture_response_body=False,
        )
        # The instrumentation that actually matters for this codebase: emit
        # per-model-call spans (with token usage), per-tool-call spans, and
        # parent/child structure under the manually-opened mcp.* spans.
        try:
            logfire.instrument_pydantic_ai()
        except (AttributeError, ModuleNotFoundError):
            # Older Logfire (< ~3.x) lacks this helper; the env still benefits
            # from the other instrumentation. Skip silently.
            logger.debug("logfire.instrument_pydantic_ai() unavailable on this Logfire version")
        LOGFIRE_ENABLED = True
        logger.info("Logfire configured — pydantic-ai + pydantic + httpx instrumented")
    except TypeError:
        # Older Logfire signatures lack capture_*_body kwargs; retry without them.
        try:
            logfire.configure(service_name="orchestration", inspect_arguments=False)
            logfire.instrument_pydantic()
            logfire.instrument_httpx(capture_request_headers=False, capture_response_headers=False)
            with contextlib.suppress(AttributeError, ModuleNotFoundError):
                logfire.instrument_pydantic_ai()
            LOGFIRE_ENABLED = True
            logger.info(
                "Logfire configured (legacy mode — body capture flags unavailable; "
                "verify scrubbing is on for prompts that may contain secrets)"
            )
        except Exception as inner:
            LOGFIRE_CONFIGURE_ERROR = f"Logfire configuration failed (legacy retry): {inner}"
            logger.warning(LOGFIRE_CONFIGURE_ERROR)
    except Exception as e:
        LOGFIRE_CONFIGURE_ERROR = f"Logfire configuration failed: {e}"
        logger.warning(LOGFIRE_CONFIGURE_ERROR)
