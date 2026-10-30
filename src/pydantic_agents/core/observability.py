"""Centralised Logfire / OpenTelemetry setup.

One ``configure_observability()`` call per process. Spans go to whichever backends are configured:

- **Langfuse** (recommended in the cluster): set ``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY`` and
  ``LANGFUSE_SECRET_KEY``; the whole in-process tree (``mcp.*``, ``pipeline.*``, ``agent.run``, ``tool.*``,
  ``a2a.delegate`` and pydantic-ai's GenAI spans with token counts) is exported over OTLP/HTTP to
  ``{LANGFUSE_HOST}/api/public/otel``. ``instrument_httpx`` injects ``traceparent`` into the requests to LiteLLM,
  so LiteLLM's ``langfuse_otel`` generations attach under the ``agent.run`` span of the agent that made them.
- **Logfire cloud**: set ``LOGFIRE_TOKEN`` (or run ``logfire auth``).

Both can be on at once. With neither, spans are still created (one per agent run is cheap) but exported nowhere;
the pipeline still runs. Before this module exported
to Langfuse, the agents pod (which has no Logfire token) dropped every in-process span, leaving only LiteLLM's
per-generation observations without parents.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os

import logfire

from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

_initialized = False

LOGFIRE_ENABLED = False
LANGFUSE_ENABLED = False
LOGFIRE_CONFIGURE_ERROR: str | None = None

SERVICE_NAME = "pydantic-agents"


def _langfuse_otlp() -> tuple[str, dict[str, str]] | None:
    """The OTLP traces endpoint and auth header for Langfuse, when its three environment variables are set."""
    host = os.environ.get("LANGFUSE_HOST", "").rstrip("/")
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not (host and public and secret):
        return None
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    return f"{host}/api/public/otel/v1/traces", {"Authorization": f"Basic {auth}"}


def _langfuse_processor():
    """A batch span processor exporting to Langfuse, or None when Langfuse is not configured."""
    target = _langfuse_otlp()
    if target is None:
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint, headers = target
    return BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))


def _instrument() -> None:
    """Instrument pydantic, httpx (bodies suppressed so generated code never leaves the host) and pydantic-ai."""
    logfire.instrument_pydantic()
    try:
        logfire.instrument_httpx(
            capture_request_headers=False,
            capture_response_headers=False,
            capture_request_body=False,
            capture_response_body=False,
        )
    except TypeError:  # older Logfire signatures lack the capture_*_body kwargs
        logfire.instrument_httpx(capture_request_headers=False, capture_response_headers=False)
    # The instrumentation that matters most: per-model-call spans (with token usage), per-tool-call spans, and
    # parent/child structure under the manually opened mcp.* / pipeline.* / agent.run spans.
    with contextlib.suppress(AttributeError, ModuleNotFoundError):
        try:
            logfire.instrument_pydantic_ai(include_content=False)
        except TypeError:
            logfire.instrument_pydantic_ai()


def configure_observability() -> None:
    """Configure Logfire / OTel once per process (see the module docstring). Idempotent."""
    global _initialized, LOGFIRE_ENABLED, LANGFUSE_ENABLED, LOGFIRE_CONFIGURE_ERROR
    if _initialized:
        return
    _initialized = True

    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        LOGFIRE_CONFIGURE_ERROR = "OTEL_SDK_DISABLED is true — telemetry disabled"
        logger.info(LOGFIRE_CONFIGURE_ERROR)
        return

    logfire_token = os.environ.get("LOGFIRE_TOKEN", "") or getattr(settings, "logfire_token", "") or ""
    logfire_credentials = os.path.expanduser("~/.logfire/credentials.json")
    send_to_logfire = bool(logfire_token) or os.path.exists(logfire_credentials)
    try:
        langfuse_processor = _langfuse_processor()
    except Exception as e:  # the exporter package is missing or the endpoint is malformed
        logger.warning("Langfuse OTLP export unavailable: %s", e)
        langfuse_processor = None

    if not send_to_logfire and langfuse_processor is None:
        LOGFIRE_CONFIGURE_ERROR = (
            "No Logfire token or Langfuse keys found — telemetry disabled "
            "(set LANGFUSE_HOST/LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY, or LOGFIRE_TOKEN, or run 'logfire auth')"
        )
        logger.info(LOGFIRE_CONFIGURE_ERROR)
        return

    try:
        logfire.configure(
            service_name=SERVICE_NAME,
            inspect_arguments=False,
            send_to_logfire=send_to_logfire,
            additional_span_processors=[langfuse_processor] if langfuse_processor else None,
        )
        _instrument()
        LOGFIRE_ENABLED = send_to_logfire
        LANGFUSE_ENABLED = langfuse_processor is not None
        logger.info(
            "Observability configured (logfire=%s, langfuse=%s) — pydantic-ai + pydantic + httpx instrumented",
            LOGFIRE_ENABLED,
            LANGFUSE_ENABLED,
        )
    except Exception as e:
        LOGFIRE_CONFIGURE_ERROR = f"Logfire configuration failed: {e}"
        logger.warning(LOGFIRE_CONFIGURE_ERROR)
