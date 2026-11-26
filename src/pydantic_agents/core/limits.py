"""Shared run limits: LLM concurrency limiter, per-agent & pipeline usage caps, model settings & retry policy."""

import asyncio
import weakref
from contextvars import ContextVar
from typing import Any

from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UsageLimitExceeded
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from pydantic_agents.core.settings import settings

# Agents that legitimately make many tool round-trips get a higher request cap.
_ROLE_REQUEST_LIMITS = {"backend": 60, "service_engineer": 60, "frontend": 60}

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Asyncio semaphores belong to one event loop, so keep one per loop.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()

# True while the current task already holds an LLM permit. A delegated child call runs under its parent's
# permit instead of acquiring a second one, so nested chains cannot exhaust the limiter and deadlock; parallel
# child fan-out from one parent is bounded separately by ``settings.max_concurrent_child_delegations``.
permit_held: ContextVar[bool] = ContextVar("llm_permit_held", default=False)

# Id of the run the current task belongs to: a pipeline run, or one MCP tool call.
run_id_var: ContextVar[str | None] = ContextVar("pipeline_run_id", default=None)

# Cumulative RunUsage accumulator shared across all stages and retries of an active pipeline run.
pipeline_usage_var: ContextVar[RunUsage | None] = ContextVar("pipeline_run_usage", default=None)


def run_model_settings(agent_name: str) -> ModelSettings | None:
    """Per-call model settings: the output cap, and the LiteLLM/Langfuse tags (session = run id, trace = agent)."""
    ms: dict[str, Any] = {}
    if settings.agent_max_output_tokens > 0:
        ms["max_tokens"] = settings.agent_max_output_tokens
    run_id = run_id_var.get()
    if run_id is not None and settings.model_provider == "litellm":
        ms["extra_body"] = {"metadata": {"session_id": run_id, "trace_name": f"pydantic-agents/{agent_name}"}}
    return ModelSettings(**ms) if ms else None  # type: ignore[typeddict-item]


def trace_model_settings(agent_name: str) -> ModelSettings | None:
    """The LiteLLM/Langfuse tags alone (kept for callers that only want the trace metadata)."""
    run_id = run_id_var.get()
    if run_id is None or settings.model_provider != "litellm":
        return None
    return ModelSettings(extra_body={"metadata": {"session_id": run_id, "trace_name": f"pydantic-agents/{agent_name}"}})


def llm_semaphore() -> asyncio.Semaphore:
    """The process-wide (per event loop) limiter that every LLM call shares."""
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = _semaphores[loop] = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    return sem


def check_pipeline_budget(pipeline_usage: RunUsage | None) -> None:
    """Raise ``UsageLimitExceeded`` if the active pipeline run has exhausted its cumulative token or request ceiling."""
    if pipeline_usage is None:
        return
    tok_cap = settings.pipeline_total_tokens_limit
    if tok_cap > 0 and pipeline_usage.total_tokens >= tok_cap:
        raise UsageLimitExceeded(
            f"Pipeline token budget exhausted ({pipeline_usage.total_tokens:,} >= {tok_cap:,} tokens, "
            "pipeline_total_tokens_limit)."
        )
    req_cap = settings.pipeline_request_limit
    if req_cap > 0 and pipeline_usage.requests >= req_cap:
        raise UsageLimitExceeded(
            f"Pipeline request budget exhausted ({pipeline_usage.requests:,} >= {req_cap:,} requests, "
            "pipeline_request_limit)."
        )


def usage_limits(agent_name: str | None = None, *, pipeline_usage: RunUsage | None = None) -> UsageLimits:
    """Per-run request and token caps for an agent, clamped to any remaining pipeline-wide budget."""
    check_pipeline_budget(pipeline_usage)
    request_limit = _ROLE_REQUEST_LIMITS.get(agent_name or "", settings.agent_request_limit)
    token_limit = settings.agent_total_tokens_limit

    if pipeline_usage is not None:
        if settings.pipeline_request_limit > 0:
            remaining_reqs = max(1, settings.pipeline_request_limit - pipeline_usage.requests)
            request_limit = min(request_limit, remaining_reqs)
        if settings.pipeline_total_tokens_limit > 0:
            remaining_toks = max(1, settings.pipeline_total_tokens_limit - pipeline_usage.total_tokens)
            token_limit = min(token_limit, remaining_toks)
    return UsageLimits(request_limit=request_limit, total_tokens_limit=token_limit)


def record_pipeline_usage(pipeline_usage: RunUsage | None, result: Any, *, shared_usage: Any = None) -> None:
    """Fold a completed agent call's usage into ``pipeline_usage`` (no-op when ``shared_usage`` already updated it)."""
    if pipeline_usage is None or result is None or shared_usage is pipeline_usage:
        return
    try:
        call_usage = result.usage()
    except Exception:
        return
    in_tok = getattr(call_usage, "input_tokens", 0)
    out_tok = getattr(call_usage, "output_tokens", 0)
    reqs = getattr(call_usage, "requests", 0)
    if isinstance(in_tok, int) and in_tok > 0:
        pipeline_usage.input_tokens += in_tok
    if isinstance(out_tok, int) and out_tok > 0:
        pipeline_usage.output_tokens += out_tok
    if isinstance(reqs, int) and reqs > 0:
        pipeline_usage.requests += reqs


def is_transient_error(exc: BaseException) -> bool:
    """Retry rate limits, 5xx and connection failures; never timeouts (a retry would just burn another full timeout)."""
    if isinstance(exc, TimeoutError):
        return False
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _RETRYABLE_STATUS
    return isinstance(exc, (ModelAPIError, ConnectionError))
