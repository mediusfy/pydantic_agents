"""Shared run limits: one LLM concurrency limiter, usage caps, per-call model settings and the retry policy."""

import asyncio
import weakref
from contextvars import ContextVar
from typing import Any

from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.settings import ModelSettings

from pydantic_agents.core.settings import settings

# Agents that legitimately make many tool round-trips get a higher request cap.
_ROLE_REQUEST_LIMITS = {"backend": 60, "service_engineer": 60, "frontend": 60}

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Asyncio semaphores belong to one event loop, so keep one per loop.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()

# True while the current task already holds an LLM permit. A delegated child call runs under its parent's
# permit instead of acquiring a second one, so nested chains cannot exhaust the limiter and deadlock.
permit_held: ContextVar[bool] = ContextVar("llm_permit_held", default=False)

# Id of the run the current task belongs to: a pipeline run, or one MCP tool call. Sent to LiteLLM as the request's
# session id so Langfuse groups a run's model calls instead of showing them as unrelated generations, and set as the
# ``langfuse.session.id`` attribute on the spans this process exports. Tasks created after it is set inherit it
# (contextvars are copied into new tasks); work started from a thread pool or another loop does not, and then simply
# goes untagged.
run_id_var: ContextVar[str | None] = ContextVar("pipeline_run_id", default=None)


def run_model_settings(agent_name: str) -> ModelSettings | None:
    """Per-call model settings: the output cap, and the LiteLLM/Langfuse tags (session = run id, trace = agent).

    The tags are only sent with MODEL_PROVIDER=litellm and inside a run: other providers would receive an unknown
    body field. The output cap applies everywhere; 0 disables it.
    """
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


def usage_limits(agent_name: str | None = None) -> UsageLimits:
    """Per-run request and token caps for an agent."""
    request_limit = _ROLE_REQUEST_LIMITS.get(agent_name or "", settings.agent_request_limit)
    return UsageLimits(request_limit=request_limit, total_tokens_limit=settings.agent_total_tokens_limit)


def is_transient_error(exc: BaseException) -> bool:
    """Retry rate limits, 5xx and connection failures; never timeouts (a retry would just burn another full timeout)."""
    if isinstance(exc, TimeoutError):
        return False
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _RETRYABLE_STATUS
    return isinstance(exc, (ModelAPIError, ConnectionError))
