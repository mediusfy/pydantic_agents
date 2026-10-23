"""Shared run limits: one LLM concurrency limiter, usage caps and the retry policy."""

import asyncio
import weakref
from contextvars import ContextVar

from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from pydantic_agents.core.settings import settings

# Agents that legitimately make many tool round-trips get a higher request cap.
_ROLE_REQUEST_LIMITS = {"backend": 60, "service_engineer": 60, "frontend": 60}

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Asyncio semaphores belong to one event loop, so keep one per loop.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()

# True while the current task already holds an LLM permit. A delegated child call runs under its parent's
# permit instead of acquiring a second one, so nested chains cannot exhaust the limiter and deadlock.
permit_held: ContextVar[bool] = ContextVar("llm_permit_held", default=False)


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
