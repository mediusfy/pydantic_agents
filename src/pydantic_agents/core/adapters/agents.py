import asyncio
import contextlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pydantic_agents.core import platform_metrics
from pydantic_agents.core.limits import is_transient_error, permit_held, run_id_var, run_model_settings, usage_limits
from pydantic_agents.core.settings import settings
from pydantic_agents.core.telemetry import telemetry

logger = logging.getLogger(__name__)

# Share of the character budget kept from the head of an oversized prompt; the rest is kept from the tail. The tail
# weighs more because the assembled input puts retrieved context first and the actual task last (see
# mcp_server.run_agent_by_name), and a model that loses the task answers the wrong question confidently.
_HEAD_SHARE = 0.3


def prompt_char_budget() -> int:
    """The character budget for one prompt, or 0 when the cap is switched off.

    ``agent_prompt_token_budget`` is a token figure because that is what the model bills and what the incident was
    measured in (a 555k-input-token ``cto`` run), but nothing here can tokenize text without loading a tokenizer per
    model, so it is converted with the ``agent_prompt_chars_per_token`` estimate. A non-positive value for either
    setting disables the cap.
    """
    budget = settings.agent_prompt_token_budget
    ratio = settings.agent_prompt_chars_per_token
    if budget <= 0 or ratio <= 0:
        return 0
    return int(budget * ratio)


def cap_prompt(agent_name: str, prompt: str) -> str:
    """``prompt`` unchanged when it fits the budget, else truncated in the middle with a notice where the cut is.

    This is the last point before ``Agent.run``, so it catches every path that assembles input — MCP context
    enrichment, pipeline artifact dumps, tool output accumulated over a run — instead of each of them needing its
    own limit. The notice is part of the prompt on purpose: a model given a silently shortened context cannot tell
    that it is reasoning about a partial view, and would report what is missing as absent.
    """
    budget = prompt_char_budget()
    if budget <= 0 or len(prompt) <= budget:
        return prompt
    omitted = len(prompt) - budget
    notice = (
        f"\n\n... (TRUNCATED at the {budget}-character prompt budget (agent_prompt_token_budget): "
        f"{omitted} characters from the middle of this input are not shown. "
        f"This is an INCOMPLETE view; say so instead of treating what is missing as absent.)\n\n"
    )
    # The notice spends part of the budget rather than being added on top of it, so the result never exceeds it.
    keep = max(0, budget - len(notice))
    head_len = int(keep * _HEAD_SHARE)
    tail_len = keep - head_len
    logger.warning(
        "agent '%s' prompt is %d characters, over the %d-character budget (agent_prompt_token_budget=%d): "
        "keeping %d characters, %d omitted",
        agent_name,
        len(prompt),
        budget,
        settings.agent_prompt_token_budget,
        keep,
        omitted,
    )
    head = prompt[:head_len]
    tail = prompt[len(prompt) - tail_len :] if tail_len else ""
    return f"{head}{notice}{tail}"


async def _record_metric(agent_name: str, start: float, result: Any) -> None:
    """One platform_metrics row per agent run (duration, tokens, verdict); best-effort and never raises.

    ``result is None`` means the run failed or was cancelled.
    """
    in_tokens, out_tokens = platform_metrics.token_counts(result) if result is not None else (None, None)
    total_tokens = (in_tokens or 0) + (out_tokens or 0)
    if total_tokens > settings.agent_token_alert_threshold:
        logger.warning(
            "agent '%s' run used %d tokens (in=%s out=%s), over the %d alert threshold (agent_token_alert_threshold)",
            agent_name,
            total_tokens,
            in_tokens,
            out_tokens,
            settings.agent_token_alert_threshold,
        )
    with contextlib.suppress(Exception):
        await platform_metrics.record_call(
            agent_name,
            run_id=run_id_var.get() or None,
            duration=time.time() - start,
            success=result is not None,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=platform_metrics.estimate_cost_usd(agent_name, in_tokens, out_tokens),
            verdict=platform_metrics.verdict_of(result) if result is not None else None,
        )


class AgentProviderPort(ABC):
    """Abstracts the interaction with the underlying LLM framework."""

    @abstractmethod
    async def run_agent(
        self,
        semaphore: asyncio.Semaphore,
        agent_name: str,
        agent: Any,
        prompt: str,
        deps: Any = None,
        timeout_seconds: float | None = None,
        usage: Any = None,
    ) -> Any:
        pass


class PydanticAIAgentProvider(AgentProviderPort):
    """Concrete implementation using pydantic_ai for agent execution.

    This is the only place that calls ``Agent.run``: it applies the concurrency permit, the per-run usage caps, the
    timeout, the transport retry, the per-call model settings (output cap, Langfuse session tags), the JSONL
    telemetry and an ``agent.run`` span tagged for Langfuse. ``usage`` lets a delegated call share its caller's
    budget instead of getting a fresh one. It is also where the prompt character budget is enforced (``cap_prompt``).
    """

    async def run_agent(
        self,
        semaphore: asyncio.Semaphore,
        agent_name: str,
        agent: Any,
        prompt: str,
        deps: Any = None,
        timeout_seconds: float | None = None,
        usage: Any = None,
    ) -> Any:
        timeout_seconds = timeout_seconds or settings.agent_timeout_seconds
        prompt = cap_prompt(agent_name, prompt)

        @retry(
            stop=stop_after_attempt(settings.agent_transport_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        )
        async def _run_with_retry():
            kwargs: dict[str, Any] = {"usage_limits": usage_limits(agent_name)}
            if (ms := run_model_settings(agent_name)) is not None:
                agent_ms = getattr(agent, "model_settings", None)  # pydantic_ai's Agent(model_settings=...)
                if isinstance(agent_ms, dict) and "max_tokens" in agent_ms:
                    ms = {**ms, "max_tokens": agent_ms["max_tokens"]}  # type: ignore[assignment]
                kwargs["model_settings"] = ms
            if deps is not None:
                kwargs["deps"] = deps
            if usage is not None:
                kwargs["usage"] = usage
            return await asyncio.wait_for(agent.run(prompt, **kwargs), timeout=timeout_seconds)

        async def _traced():
            tags = {
                "langfuse.session.id": run_id_var.get() or "",
                "langfuse.trace.name": f"pydantic-agents/{agent_name}",
            }
            start = time.time()
            result: Any = None
            with logfire.span("agent.run", agent_name=agent_name, prompt_length=len(prompt), **tags, _level="info"):
                try:
                    result = await telemetry.track(agent_name, _run_with_retry)
                    return result
                finally:
                    await _record_metric(agent_name, start, result)

        if permit_held.get():
            # Delegated call: the parent's permit already covers it. The flag is inherited by tasks created while
            # the permit is held (contextvars are copied into new tasks), so a fire-and-forget task spawned from
            # inside an agent run would bypass the limiter too: do not spawn unrelated background work from tools.
            return await _traced()

        async with semaphore:
            token = permit_held.set(True)
            try:
                return await _traced()
            finally:
                permit_held.reset(token)
