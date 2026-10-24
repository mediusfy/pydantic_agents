"""Post-deploy smoke test: call real agents through the MCP endpoint and check the output parses.

Usage: ``post_deploy_smoke.py`` with ``MCP_URL`` (default ``$LITELLM_BASE_URL/mcp``) and ``MCP_KEY`` (default
``$LITELLM_VIRTUAL_KEY``) set. Exits non-zero if the endpoint, a tool call or an output check fails.

It calls ``run_review``, ``run_security`` and ``run_crypto_oracle`` once each, so a deploy that cannot reach the model
gateway, the database behind the memory tools, or that returns output clients cannot parse is caught right after the
rollout. Each call costs tokens; run it after a deploy, not in CI.
"""

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from pydantic_agents.core.models.analysis import Forecast
from pydantic_agents.core.models.core import SECURITY_PASS_SCORE, ReviewResult, SecurityReport

REVIEW_SNIPPET = """\
func health(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("ok"))
}
"""

# An agent call is a full LLM run with tool use, so it can take minutes; do not tidy this down.
TOOL_TIMEOUT_SECONDS = 600

if TYPE_CHECKING:
    from mcp import ClientSession

CallTool = Callable[[str, str], Awaitable[str]]


# The checks below prove the output parses as its typed model (and, for security, that the verdict follows the score).
# They are a smoke test, not a quality gate: a poor but well-formed review passes.


def check_review(text: str) -> str:
    result = ReviewResult.model_validate_json(text)
    return f"approved={result.approved} security_score={result.security_score}"


def check_security(text: str) -> str:
    report = SecurityReport.model_validate_json(text)
    if report.passed != (report.score >= SECURITY_PASS_SCORE):
        raise ValueError(f"passed={report.passed} does not follow score={report.score}")
    return f"passed={report.passed} score={report.score}"


def check_forecast(text: str) -> str:
    forecast = Forecast.model_validate_json(text)
    return f"{forecast.asset}: {len(forecast.scenarios)} scenarios, {len(forecast.sources)} sources"


@dataclass(frozen=True)
class Check:
    tool: str
    request: str
    validate: Callable[[str], str]


CHECKS = (
    Check("run_review", f"Review this Go snippet for production readiness:\n\n{REVIEW_SNIPPET}", check_review),
    Check("run_security", f"Audit this Go snippet:\n\n{REVIEW_SNIPPET}", check_security),
    Check("run_crypto_oracle", "Forecast BTC over the next 30 days.", check_forecast),
)


def find_tool(names: list[str], suffix: str) -> str | None:
    """The gateway prefixes tool names (``pydantic_agents-run_review``); match on the ``run_<agent>`` suffix.

    Raises ValueError when more than one tool matches, so a versioned or duplicated name cannot silently pick one.
    """
    matches = [n for n in names if n == suffix or n.endswith(f"-{suffix}") or n.endswith(f"_{suffix}")]
    if len(matches) > 1:
        raise ValueError(f"{suffix} is ambiguous: {sorted(matches)}")
    return matches[0] if matches else None


async def run_checks(names: list[str], call: CallTool, checks: tuple[Check, ...] = CHECKS) -> list[tuple[str, str]]:
    """Returns (check name, error) for every failed check; an empty list means all passed."""
    failures: list[tuple[str, str]] = []
    for check in checks:
        try:
            tool = find_tool(names, check.tool)
            if tool is None:
                failures.append((check.tool, "tool not exposed by the endpoint"))
                continue
            summary = check.validate(await call(tool, check.request))
        except Exception as e:
            failures.append((check.tool, f"{type(e).__name__}: {str(e)[:300]}"))
        else:
            print(f"ok   {check.tool}: {summary}")
    return failures


async def _tool_names(session: "ClientSession") -> list[str]:
    return [t.name for t in (await session.list_tools()).tools]


def _caller(session: "ClientSession") -> CallTool:
    async def call(tool: str, request: str) -> str:
        result = await session.call_tool(
            tool, {"request": request}, read_timeout_seconds=timedelta(seconds=TOOL_TIMEOUT_SECONDS)
        )
        text = "".join(part.text for part in result.content if part.type == "text")
        if result.isError:
            raise RuntimeError(text[:300])
        if not text:
            raise RuntimeError("the tool returned no text content")
        return text

    return call


async def main() -> int:
    base = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
    url = os.environ.get("MCP_URL") or (f"{base}/mcp" if base else "")
    key = os.environ.get("MCP_KEY") or os.environ.get("LITELLM_VIRTUAL_KEY")
    if not url:
        print("set MCP_URL (or LITELLM_BASE_URL)", file=sys.stderr)
        return 2

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    # LiteLLM and the pod's own MCP server both accept a bearer token; set MCP_URL for a gateway that wants another scheme.
    headers = {"Authorization": f"Bearer {key}"} if key else None
    async with AsyncExitStack() as stack:
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(url, headers=headers, sse_read_timeout=TOOL_TIMEOUT_SECONDS)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        failures = await run_checks(await _tool_names(session), _caller(session))

    for name, error in failures:
        print(f"FAIL {name}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
