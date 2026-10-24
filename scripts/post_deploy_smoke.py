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
from dataclasses import dataclass

from pydantic_agents.core.models.analysis import Forecast
from pydantic_agents.core.models.core import SECURITY_PASS_SCORE, ReviewResult, SecurityReport

REVIEW_SNIPPET = """\
func health(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("ok"))
}
"""

CallTool = Callable[[str, str], Awaitable[str]]


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
    """The gateway prefixes tool names (``pydantic_agents-run_review``); match on the ``run_<agent>`` suffix."""
    return next((n for n in names if n == suffix or n.endswith(f"-{suffix}") or n.endswith(f"_{suffix}")), None)


async def run_checks(names: list[str], call: CallTool, checks: tuple[Check, ...] = CHECKS) -> list[tuple[str, str]]:
    """Returns (check name, error) for every failed check; an empty list means all passed."""
    failures: list[tuple[str, str]] = []
    for check in checks:
        tool = find_tool(names, check.tool)
        if tool is None:
            failures.append((check.tool, "tool not exposed by the endpoint"))
            continue
        try:
            summary = check.validate(await call(tool, check.request))
        except Exception as e:
            failures.append((check.tool, f"{type(e).__name__}: {str(e)[:300]}"))
        else:
            print(f"ok   {check.tool}: {summary}")
    return failures


async def main() -> int:
    from fastmcp import Client

    base = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
    url = os.environ.get("MCP_URL") or (f"{base}/mcp" if base else "")
    key = os.environ.get("MCP_KEY") or os.environ.get("LITELLM_VIRTUAL_KEY")
    if not url:
        print("set MCP_URL (or LITELLM_BASE_URL)", file=sys.stderr)
        return 2

    async with Client(url, auth=key) as client:
        names = [t.name for t in await client.list_tools()]

        async def call(tool: str, request: str) -> str:
            result = await client.call_tool(tool, {"request": request}, timeout=600)
            return "".join(getattr(part, "text", "") for part in result.content)

        failures = await run_checks(names, call)

    for name, error in failures:
        print(f"FAIL {name}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
