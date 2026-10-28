"""The ``observer`` agent (ADR-0001 / AP-77): continuous quality/performance review across runs.

Reads ``platform_metrics`` (per-call agent duration/success/verdict, AP-78) and the already-open
``platform_findings`` (AP-79) so it does not re-report an issue that is already tracked, then scans for
regressions, flaky agents, and latency drift. Read-only with respect to the rest of the platform: it never
touches code, config, or the filesystem — the only write path is ``platform_findings`` rows.
"""

import json
from typing import Any

from pydantic_ai import Agent, RunContext

from pydantic_agents.core import platform_findings, platform_metrics
from pydantic_agents.core.models.core import ObserverReport
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.observer_analysis import detect_findings
from pydantic_agents.core.settings import settings

# Upper bound on any model-chosen row limit, so one tool call cannot pull the whole table into the context.
_MAX_ROWS = 500

observer_prompt = """You are the Observer Agent: a continuous quality/performance reviewer for this platform's own
SDLC agents, not for user code.

You have three tools:
- `recent_metrics`: recent platform_metrics rows (agent, duration, success, verdict).
- `open_findings`: already-open platform_findings rows, so you do not re-report a tracked issue.
- `scan_for_findings`: deterministically detects regressions, flaky agents, and latency drift across recent
  metrics and persists each one as a new open platform_findings row. Call this to do the actual detection and
  writing; do not try to write findings yourself by other means — there is no other write path.

Workflow:
1. Call `open_findings` first so you know what is already tracked.
2. Call `scan_for_findings` to detect and persist anything new.
3. Optionally call `recent_metrics` for extra context if you want to comment on a specific agent.
4. Summarize what you found (or that nothing new was found) in `narrative`, and mirror any newly written findings
   in `findings` (kind/severity/agent/summary) so a caller reading only your output still sees them without a
   second platform_findings query.

You never propose or make code/config changes yourself; you only observe and record findings. Anything beyond
"flag it" is a human/CTO decision.
"""

observer_agent = Agent(
    create_model(settings.resolve_model(settings.observer_model)),
    system_prompt=observer_prompt,
    output_type=ObserverReport,
    defer_model_check=True,
    retries=settings.agent_retries,
)


@observer_agent.tool
async def recent_metrics(ctx: RunContext[Any], agent: str | None = None, limit: int = 100) -> str:
    """Recent platform_metrics rows (agent, duration, success, verdict), newest first, as JSON."""
    rows = await platform_metrics.recent_calls(agent=agent, limit=min(limit, _MAX_ROWS))
    return json.dumps(rows, default=str)


@observer_agent.tool
async def open_findings(ctx: RunContext[Any], limit: int = 50) -> str:
    """Already-open platform_findings rows, newest first, as JSON."""
    rows = await platform_findings.list_findings(status="open", limit=min(limit, _MAX_ROWS))
    return json.dumps(rows, default=str)


@observer_agent.tool
async def scan_for_findings(ctx: RunContext[Any], limit: int = 200) -> str:
    """Scan the last `limit` platform_metrics rows for regressions, flaky agents and latency drift (grouped by
    agent), persist each one not already open (same kind and agent) as an open platform_findings row, and return
    one line per finding (or 'No findings.' when nothing crossed the threshold).
    """
    rows = await platform_metrics.recent_calls(limit=min(limit, _MAX_ROWS))
    # A scheduled scan sees the same persistent issue every time; one open row per (kind, agent) is enough.
    already_open = {
        (row.get("kind"), row.get("agent"))
        for row in await platform_findings.list_findings(status="open", limit=_MAX_ROWS)
    }
    lines = []
    for finding in detect_findings(rows):
        if (finding.kind, finding.agent) in already_open:
            lines.append(f"[{finding.kind}/{finding.severity}] {finding.summary} (already open, not re-recorded)")
            continue
        finding_id = await platform_findings.save_finding(
            finding.kind, finding.severity, finding.summary, agent=finding.agent, evidence=finding.evidence
        )
        lines.append(f"[{finding.kind}/{finding.severity}] {finding.summary} (finding_id={finding_id})")
    return "\n".join(lines) if lines else "No findings."
