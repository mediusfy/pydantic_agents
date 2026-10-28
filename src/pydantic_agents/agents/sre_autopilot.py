"""The ``sre_autopilot`` agent (ADR-0001 / AP-77): continuous health loop with a narrow, explicit
auto-remediation allowlist.

Extends ``core/healthcheck.py``'s checks from a one-shot startup call to something callable repeatedly, and
reads ``pipeline_jobs.list_interrupted_runs()`` for runs cut off by a restart. The allowlist
(``core/sre_remediation.py``) is a literal set in code: an interrupted pipeline run is auto-resumed via the
existing ``resume_pipeline_run``; every startup-check failure (bad key, unwritable directory, unreachable
database) always escalates as an open ``platform_findings`` row, since none of those has a generic reversible
fix. Nothing outside the allowlist is ever acted on automatically.
"""

import json
from typing import Any

from pydantic_ai import Agent, RunContext

from pydantic_agents.core import healthcheck, pipeline_jobs, platform_findings
from pydantic_agents.core.models.core import SreAutopilotReport
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings
from pydantic_agents.core.sre_remediation import plan_remediations

sre_autopilot_prompt = """You are the SRE Autopilot Agent: a continuous health-check loop for this platform's
own infrastructure (not user code).

You have two tools:
- `health_snapshot`: current startup-check errors (API keys, directories, database) and any pipeline runs that
  were interrupted by a restart. Read-only.
- `scan_and_remediate`: deterministically classifies each detected issue against a fixed allowlist. An
  interrupted pipeline run is auto-resumed (the only allowlisted action in v1). Everything else is written as
  an open `platform_findings` row for human/CTO review — it is never acted on automatically. Call this to do
  the actual work; there is no other way to remediate or escalate.

Workflow:
1. Call `health_snapshot` to see what is wrong.
2. Call `scan_and_remediate` to act on the allowlist and record everything else.
3. Summarize what was auto-fixed vs. escalated in `narrative`, and mirror each action in `records`.

You never act outside the allowlist, and you never change code, prompts, or pipeline configuration yourself —
only the two things this module already does: resume an interrupted run, or write a finding for a human to see.
"""

sre_autopilot_agent = Agent(
    create_model(settings.resolve_model(settings.sre_autopilot_model)),
    system_prompt=sre_autopilot_prompt,
    output_type=SreAutopilotReport,
    defer_model_check=True,
    retries=settings.agent_retries,
)


@sre_autopilot_agent.tool
async def health_snapshot(ctx: RunContext[Any]) -> str:
    """Current startup-check errors and interrupted pipeline runs, as JSON."""
    errors = healthcheck.run_startup_checks()
    interrupted = pipeline_jobs.list_interrupted_runs()
    return json.dumps({"startup_check_errors": errors, "interrupted_runs": interrupted}, default=str)


@sre_autopilot_agent.tool
async def scan_and_remediate(ctx: RunContext[Any]) -> str:
    """Detect issues, auto-resume any allowlisted interrupted pipeline run, and persist every other issue as an
    open platform_findings row. Returns one line per action taken (or 'No issues.' when nothing was found)."""
    errors = healthcheck.run_startup_checks()
    interrupted = pipeline_jobs.list_interrupted_runs()
    plans = plan_remediations(startup_check_errors=errors, interrupted_runs=interrupted)

    lines: list[str] = []
    for plan in plans:
        finding_id = await platform_findings.save_finding(
            plan.kind, plan.severity, plan.summary, agent=plan.agent, evidence=plan.evidence
        )
        if plan.auto_remediable and plan.run_id is not None:
            await pipeline_jobs.resume_pipeline_run(plan.run_id)
            lines.append(f"[auto-remediated] {plan.summary} (finding_id={finding_id})")
        else:
            lines.append(f"[escalated] {plan.summary} (finding_id={finding_id})")
    return "\n".join(lines) if lines else "No issues."
