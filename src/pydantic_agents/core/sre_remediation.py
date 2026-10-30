"""Deterministic remediation policy for the ``sre_autopilot`` agent (AP-83 / ADR-0001).

Pure, no I/O: given a health issue, decide whether it is on the v1 auto-remediation allowlist or must be escalated
as a ``platform_findings`` row for human/CTO review. The allowlist is a literal set in code, not inferred by the
model at runtime, per ADR-0001's guardrail that `sre_autopilot` never acts outside a pre-approved, explicit,
reversible set of actions.

v1 allowlist is intentionally one item: resuming an interrupted ``pipeline_runs`` row via the existing
``resume_pipeline_run`` (reversible — it restarts from the run's own checkpoints). ADR-0001 also named "retry one
stuck broker task" as a second allowlisted action; that action no longer applies because the event broker it
targeted was removed from the codebase (PR #60, 2026-10-28, same day as the ADR) before this ticket was picked up.
"""

from dataclasses import dataclass
from typing import Any

ACTION_RESUME_INTERRUPTED_RUN = "resume_interrupted_pipeline_run"

# A run that is interrupted again after this many resumes is probably what is killing the server (an out-of-memory
# run would otherwise be resumed on every restart, forever): it is escalated to a human instead.
MAX_AUTO_RESUMES = 3

# Literal, reviewable allowlist. Extending it is a deliberate code change (and an ADR amendment per ADR-0001),
# never a runtime decision by the model.
ALLOWLISTED_ACTIONS = frozenset({ACTION_RESUME_INTERRUPTED_RUN})


@dataclass
class RemediationPlan:
    action: str | None  # one of ALLOWLISTED_ACTIONS, or None when this issue must escalate
    kind: str  # platform_findings kind: "health"
    severity: str
    agent: str | None
    summary: str
    evidence: dict[str, Any]
    run_id: str | None = None  # set only for ACTION_RESUME_INTERRUPTED_RUN, so the caller never digs it out of evidence

    @property
    def auto_remediable(self) -> bool:
        return self.action is not None and self.action in ALLOWLISTED_ACTIONS


def plan_for_interrupted_run(run_id: str, run: dict[str, Any]) -> RemediationPlan:
    """An interrupted pipeline run is always on the allowlist: resuming it is reversible and matches exactly
    what a human would otherwise do by calling ``resume_pipeline_run`` themselves, up to ``MAX_AUTO_RESUMES``
    times: a run that keeps being interrupted escalates instead."""
    attempts = int(run.get("resume_attempts") or 0)
    if attempts >= MAX_AUTO_RESUMES:
        return RemediationPlan(
            action=None,
            kind="health",
            severity="high",
            agent=None,
            summary=(
                f"pipeline run {run_id} ({run.get('project', 'unknown project')}) was interrupted again after "
                f"{attempts} automatic resumes; not resuming it again"
            ),
            evidence={"run_id": run_id, "project": run.get("project"), "resume_attempts": attempts},
        )
    return RemediationPlan(
        action=ACTION_RESUME_INTERRUPTED_RUN,
        kind="health",
        severity="medium",
        agent=None,
        summary=f"pipeline run {run_id} ({run.get('project', 'unknown project')}) was interrupted; auto-resuming",
        evidence={"run_id": run_id, "project": run.get("project"), "status": run.get("status")},
        run_id=run_id,
    )


def plan_for_startup_check_error(message: str) -> RemediationPlan:
    """Startup-check failures (bad API key, unwritable directory, unreachable database) are never on the
    allowlist in v1 — none of them has a reversible, generic fix; all require a human to change config or
    infrastructure. Always escalates."""
    return RemediationPlan(
        action=None,
        kind="health",
        severity="high",
        agent=None,
        summary=message,
        evidence={"source": "startup_check"},
    )


def plan_remediations(
    *, startup_check_errors: list[str], interrupted_runs: dict[str, dict[str, Any]]
) -> list[RemediationPlan]:
    """One plan per detected issue: interrupted runs are auto-remediable, startup-check errors always escalate."""
    plans = [plan_for_interrupted_run(run_id, run) for run_id, run in interrupted_runs.items()]
    plans.extend(plan_for_startup_check_error(message) for message in startup_check_errors)
    return plans
