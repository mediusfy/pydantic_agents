"""core/sre_remediation.py: pure allowlist/escalation policy, no I/O."""

from pydantic_agents.core.sre_remediation import (
    ACTION_RESUME_INTERRUPTED_RUN,
    ALLOWLISTED_ACTIONS,
    plan_for_interrupted_run,
    plan_for_startup_check_error,
    plan_remediations,
)


def test_interrupted_run_is_allowlisted():
    plan = plan_for_interrupted_run("pipe_abc123", {"project": "demo", "status": "interrupted"})
    assert plan.auto_remediable is True
    assert plan.action == ACTION_RESUME_INTERRUPTED_RUN
    assert plan.evidence["run_id"] == "pipe_abc123"


def test_startup_check_error_always_escalates():
    plan = plan_for_startup_check_error("DATABASE_URL not set")
    assert plan.auto_remediable is False
    assert plan.action is None
    assert plan.summary == "DATABASE_URL not set"


def test_plan_remediations_combines_both_sources():
    plans = plan_remediations(
        startup_check_errors=["bad api key"],
        interrupted_runs={"pipe_1": {"project": "demo", "status": "interrupted"}},
    )
    assert len(plans) == 2
    assert sum(p.auto_remediable for p in plans) == 1
    assert sum(not p.auto_remediable for p in plans) == 1


def test_plan_remediations_empty_input_yields_no_plans():
    assert plan_remediations(startup_check_errors=[], interrupted_runs={}) == []


def test_allowlist_has_exactly_one_action_in_v1():
    # Documents the deliberately small v1 scope (ADR-0001); extending this is a code change, not a runtime one.
    assert len(ALLOWLISTED_ACTIONS) == 1
    assert ACTION_RESUME_INTERRUPTED_RUN in ALLOWLISTED_ACTIONS
