"""Centralized agent registration.

All registry.register() calls are collected here instead of living
as module-level side effects in individual agent files.
"""

from pydantic_agents.core.registry import registry

# Module-level flag is the fast path for "we've already registered". Set as the
# LAST step of register_all_agents() so a crash mid-loop correctly leaves the
# flag False and the next call retries.
#
# Tests sometimes mutate the global registry directly (e.g. via a fresh
# AgentRegistry fixture that doesn't share state). To stay correct under that
# pattern, we also confirm the registry actually contains a known agent — if
# the flag claims "done" but the registry is empty, we re-register.
_registered: bool = False
_KNOWN_AGENT = "ba"


def register_all_agents() -> None:
    """Register all agents (pipeline + travel/finance) on the unified registry.

    This is the single registration entry point for the entire ecosystem.
    Safe to call repeatedly — subsequent calls are no-ops.
    """
    global _registered
    if _registered and registry.is_registered(_KNOWN_AGENT):
        return
    # Either we never registered, or a test cleared the registry — reset.
    _registered = False

    # Fail at startup (and in `make smoke` / CI), not mid-pipeline, if pydantic_ai changed how agents store tools.
    from pydantic_agents.core.pydantic_ai_compat import check_pydantic_ai_compatibility

    check_pydantic_ai_compatibility()

    _register_sdlc_agents()

    _registered = True


def _register_sdlc_agents() -> None:
    """Register all technical SDLC pipeline agents."""
    from pydantic_agents.agents.ba import ba_agent
    from pydantic_agents.agents.backend import backend_agent, default_project_state
    from pydantic_agents.agents.cto import cto_agent, default_org_context
    from pydantic_agents.agents.database import DatabaseDeps, database_agent
    from pydantic_agents.agents.devops import DevopsDeps, devops_agent
    from pydantic_agents.agents.docs import docs_agent
    from pydantic_agents.agents.frontend import frontend_agent
    from pydantic_agents.agents.planner_agent import planner_agent
    from pydantic_agents.agents.qa_executor import qa_executor_agent
    from pydantic_agents.agents.quality_gate import quality_gate_agent
    from pydantic_agents.agents.review import ReviewCriteria, review_agent
    from pydantic_agents.agents.security import SecurityDeps, security_agent
    from pydantic_agents.agents.service_engineer import service_engineer_agent
    from pydantic_agents.agents.sre import sre_agent
    from pydantic_agents.agents.test_eng import test_eng_agent
    from pydantic_agents.agents.ux import ux_agent

    registry.register("ba", ba_agent, ["ba", "requirements"])
    # Default deps for calls that pass none; documented, settings-driven placeholders (see default_org_context).
    registry.register("cto", cto_agent, ["cto", "architecture", "advisory"], default_org_context)
    registry.register("backend", backend_agent, ["backend", "implementation.backend"], default_project_state)
    registry.register("frontend", frontend_agent, ["frontend", "implementation.frontend"])
    registry.register("database", database_agent, ["database", "db_schema", "migrations"], lambda: DatabaseDeps())
    registry.register("devops", devops_agent, ["devops", "ci_cd", "deployment"], lambda: DevopsDeps())
    registry.register("docs", docs_agent, ["docs", "documentation"])
    registry.register("sre", sre_agent, ["sre", "infrastructure"])
    registry.register("test_eng", test_eng_agent, ["test", "testing"])
    registry.register("review", review_agent, ["review", "code_review"], lambda: ReviewCriteria())
    registry.register("security", security_agent, ["security", "audit"], lambda: SecurityDeps())
    registry.register("ux", ux_agent, ["ux", "design"])
    registry.register("qa_executor", qa_executor_agent, ["qa_executor", "test_execution"])
    registry.register("quality_gate", quality_gate_agent, ["quality_gate"])
    registry.register(
        "service_engineer", service_engineer_agent, ["service_engineer", "existing_code", "implementation.modification"]
    )
    registry.register("planner", planner_agent, ["plan", "planning"])




