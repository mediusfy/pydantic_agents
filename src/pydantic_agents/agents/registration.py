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

    _register_sdlc_agents()
    _register_utility_agents()
    _register_platform_intelligence_agents()

    _registered = True


def _register_sdlc_agents() -> None:
    """Register all technical SDLC pipeline agents."""
    from pydantic_agents.agents.api_contract import api_contract_agent
    from pydantic_agents.agents.ba import ba_agent
    from pydantic_agents.agents.backend import ProjectState, backend_agent
    from pydantic_agents.agents.critic import critic_agent
    from pydantic_agents.agents.cto import OrgContext, cto_agent
    from pydantic_agents.agents.database import DatabaseDeps, database_agent
    from pydantic_agents.agents.devops import DevopsDeps, devops_agent
    from pydantic_agents.agents.docs import docs_agent
    from pydantic_agents.agents.finops import finops_agent
    from pydantic_agents.agents.frontend import component_gen, frontend_agent
    from pydantic_agents.agents.observability_engineer import observability_engineer_agent
    from pydantic_agents.agents.planner_agent import planner_agent
    from pydantic_agents.agents.qa_executor import qa_executor_agent
    from pydantic_agents.agents.quality_gate import quality_gate_agent
    from pydantic_agents.agents.release_manager import release_manager_agent
    from pydantic_agents.agents.review import ReviewCriteria, review_agent
    from pydantic_agents.agents.security import SecurityDeps, security_agent
    from pydantic_agents.agents.service_engineer import service_engineer_agent
    from pydantic_agents.agents.sre import sre_agent
    from pydantic_agents.agents.test_eng import go_test_gen, test_eng_agent, test_gen
    from pydantic_agents.agents.ux import ux_agent

    registry.register("ba", ba_agent, ["ba", "requirements"])
    registry.register(
        "cto",
        cto_agent,
        ["cto", "architecture", "advisory"],
        lambda: OrgContext(
            tech_stack=["Go", "Vue 3", "PostgreSQL", "NATS", "gRPC"],
            compliance_tier="Standard",
            monthly_cloud_budget=5000.0,
        ),
    )
    registry.register(
        "backend",
        backend_agent,
        ["backend", "implementation.backend"],
        lambda: ProjectState(base_path=".", environment="development", team_owner="platform"),
    )
    registry.register("frontend", frontend_agent, ["frontend", "implementation.frontend"])
    registry.register("frontend_component", component_gen, ["frontend_component"])
    registry.register("database", database_agent, ["database", "db_schema", "migrations"], lambda: DatabaseDeps())
    registry.register("devops", devops_agent, ["devops", "ci_cd", "deployment"], lambda: DevopsDeps())
    registry.register("docs", docs_agent, ["docs", "documentation"])
    registry.register("sre", sre_agent, ["sre", "infrastructure"])
    registry.register("test_eng", test_eng_agent, ["test", "testing"])
    registry.register("test_gen", test_gen, ["test_gen", "test_generation"])
    registry.register("go_test", go_test_gen, ["go_test", "go_testing"])
    registry.register("review", review_agent, ["review", "code_review"], lambda: ReviewCriteria())
    registry.register("security", security_agent, ["security", "audit"], lambda: SecurityDeps())
    registry.register("ux", ux_agent, ["ux", "design"])
    registry.register("qa_executor", qa_executor_agent, ["qa_executor", "test_execution"])
    registry.register("quality_gate", quality_gate_agent, ["quality_gate"])
    registry.register(
        "service_engineer", service_engineer_agent, ["service_engineer", "existing_code", "implementation.modification"]
    )
    registry.register("planner", planner_agent, ["plan", "planning"])
    registry.register("critic", critic_agent, ["critic", "verification"])
    registry.register("api_contract", api_contract_agent, ["api_contract", "contract_drift"])
    registry.register("observability_engineer", observability_engineer_agent, ["observability", "alerting"])
    registry.register("finops", finops_agent, ["finops", "cost"])
    registry.register("release_manager", release_manager_agent, ["release", "changelog"])


def _register_platform_intelligence_agents() -> None:
    """Register the Platform Intelligence Layer agents (ADR-0001 / AP-77)."""
    from pydantic_agents.agents.innovation_scout import innovation_scout_agent
    from pydantic_agents.agents.observer import observer_agent
    from pydantic_agents.agents.product_strategist import product_strategist_agent
    from pydantic_agents.agents.sre_autopilot import sre_autopilot_agent

    registry.register("observer", observer_agent, ["observer", "platform_observability"])
    registry.register("sre_autopilot", sre_autopilot_agent, ["sre_autopilot", "platform_health"])
    registry.register("innovation_scout", innovation_scout_agent, ["innovation_scout", "platform_intelligence"])
    registry.register("product_strategist", product_strategist_agent, ["product_strategist", "product_strategy"])


def _register_utility_agents() -> None:
    """Register utility and travel/finance agents (non-technical)."""
    from pydantic_agents.agents.non_technical.accountant import AccountantDeps, accountant_agent
    from pydantic_agents.agents.non_technical.asymmetric_reviewer import (
        AsymmetricReviewerDeps,
        asymmetric_reviewer_agent,
    )
    from pydantic_agents.agents.non_technical.crypto_oracle import CryptoOracleDeps, crypto_oracle_agent
    from pydantic_agents.agents.non_technical.current_events import CurrentEventsDeps, current_events_agent
    from pydantic_agents.agents.non_technical.holiday_planner import HolidayPlannerDeps, holiday_planner_agent
    from pydantic_agents.agents.non_technical.market_shark import MarketSharkDeps, market_shark_agent
    from pydantic_agents.agents.non_technical.price_analyst import PriceAnalystDeps, price_analyst_agent
    from pydantic_agents.agents.non_technical.region_analyzer import RegionAnalyzerDeps, region_analyzer_agent
    from pydantic_agents.agents.non_technical.weather import WeatherDeps, weather_agent

    registry.register("accountant", accountant_agent, ["finance.accounting"], lambda: AccountantDeps())
    registry.register(
        "asymmetric_reviewer",
        asymmetric_reviewer_agent,
        ["legal", "finance.legal"],
        lambda: AsymmetricReviewerDeps(),
    )
    registry.register("current_events", current_events_agent, ["news", "history"], lambda: CurrentEventsDeps())
    registry.register("region_analyzer", region_analyzer_agent, ["analysis.region"], lambda: RegionAnalyzerDeps())
    registry.register("weather", weather_agent, ["weather"], lambda: WeatherDeps())
    registry.register(
        "crypto_oracle", crypto_oracle_agent, ["finance.crypto", "news.crypto", "research"], lambda: CryptoOracleDeps()
    )
    registry.register(
        "holiday_planner",
        holiday_planner_agent,
        ["planning.travel", "travel"],
        lambda: HolidayPlannerDeps(),
    )
    registry.register(
        "market_shark",
        market_shark_agent,
        ["finance.stocks", "crypto", "stocks", "investing"],
        lambda: MarketSharkDeps(),
    )
    registry.register(
        "price_analyst",
        price_analyst_agent,
        ["pricing", "analysis.pricing", "purchasing"],
        lambda: PriceAnalystDeps(),
    )
