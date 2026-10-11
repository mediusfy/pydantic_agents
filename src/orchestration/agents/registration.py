"""Centralized agent registration.

All registry.register() calls are collected here instead of living
as module-level side effects in individual agent files.
"""

from orchestration.registry import registry

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

    _registered = True


def _register_sdlc_agents() -> None:
    """Register all technical SDLC pipeline agents."""
    from orchestration.agents.ba import ba_agent
    from orchestration.agents.backend import backend_agent
    from orchestration.agents.cto import cto_agent
    from orchestration.agents.database import database_agent
    from orchestration.agents.devops import devops_agent
    from orchestration.agents.docs import docs_agent
    from orchestration.agents.frontend import component_gen, frontend_agent
    from orchestration.agents.planner_agent import planner_agent
    from orchestration.agents.quality_gate import quality_gate_agent
    from orchestration.agents.review import review_agent
    from orchestration.agents.security import security_agent
    from orchestration.agents.service_engineer import service_engineer_agent
    from orchestration.agents.sre import sre_agent
    from orchestration.agents.test_eng import go_test_gen, test_eng_agent, test_gen
    from orchestration.agents.ux import ux_agent

    registry.register("ba", ba_agent, ["ba", "requirements"])
    registry.register("cto", cto_agent, ["cto", "architecture", "advisory"])
    registry.register("backend", backend_agent, ["backend", "implementation.backend"])
    registry.register("frontend", frontend_agent, ["frontend", "implementation.frontend"])
    registry.register("frontend_component", component_gen, ["frontend_component"])
    registry.register("database", database_agent, ["database", "db_schema", "migrations"])
    registry.register("devops", devops_agent, ["devops", "ci_cd", "deployment"])
    registry.register("docs", docs_agent, ["docs", "documentation"])
    registry.register("sre", sre_agent, ["sre", "infrastructure"])
    registry.register("test_eng", test_eng_agent, ["test", "testing"])
    registry.register("test_gen", test_gen, ["test_gen", "test_generation"])
    registry.register("go_test", go_test_gen, ["go_test", "go_testing"])
    registry.register("review", review_agent, ["review", "code_review"])
    registry.register("security", security_agent, ["security", "audit"])
    registry.register("ux", ux_agent, ["ux", "design"])
    registry.register("quality_gate", quality_gate_agent, ["quality_gate"])
    registry.register(
        "service_engineer", service_engineer_agent, ["service_engineer", "existing_code", "implementation.modification"]
    )
    registry.register("planner", planner_agent, ["plan", "planning"])


def _register_utility_agents() -> None:
    """Register utility and travel/finance agents (non-technical)."""
    from orchestration.agents.non_technical.accountant import AccountantDeps, accountant_agent
    from orchestration.agents.non_technical.asymmetric_reviewer import (
        AsymmetricReviewerDeps,
        asymmetric_reviewer_agent,
    )
    from orchestration.agents.non_technical.crypto_oracle import CryptoOracleDeps, crypto_oracle_agent
    from orchestration.agents.non_technical.current_events import CurrentEventsDeps, current_events_agent
    from orchestration.agents.non_technical.holiday_planner import HolidayPlannerDeps, holiday_planner_agent
    from orchestration.agents.non_technical.market_shark import MarketSharkDeps, market_shark_agent
    from orchestration.agents.non_technical.price_analyst import PriceAnalystDeps, price_analyst_agent
    from orchestration.agents.non_technical.region_analyzer import RegionAnalyzerDeps, region_analyzer_agent
    from orchestration.agents.non_technical.weather import WeatherDeps, weather_agent

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
