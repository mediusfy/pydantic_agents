AGENT_TOOL_CATEGORIES = {
    "ba": ["web", "memory", "documentation", "a2a"],
    "backend": ["web", "search", "security", "validation", "deps", "memory", "a2a"],
    "frontend": ["web", "search", "deps", "memory", "a2a"],
    "frontend_component": ["web", "search", "deps", "memory", "documentation", "a2a"],
    "test_eng": ["search", "validation", "memory", "a2a"],
    "test_gen": ["search", "validation", "memory", "a2a"],
    "go_test": ["search", "validation", "memory", "a2a"],
    "sre": ["web", "security", "validation", "deps", "memory", "a2a"],
    "quality_gate": ["validation", "memory", "a2a"],
    "review": ["security", "memory", "documentation", "a2a"],
    "security": ["search", "security", "memory", "a2a"],
    "docs": ["web", "documentation", "a2a"],
    "cto": ["search", "memory", "documentation", "a2a"],
    "ux": ["web", "search", "memory", "documentation", "a2a"],
    "planner": ["web", "search", "memory", "a2a", "planning"],
    "service_engineer": ["search", "memory", "a2a"],
    "price_analyst": ["web", "search", "memory", "a2a", "pricing"],
    "holiday_planner": ["web", "search", "memory", "a2a", "planning"],
    "market_shark": ["web", "search", "memory", "a2a", "finance"],
    "crypto_oracle": ["web", "search", "memory", "finance", "news", "a2a"],
    "database": ["web", "search", "memory", "documentation", "a2a"],
    "devops": ["web", "search", "memory", "documentation", "a2a"],
    "accountant": ["web", "search", "memory", "a2a"],
    "asymmetric_reviewer": ["web", "search", "memory", "a2a"],
    "current_events": ["web", "search", "memory", "a2a"],
    "region_analyzer": ["web", "search", "memory", "a2a"],
    "weather": ["web", "search", "memory", "a2a"],
}


def get_tool_categories(agent_name: str) -> list[str]:
    return AGENT_TOOL_CATEGORIES.get(agent_name, [])
