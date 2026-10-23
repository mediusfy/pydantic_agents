AGENT_TOOL_CATEGORIES = {
    "ba": ["web", "memory", "documentation", "a2a", "fs_read"],
    "backend": ["web", "search", "security", "validation", "deps", "memory", "a2a", "fs_read"],
    "frontend": ["web", "search", "deps", "memory", "a2a", "fs_read"],
    "frontend_component": ["web", "search", "deps", "memory", "documentation", "a2a", "fs_read"],
    "test_eng": ["search", "validation", "memory", "a2a", "fs_read"],
    "test_gen": ["search", "validation", "memory", "a2a", "fs_read"],
    "go_test": ["search", "validation", "memory", "a2a", "fs_read"],
    "sre": ["web", "security", "validation", "deps", "memory", "a2a", "fs_read"],
    "quality_gate": ["validation", "memory", "a2a"],
    "review": ["security", "memory", "documentation", "a2a", "fs_read"],
    "security": ["search", "security", "memory", "a2a", "fs_read"],
    "docs": ["web", "documentation", "a2a", "fs_read"],
    "cto": ["search", "memory", "documentation", "a2a", "fs_read"],
    "ux": ["web", "search", "memory", "documentation", "a2a", "fs_read"],
    "planner": ["web", "search", "memory", "a2a", "planning", "fs_read"],
    "service_engineer": ["search", "memory", "a2a"],
    "price_analyst": ["web", "search", "memory", "a2a", "pricing"],
    "holiday_planner": ["web", "search", "memory", "a2a", "planning"],
    "market_shark": ["web", "search", "memory", "a2a", "finance"],
    "crypto_oracle": ["web", "search", "memory", "finance", "news", "a2a"],
    "database": ["web", "search", "memory", "documentation", "a2a", "fs_read"],
    "devops": ["web", "search", "memory", "documentation", "a2a", "fs_read"],
    "accountant": ["web", "search", "memory", "a2a"],
    "asymmetric_reviewer": ["web", "search", "memory", "a2a"],
    "current_events": ["web", "search", "memory", "a2a"],
    "region_analyzer": ["web", "search", "memory", "a2a"],
    "weather": ["web", "search", "memory", "a2a"],
}


def get_tool_categories(agent_name: str) -> list[str]:
    return AGENT_TOOL_CATEGORIES.get(agent_name, [])
