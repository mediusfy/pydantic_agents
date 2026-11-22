AGENT_TOOL_CATEGORIES = {
    "ba": ["web", "memory", "documentation", "a2a", "fs_read"],
    "backend": ["web", "search", "security", "validation", "go_build", "deps", "memory", "a2a", "fs_read"],
    "frontend": ["web", "search", "deps", "frontend_validation", "memory", "a2a", "fs_read"],
    "test_eng": ["search", "validation", "go_build", "test_run", "memory", "a2a", "fs_read"],
    "qa_executor": ["validation", "go_build", "test_run", "memory", "a2a", "fs_read"],
    "sre": ["web", "security", "validation", "deps", "memory", "a2a", "fs_read"],
    # Verifiers get read and execute tools only: a gate that could delegate to the agent it judges is no gate.
    "quality_gate": ["validation", "go_build", "memory"],
    "review": ["security", "memory", "fs_read"],
    "security": ["search", "security", "memory", "fs_read"],
    "docs": ["web", "documentation", "a2a", "fs_read"],
    "cto": ["search", "memory", "a2a", "fs_read"],
    "ux": ["web", "search", "memory", "documentation", "a2a", "fs_read"],
    "planner": ["web", "search", "memory", "a2a", "planning", "fs_read"],
    "service_engineer": ["search", "memory", "a2a"],
    "database": ["web", "search", "memory", "documentation", "a2a", "fs_read"],
    "devops": ["web", "search", "config_validation", "memory", "documentation", "a2a", "fs_read"],
}


def get_tool_categories(agent_name: str) -> list[str]:
    return AGENT_TOOL_CATEGORIES.get(agent_name, [])
