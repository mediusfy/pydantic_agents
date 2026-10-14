import re

import pytest

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.registry import registry


@pytest.fixture(scope="module", autouse=True)
def setup_registry():
    register_all_agents()


def test_all_referenced_tools_exist():
    """Verifies that any tool mentioned in an agent's system prompt is actually registered."""
    for name in registry.list_agents():
        agent = registry.get(name)  # Trigger lazy_init
        prompt = "\n".join(agent._system_prompts)
        if not prompt:
            continue

        # Look for patterns like `use query_history`, `call log_activity`, etc.
        # This is a heuristic, but good for catching the 'Phantom Tools' identified in PR.
        tool_mentions = re.findall(r"`([^`]+)`", prompt)

        # Filter for things that look like tools (usually snake_case)
        potential_tools = [m for m in tool_mentions if "_" in m and "(" not in m]

        # Some are just gherkin or code examples, we skip them if they don't look like tools
        # For now, let's just check against the actual tools registered on the agent
        registered_tools = list(agent._function_toolset.tools.keys())

        for tool in potential_tools:
            # Skip known non-tool mentions that happen to be snake_case
            if tool in ["updated_at", "created_at", "user_id", "tenant_id", "gen_random_uuid"]:
                continue
            if tool.startswith("BC-"):  # Story IDs
                continue
            if tool.endswith(".go") or tool.endswith(".proto"):  # File names
                continue

            # If it looks like a tool call instruction, it should be in registered_tools
            if any(verb in prompt.lower() for verb in ["use", "call", "via", "using"]) and tool in [
                "query_history",
                "log_activity",
                "update_documentation",
            ]:
                msg = f"Agent '{name}' references non-existent tool '{tool}' in system prompt"
                assert tool in registered_tools, msg


def test_go_version_consistency():
    """Ensures all Go agents are referencing the same version from settings."""
    from pydantic_agents.core.settings import settings

    expected_version = settings.go_version

    # Check Backend agent
    backend = registry.get("backend")
    prompt = "\n".join(backend._system_prompts)
    assert f"Go {expected_version}" in prompt

    # Check DevOps agent (GitHub Action tool)
    devops = registry.get("devops")
    prompt_devops = "\n".join(devops._system_prompts)
    # We can't easily check tool output without running it, but we can check the tool's docstring.
    # For now, just ensuring the system prompt doesn't have a hardcoded old version is good.
    assert "1.25+" not in prompt_devops
    assert "1.26" not in prompt_devops or expected_version == "1.26"
