"""promptfoo Python provider: runs a registered agent directly.

Bypasses the MCP server's vector-context enrichment so evals need only the
LiteLLM proxy (LITELLM_BASE_URL / LITELLM_VIRTUAL_KEY), not Postgres.

Select the agent per provider with ``config.agent`` in promptfooconfig.yaml.
Structured outputs are returned as JSON so ``is-json`` / ``javascript``
assertions can inspect fields.
"""

from typing import Any

from pydantic import BaseModel

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.registry import registry


async def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    agent_name = options.get("config", {}).get("agent")
    if not agent_name:
        return {"error": "provider config.agent is required"}

    register_all_agents()
    try:
        agent = registry.get(agent_name)
    except ValueError as e:
        return {"error": str(e)}

    deps_factory = registry._agents[agent_name][2]
    deps = deps_factory() if callable(deps_factory) else {}

    try:
        result = await agent.run(prompt, deps=deps)
    except Exception as e:
        return {"error": f"agent '{agent_name}' failed: {e}"}

    output = result.output
    text = output.model_dump_json() if isinstance(output, BaseModel) else str(output)

    usage = result.usage()
    return {
        "output": text,
        "tokenUsage": {
            "prompt": usage.input_tokens or 0,
            "completion": usage.output_tokens or 0,
            "total": (usage.input_tokens or 0) + (usage.output_tokens or 0),
        },
    }
