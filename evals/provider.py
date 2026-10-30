"""promptfoo Python provider: runs a registered agent directly.

Bypasses the MCP server's vector-context enrichment so evals need only the
LiteLLM proxy (LITELLM_BASE_URL / LITELLM_VIRTUAL_KEY), not Postgres.

Select the agent per provider with ``config.agent`` in promptfooconfig.yaml.
Structured outputs are returned as JSON so ``is-json`` / ``javascript``
assertions can inspect fields.

Agents run through ``registry.run``, the guarded production path (deps factory, usage caps, concurrency permit,
transport retry), with a timeout below promptfoo's 300 s worker limit so a stalled run (for example a delegation
stuck behind a gateway rate limit) reports an error instead of being killed silently.
"""

import os
from typing import Any

from pydantic import BaseModel

# Evals do not export traces; silence logfire's "not configured" warning on every tool call.
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.exceptions import AgentNotFoundError
from pydantic_agents.core.registry import registry

_EVAL_TIMEOUT_SECONDS = float(os.environ.get("EVAL_AGENT_TIMEOUT_SECONDS", "240"))


async def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001 - promptfoo signature
    agent_name = options.get("config", {}).get("agent")
    if not agent_name:
        return {"error": "provider config.agent is required"}

    register_all_agents()
    try:
        result = await registry.run(agent_name, prompt, timeout_seconds=_EVAL_TIMEOUT_SECONDS)
    except AgentNotFoundError as e:
        return {"error": str(e)}
    except TimeoutError:
        return {"error": f"agent '{agent_name}' timed out after {_EVAL_TIMEOUT_SECONDS:.0f}s"}
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
