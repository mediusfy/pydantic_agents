"""agent_sync — discovers registered pydantic_ai agents and generates
.agents/agent.json definitions for agy native discovery.

This module creates the agent.json files that agy's brain reads to display
agents as native subagents with system prompts, tool lists, and model configs.
It reads from the shared AgentRegistry and writes to the agy brain directory.

Usage:
    from pydantic_agents.core.agy_bridge.agent_sync import sync_agent_definitions
    await sync_agent_definitions()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic_agents.core.agy_bridge.bridge_agent import BridgeAgent
from pydantic_agents.core.exceptions import AgentNotFoundError
from pydantic_agents.core.registry import registry
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)


def _find_agy_brain_dirs() -> list[Path]:
    """Find agy brain directories containing .agents/agents/ structure."""
    candidates = []
    base_dirs = [
        Path.home() / ".gemini" / "antigravity-cli" / "brain",
        Path.home() / ".gemini" / "antigravity-ide" / "brain",
    ]

    for base in base_dirs:
        if not base.exists():
            continue
        for brain_dir in base.iterdir():
            if brain_dir.is_dir() and (agents_dir := brain_dir / ".agents" / "agents").exists():
                candidates.append(agents_dir)

    return candidates


def _describe_model(model_spec: str) -> str:
    """Map a pydantic_ai model spec to a human-readable agy model name."""
    model_map: dict[str, str] = {
        "deepseek-reasoner": "deepseek:deepseek-reasoner",
        "claude-sonnet-4-20250514": "claude:claude-sonnet-4-20250514",
        "gpt-4o": "openai:gpt-4o",
        "gemini-2.0-flash": "gemini:gemini-2.0-flash",
        "gemini-2.0-pro": "gemini:gemini-2.0-pro",
        "gemini-3.1-pro": "gemini:gemini-3.1-pro",
        "gemini-3.1-flash": "gemini:gemini-3.1-flash",
    }
    clean = model_spec.split(":", 1)[-1] if ":" in model_spec else model_spec
    return model_map.get(clean, model_spec)


def _get_capabilities_docstring(name: str) -> str:
    """Return a human-readable description from known docstrings."""
    from pydantic_agents.mcp_server import AGENT_DOCSTRINGS

    return AGENT_DOCSTRINGS.get(name, f"Orgitration {name} agent")


def build_agent_definition(agent_name: str, force_model: str = "") -> dict[str, Any]:
    """Build a single agent.json definition from the registry."""
    try:
        bridge = BridgeAgent.from_registry(agent_name)
    except AgentNotFoundError:
        logger.warning("Agent '%s' not in registry, skipping", agent_name)
        return {}

    model = force_model
    if not model:
        model_attr = f"{agent_name}_model"
        if hasattr(settings, model_attr):
            raw = getattr(settings, model_attr)
            model = settings.resolve_model(raw)
        else:
            model = settings.resolve_model(settings.ba_model)

    description = _get_capabilities_docstring(agent_name)

    return {
        "name": agent_name,
        "description": description,
        "config": {
            "customAgent": {
                "systemPromptSections": [
                    {
                        "title": "System Instructions",
                        "content": bridge.get_system_prompt(),
                    },
                    {
                        "title": "Capabilities",
                        "content": description,
                    },
                ],
                "toolNames": bridge.get_tool_names(),
                "modelPreference": model,
                "systemPromptConfig": {
                    "includeSections": [
                        "user_information",
                        "mcp_servers",
                        "skills",
                        "subagent_reminder",
                        "messaging",
                        "artifacts",
                        "user_rules",
                    ]
                },
            }
        },
    }


async def sync_agent_definitions(target_dir: str | Path | None = None) -> int:
    """Generate and write agent.json files for all registered agents."""
    from pydantic_agents.agents.registration import register_all_agents

    register_all_agents()

    targets = [Path(target_dir)] if target_dir else _find_agy_brain_dirs()

    if not targets:
        logger.warning(
            "No agy brain .agents/agents/ directories found. Run agy to initialize one, or pass an explicit target_dir."
        )
        return 0

    written = 0
    agent_names = registry.list_agents()
    model_overrides: dict[str, str] = {}

    for name in agent_names:
        model_attr = f"{name}_model"
        if hasattr(settings, model_attr):
            raw = getattr(settings, model_attr)
            model_overrides[name] = settings.resolve_model(raw)
        else:
            model_overrides[name] = settings.resolve_model(settings.ba_model)

    for target in targets:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        logger.info("Syncing agent definitions to %s", target)

        for agent_name in agent_names:
            definition = build_agent_definition(agent_name, force_model=model_overrides.get(agent_name, ""))
            if not definition:
                continue

            file_path = target / f"{agent_name}.agent.json"
            file_path.write_text(json.dumps(definition, indent=2, ensure_ascii=False))
            written += 1
            logger.debug("Wrote %s", file_path)

        index = {
            "agents": [
                {
                    "name": name,
                    "description": definition.get("description", ""),
                    "configFile": f"{name}.agent.json",
                }
                for name in agent_names
                if (definition := build_agent_definition(name, model_overrides.get(name, "")))
            ]
        }
        index_path = target / ".." / "agents-index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
            logger.debug("Wrote agent index to %s", index_path)
        except Exception as e:
            logger.warning("Failed to write agent index: %s", e)

    return written


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(sync_agent_definitions())
