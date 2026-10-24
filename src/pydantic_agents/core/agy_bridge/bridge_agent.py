"""BridgeAgent — wraps a registered pydantic_ai agent for the agy agentapi protocol.

Each BridgeAgent instance maps to one registry entry. It handles:
  - Conversation lifecycle (new → run → complete)
  - Context enrichment (reuses retrieve_context from the MCP layer)
  - Deps resolution via the registry's deps_factory
  - Telemetry tracking (reuses existing telemetry module)
  - Hook interception (pre-turn, post-turn, pre-tool-call)

Usage:
    from pydantic_agents.core.agy_bridge.bridge_agent import BridgeAgent
    agent = BridgeAgent.from_registry("ux")
    result = await agent.run("Evaluate this user flow", usage_limits=usage_limits("ux"))
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from pydantic_agents.core.exceptions import AgentExecutionError, AgentNotFoundError
from pydantic_agents.core.limits import usage_limits
from pydantic_agents.core.registry import registry
from pydantic_agents.core.settings import settings
from pydantic_agents.core.utils import serialize_output

logger = logging.getLogger(__name__)


# ── Conversation State ───────────────────────────────────────────────


@dataclass
class Conversation:
    """Holds state for one agent conversation session."""

    conversation_id: str
    agent_name: str
    model: str
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    status: str = "active"  # active | completed | failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "status": self.status,
            "message_count": len(self.history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ── Hook Protocol


@dataclass
class HookResult:
    """Result from a hook invocation. allow=False blocks the operation."""

    allow: bool = True
    message: str = ""
    modified_input: str | None = None


PreTurnHook: TypeAlias = Callable[[str], Awaitable[HookResult | None]]
PostTurnHook: TypeAlias = Callable[[str], Awaitable[None]]
PreToolHook: TypeAlias = Callable[[dict], Awaitable[HookResult | None]]
PostToolHook: TypeAlias = Callable[[dict], Awaitable[None]]


class Hooks:
    """Container for lifecycle hooks attached to a BridgeAgent."""

    def __init__(self) -> None:
        self.pre_turn: list[PreTurnHook] = []
        self.post_turn: list[PostTurnHook] = []
        self.pre_tool_call: list[PreToolHook] = []
        self.post_tool_call: list[PostToolHook] = []

    def on_pre_turn(self, hook: PreTurnHook) -> PreTurnHook:
        self.pre_turn.append(hook)
        return hook

    def on_post_turn(self, hook: PostTurnHook) -> PostTurnHook:
        self.post_turn.append(hook)
        return hook

    def on_pre_tool_call(self, hook: PreToolHook) -> PreToolHook:
        self.pre_tool_call.append(hook)
        return hook

    def on_post_tool_call(self, hook: PostToolHook) -> PostToolHook:
        self.post_tool_call.append(hook)
        return hook


# ── BridgeAgent ──────────────────────────────────────────────────────


class BridgeAgent:
    """Wraps a registered pydantic_ai agent for the agentapi protocol."""

    def __init__(
        self,
        name: str,
        model: str = "",
    ) -> None:
        self._name = name
        self._model = model or settings.resolve_model(settings.ba_model)
        self._agent = None
        self._deps_factory = None
        self._hooks = Hooks()
        self._tools_wrapped = False

        try:
            self._resolve()
        except ValueError as e:
            raise AgentNotFoundError(f"Agent '{name}' not registered") from e

    @classmethod
    def from_registry(cls, name: str) -> BridgeAgent:
        agent_tuple = registry._agents.get(name)
        if not agent_tuple:
            raise AgentNotFoundError(f"Agent '{name}' not registered")
        return cls(name)

    def _resolve(self):
        """Lazily resolve the pydantic_ai Agent instance and wrap tools."""
        if self._agent is None:
            self._agent = registry.get(self._name)
            agent_tuple = registry._agents.get(self._name)
            if agent_tuple:
                self._deps_factory = agent_tuple[2]
            self._wrap_tools()

    def _wrap_tools(self):
        """Intercepts internal pydantic_ai tool calls to trigger hooks."""
        if self._tools_wrapped or not self._agent or not hasattr(self._agent, "_function_tools"):
            return

        for tool in self._agent._function_tools:
            original_func = tool.func

            async def hooked_tool(run_context, *args, _tool=tool, _original_func=original_func, **kwargs):
                payload = {"tool": _tool.name, "args": args, "kwargs": kwargs}

                # ── Hook: pre-tool-call ──
                for hook in self._hooks.pre_tool_call:
                    result = await hook(payload)
                    if result is not None and not result.allow:
                        return f"System blocked tool execution: {result.message}"
                try:
                    tool_result = await _original_func(run_context, *args, **kwargs)
                except TypeError:
                    # Fallback if the tool doesn't expect run_context
                    tool_result = await _original_func(*args, **kwargs)

                # ── Hook: post-tool-call
                payload["result"] = tool_result
                for hook in self._hooks.post_tool_call:
                    await hook(payload)

                return tool_result

            tool.func = hooked_tool

        self._tools_wrapped = True

    @property
    def hooks(self) -> Hooks:
        return self._hooks

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def get_system_prompt(self) -> str:
        self._resolve()
        if self._agent:
            if hasattr(self._agent, "_system_prompts"):
                prompts = self._agent._system_prompts
                if prompts and isinstance(prompts, (list, tuple)) and any(prompts):
                    return "\n\n".join(str(p) for p in prompts if p)
            if hasattr(self._agent, "_system_prompt"):
                prompt = self._agent._system_prompt
                if callable(prompt):
                    return prompt() if callable(prompt) else str(prompt)
                return str(prompt)
        return ""

    def get_tool_names(self) -> list[str]:
        self._resolve()
        if self._agent and hasattr(self._agent, "_function_tools"):
            return [t.name for t in self._agent._function_tools]
        return []

    def get_definition(self) -> dict[str, Any]:
        self._resolve()
        return {
            "name": self._name,
            "description": self.get_system_prompt()[:200] if self.get_system_prompt() else "",
            "config": {
                "customAgent": {
                    "systemPromptSections": [
                        {
                            "title": "Agent System Instructions",
                            "content": self.get_system_prompt(),
                        }
                    ],
                    "toolNames": self.get_tool_names(),
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

    async def run(
        self,
        prompt: str,
        context: str = "",
        conversation: Conversation | None = None,
    ) -> str:
        self._resolve()

        # ── Hook: pre-turn
        modified_prompt = prompt
        for hook in self._hooks.pre_turn:
            result = await hook(modified_prompt)
            if result is not None:
                if not result.allow:
                    return f"Hook denied: {result.message}"
                if result.modified_input:
                    modified_prompt = result.modified_input
        # ──────────────────────────────────────────────────────────────────

        # ── Context enrichment
        enriched = modified_prompt
        if context:
            # Bug fix: use modified_prompt instead of context for user_task
            enriched = (
                f"Context from IDE:\n<ide_context>\n{context}\n"
                f"</ide_context>\n\nTask:\n<user_task>\n{modified_prompt}\n</user_task>"
            )
        # ──────────────────────────────────────────────────────────────────

        if self._agent is None:
            raise AgentExecutionError(f"BridgeAgent '{self._name}' has no resolved agent", is_retryable=False)

        start = time.time()
        success = False

        try:
            resolved_deps: dict[str, Any] = self._deps_factory() if callable(self._deps_factory) else {}
            result = await self._agent.run(enriched, deps=resolved_deps, usage_limits=usage_limits(self._name))
            success = True

            output = serialize_output(result)

            # ── Update conversation state
            if conversation:
                conversation.history.append({"role": "user", "content": modified_prompt})
                conversation.history.append({"role": "assistant", "content": output})
                conversation.updated_at = time.time()
                conversation.status = "active"
            # ──────────────────────────────────────────────────────────────

            # ── Hook: post-turn
            for hook in self._hooks.post_turn:
                await hook(output)
            # ──────────────────────────────────────────────────────────────

            return output

        except Exception as e:
            if conversation:
                conversation.status = "failed"
                conversation.updated_at = time.time()
            raise AgentExecutionError(f"BridgeAgent '{self._name}' failed: {e}", is_retryable=False) from e
        finally:
            duration = time.time() - start
            with contextlib.suppress(Exception):
                from pydantic_agents.core.telemetry import telemetry

                await telemetry.log_call(
                    self._name,
                    duration,
                    success,
                    metadata={"bridge": "agy", "prompt_length": len(prompt)},
                )
