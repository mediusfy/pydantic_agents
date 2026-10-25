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
            self._hook_tool(tool)

        self._tools_wrapped = True

    async def _blocked_by_pre_tool_hook(self, payload: dict[str, Any]) -> str | None:
        """The message to return instead of running the tool if a pre-tool-call hook denies it, else None."""
        for hook in self._hooks.pre_tool_call:
            result = await hook(payload)
            if result is not None and not result.allow:
                return f"System blocked tool execution: {result.message}"
        return None

    def _hook_tool(self, tool) -> None:
        """Replace ``tool.func`` with a version that runs the pre- and post-tool-call hooks around the original."""
        original_func = tool.func

        async def hooked_tool(run_context, *args, **kwargs):
            payload = {"tool": tool.name, "args": args, "kwargs": kwargs}

            blocked = await self._blocked_by_pre_tool_hook(payload)
            if blocked is not None:
                return blocked
            try:
                tool_result = await original_func(run_context, *args, **kwargs)
            except TypeError:
                # Fallback if the tool doesn't expect run_context
                tool_result = await original_func(*args, **kwargs)

            payload["result"] = tool_result
            for hook in self._hooks.post_tool_call:
                await hook(payload)

            return tool_result

        tool.func = hooked_tool

    @property
    def hooks(self) -> Hooks:
        return self._hooks

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _joined_system_prompts(agent: Any) -> str:
        """The agent's static system prompts joined, or "" when it has none."""
        prompts = getattr(agent, "_system_prompts", None)
        if isinstance(prompts, (list, tuple)) and any(prompts):
            return "\n\n".join(str(p) for p in prompts if p)
        return ""

    @staticmethod
    def _single_system_prompt(agent: Any) -> str | None:
        """The agent's ``_system_prompt`` (called if it is a callable), or None when it has no such attribute."""
        if not hasattr(agent, "_system_prompt"):
            return None
        prompt = agent._system_prompt
        return prompt() if callable(prompt) else str(prompt)

    def get_system_prompt(self) -> str:
        self._resolve()
        if not self._agent:
            return ""
        joined = self._joined_system_prompts(self._agent)
        if joined:
            return joined
        single = self._single_system_prompt(self._agent)
        return "" if single is None else single

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

    async def _run_pre_turn_hooks(self, prompt: str) -> tuple[str, str | None]:
        """Apply the pre-turn hooks. Returns the (possibly modified) prompt and a denial message if a hook denied."""
        modified_prompt = prompt
        for hook in self._hooks.pre_turn:
            result = await hook(modified_prompt)
            if result is None:
                continue
            if not result.allow:
                return modified_prompt, f"Hook denied: {result.message}"
            if result.modified_input:
                modified_prompt = result.modified_input
        return modified_prompt, None

    @staticmethod
    def _record_turn(conversation: Conversation | None, prompt: str, output: str) -> None:
        if conversation:
            conversation.history.append({"role": "user", "content": prompt})
            conversation.history.append({"role": "assistant", "content": output})
            conversation.updated_at = time.time()
            conversation.status = "active"

    @staticmethod
    def _mark_failed(conversation: Conversation | None) -> None:
        if conversation:
            conversation.status = "failed"
            conversation.updated_at = time.time()

    async def run(
        self,
        prompt: str,
        context: str = "",
        conversation: Conversation | None = None,
    ) -> str:
        self._resolve()

        modified_prompt, denied = await self._run_pre_turn_hooks(prompt)
        if denied is not None:
            return denied

        # ── Context enrichment
        enriched = modified_prompt
        if context:
            # Bug fix: use modified_prompt instead of context for user_task
            enriched = (
                f"Context from IDE:\n<ide_context>\n{context}\n"
                f"</ide_context>\n\nTask:\n<user_task>\n{modified_prompt}\n</user_task>"
            )
        # ──────────────────────────────────────────────────────────────────

        start = time.time()
        success = False

        try:
            if self._agent is None:
                # Inside the try so the failure is recorded on the conversation and in telemetry like any other.
                raise AgentExecutionError("agent was not resolved", is_retryable=False)
            resolved_deps: dict[str, Any] = self._deps_factory() if callable(self._deps_factory) else {}
            result = await self._agent.run(enriched, deps=resolved_deps, usage_limits=usage_limits(self._name))
            success = True

            output = serialize_output(result)

            self._record_turn(conversation, modified_prompt, output)

            # ── Hook: post-turn
            for hook in self._hooks.post_turn:
                await hook(output)
            # ──────────────────────────────────────────────────────────────

            return output

        except Exception as e:
            self._mark_failed(conversation)
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
