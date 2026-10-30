"""Tool-call hooks and system-prompt lookup on a BridgeAgent, against a fake pydantic_ai agent."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pydantic_agents.core.agy_bridge.bridge_agent import BridgeAgent, HookResult


class FakeAgent:
    def __init__(self, tools=(), **attrs):
        self._function_tools = list(tools)
        for key, value in attrs.items():
            setattr(self, key, value)


def make(fake) -> BridgeAgent:
    with patch("pydantic_agents.core.agy_bridge.bridge_agent.registry") as registry:
        registry._agents = {"fake": (fake, [], None, None)}
        registry.get.return_value = fake
        return BridgeAgent("fake")


# --- tool hooks -------------------------------------------------------------------------------------------------


async def test_a_tool_without_hooks_runs_and_returns_its_result():
    async def original(ctx, value):
        return f"{ctx}:{value}"

    tool = SimpleNamespace(name="echo", func=original)
    make(FakeAgent([tool]))

    assert await tool.func("ctx", "v") == "ctx:v"


async def test_a_denying_pre_tool_hook_blocks_the_tool_and_later_hooks():
    ran, asked = [], []

    async def original(ctx):
        ran.append(ctx)
        return "ran"

    async def deny(payload):
        asked.append(payload["tool"])
        return HookResult(allow=False, message="not today")

    async def never(payload):
        asked.append("second")

    tool = SimpleNamespace(name="danger", func=original)
    agent = make(FakeAgent([tool]))
    agent.hooks.on_pre_tool_call(deny)
    agent.hooks.on_pre_tool_call(never)

    assert await tool.func("ctx") == "System blocked tool execution: not today"
    assert ran == []
    assert asked == ["danger"]


async def test_pre_tool_hooks_that_allow_or_return_nothing_let_the_tool_run():
    async def original(ctx):
        return "ran"

    async def allow(payload):
        return HookResult(allow=True)

    async def silent(payload):
        return None

    tool = SimpleNamespace(name="ok", func=original)
    agent = make(FakeAgent([tool]))
    agent.hooks.on_pre_tool_call(allow)
    agent.hooks.on_pre_tool_call(silent)

    assert await tool.func("ctx") == "ran"


async def test_post_tool_hooks_see_the_call_and_its_result():
    seen = []

    async def original(ctx, a, b=0):
        return a + b

    async def record(payload):
        seen.append(dict(payload))

    tool = SimpleNamespace(name="add", func=original)
    agent = make(FakeAgent([tool]))
    agent.hooks.on_post_tool_call(record)

    assert await tool.func("ctx", 1, b=2) == 3
    assert seen == [{"tool": "add", "args": (1,), "kwargs": {"b": 2}, "result": 3}]


async def test_a_tool_that_takes_no_run_context_is_called_without_it():
    async def original(a):
        return f"got {a}"

    tool = SimpleNamespace(name="plain", func=original)
    make(FakeAgent([tool]))

    assert await tool.func("ctx", "arg") == "got arg"  # the first call raises TypeError; the retry drops the context


async def test_each_tool_keeps_its_own_original():
    async def first(ctx):
        return "first"

    async def second(ctx):
        return "second"

    one, two = SimpleNamespace(name="one", func=first), SimpleNamespace(name="two", func=second)
    make(FakeAgent([one, two]))

    assert [await one.func("c"), await two.func("c")] == ["first", "second"]


def test_tools_are_wrapped_only_once():
    async def original(ctx):
        return "x"

    tool = SimpleNamespace(name="t", func=original)
    agent = make(FakeAgent([tool]))
    wrapped = tool.func

    agent._wrap_tools()

    assert tool.func is wrapped
    assert wrapped is not original


def test_an_agent_without_function_tools_is_left_alone():
    agent = make(SimpleNamespace())

    assert agent._tools_wrapped is False


# --- system prompt ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"_system_prompts": ["a", "", "b"]}, "a\n\nb"),
        ({"_system_prompts": ("only",)}, "only"),
        ({"_system_prompts": [], "_system_prompt": "single"}, "single"),
        ({"_system_prompts": ["", ""], "_system_prompt": lambda: "called"}, "called"),
        ({"_system_prompts": "not a list", "_system_prompt": "fallback"}, "fallback"),
        ({"_system_prompt": 42}, "42"),
        ({"_system_prompts": ["wins"], "_system_prompt": "ignored"}, "wins"),
        ({}, ""),
        ({"_system_prompts": []}, ""),
    ],
    ids=[
        "list",
        "tuple",
        "empty-then-single",
        "empty-then-callable",
        "not-a-list",
        "non-string",
        "list-wins",
        "none",
        "empty",
    ],
)
def test_system_prompt_lookup(attrs, expected):
    agent = make(FakeAgent(**attrs))

    assert agent.get_system_prompt() == expected


def test_an_unresolved_agent_has_an_empty_system_prompt(monkeypatch):
    agent = make(FakeAgent(_system_prompt="never used"))
    monkeypatch.setattr(agent, "_resolve", lambda: None)
    agent._agent = None

    assert agent.get_system_prompt() == ""


async def test_real_pydantic_ai_agent_tools_are_discovered_and_hooked():
    from pydantic_ai import Agent

    real_agent = Agent("test", system_prompt="real agent", defer_model_check=True)

    @real_agent.tool_plain
    def ping(x: str) -> str:
        return f"pong:{x}"

    bridge = make(real_agent)
    assert bridge.get_tool_names() == ["ping"]

    seen = []

    async def on_post(payload):
        seen.append(payload["result"])

    bridge.hooks.on_post_tool_call(on_post)
    tool_obj = real_agent._function_toolset.tools["ping"]
    assert await tool_obj.function("ctx", "hi") == "pong:hi"
    assert seen == ["pong:hi"]
