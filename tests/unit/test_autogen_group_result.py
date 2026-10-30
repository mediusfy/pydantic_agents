"""How an AutoGen group chat's result is folded into the pipeline state."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pydantic_agents.core.models.core import PipelineState

_AUTOGEN_MODULES = (
    "pydantic_agents.core.pipeline.autogen.bridge",
    "pydantic_agents.core.pipeline.autogen.groups",
    "pydantic_agents.core.adapters.autogen_bridge",
)


@pytest.fixture
def apply(monkeypatch):
    """``_apply_group_result`` imported with a stand-in for the optional ``autogen`` package.

    autogen is an extra that CI's test job does not install, and the bridge imports it at module level. The modules
    imported against the stand-in are purged afterwards so no other test sees them.
    """
    monkeypatch.setitem(sys.modules, "autogen", MagicMock())
    for name in _AUTOGEN_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    bridge = importlib.import_module("pydantic_agents.core.pipeline.autogen.bridge")
    yield bridge._apply_group_result
    for name in _AUTOGEN_MODULES:
        sys.modules.pop(name, None)


def _state() -> PipelineState:
    return PipelineState(run_id="r", stage="init", requirement="x")


def _chat(*messages: tuple[str, str]):
    return SimpleNamespace(
        summary="the summary", chat_history=[{"name": name, "content": content} for name, content in messages]
    )


GO_FILE = "### backend/main.go\n```go\npackage main\n```\n"
VUE_FILE = "### src/App.vue\n```vue\n<template><div/></template>\n```\n"
UNNAMED_BLOCK = "```go\npackage main\n```\n"


def test_the_design_chat_records_the_discussion_and_approves(apply):
    state = _state()
    state.cto_approved = False

    apply(state, "DesignGroupChat", _chat())

    assert state.advisory_results["design_discussion"] == "the summary"
    assert state.cto_approved is True
    assert state.artifacts == []


def test_the_debug_chat_becomes_feedback(apply):
    state = _state()

    apply(state, "DebugGroupChat", _chat())

    assert state.feedback == "the summary"


def test_an_unknown_chat_leaves_the_state_alone(apply):
    state = _state()
    before = state.model_dump()

    apply(state, "SomethingElse", _chat((("backend"), GO_FILE)))

    assert state.model_dump() == before


def test_a_result_without_a_summary_attribute_is_stringified(apply):
    state = _state()

    apply(state, "DebugGroupChat", "plain text result")

    assert state.feedback == "plain text result"


@pytest.mark.parametrize(
    ("speaker", "agent"),
    [
        ("Backend_Engineer", "backend"),
        ("frontend dev", "frontend"),
        ("QA_Lead", "test_eng"),
        ("Tester", "test_eng"),
        ("Docs_Writer", "docs"),
        ("DevOps_Eng", "devops"),
    ],
)
def test_sdlc_messages_become_artifacts_for_the_speaker_s_agent(apply, speaker, agent):
    state = _state()

    apply(state, "SdlcGroupChat", _chat((speaker, GO_FILE)))

    assert state.advisory_results["conversation"] == "the summary"
    [artifact] = state.artifacts
    assert artifact.agent_name == agent
    assert artifact.files == {"backend/main.go": "package main"}
    assert artifact.metadata == {"agent_name": agent, "user_story_id": "global"}


def test_the_first_matching_keyword_wins(apply):
    state = _state()

    apply(state, "SdlcGroupChat", _chat(("backend_test_docs", GO_FILE)))

    assert [a.agent_name for a in state.artifacts] == ["backend"]


def test_messages_that_carry_no_files_are_skipped(apply):
    state = _state()
    messages = [
        ("Backend", UNNAMED_BLOCK),  # a lone unnamed block parses to output.txt
        ("Backend", "just prose"),
        ("Backend", ""),  # no content
        ("", GO_FILE),  # no speaker
        ("Moderator", GO_FILE),  # a speaker no agent maps to
    ]

    apply(state, "SdlcGroupChat", _chat(*messages))

    assert state.artifacts == []


def test_several_speakers_each_contribute_their_files_in_order(apply):
    state = _state()

    apply(state, "SdlcGroupChat", _chat(("Backend", GO_FILE), ("Frontend", VUE_FILE)))

    assert [(a.agent_name, list(a.files)) for a in state.artifacts] == [
        ("backend", ["backend/main.go"]),
        ("frontend", ["src/App.vue"]),
    ]


def test_an_sdlc_result_without_chat_history_only_records_the_conversation(apply):
    state = _state()

    apply(state, "SdlcGroupChat", SimpleNamespace(summary="s"))

    assert state.advisory_results["conversation"] == "s"
    assert state.artifacts == []


def test_autogen_llm_config_resolves_lazy_model_and_litellm_proxy(monkeypatch):
    from pydantic_ai import Agent

    from pydantic_agents.core.adapters.autogen_bridge import _build_llm_config
    from pydantic_agents.core.models.factory import create_model
    from pydantic_agents.core.settings import settings

    monkeypatch.setenv("MODEL_PROVIDER", "litellm")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.test:4000")
    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "sk-test-virtual")
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setattr(settings, "model_provider", "litellm")

    agent = Agent(create_model("openai:deepseek-v4-pro"), system_prompt="hi", defer_model_check=True)
    (cfg,) = _build_llm_config(agent)

    assert cfg["model"] == "deepseek-v4-pro"
    assert cfg["base_url"] == "http://litellm.test:4000/v1"
    assert cfg["api_key"] == "sk-test-virtual"
    assert cfg["api_type"] == "openai"


async def test_autogen_tool_bridging_extracts_values_skips_runcontext_and_awaits_async():
    import asyncio

    from pydantic_ai import Agent, RunContext

    from pydantic_agents.core.adapters.autogen_bridge import _register_tool_functions

    agent = Agent("test", system_prompt="tool tester", defer_model_check=True)

    def sync_tool(x: int) -> int:
        return x + 1

    async def async_tool(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    def ctx_tool(ctx: RunContext[None], x: int) -> int:
        return x

    agent.tool_plain(sync_tool)
    agent.tool_plain(async_tool)
    agent.tool(ctx_tool)

    fns = _register_tool_functions(agent)

    assert set(fns) == {"sync_tool", "async_tool"}
    assert fns["sync_tool"](3) == 4
    assert fns["async_tool"](5) == 10


def test_manager_llm_config_uses_cto_litellm_connection(monkeypatch):
    from pydantic_agents.agents.registration import register_all_agents
    from pydantic_agents.core.settings import settings

    register_all_agents()
    monkeypatch.setenv("MODEL_PROVIDER", "litellm")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.test:4000")
    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "sk-test-virtual")
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setattr(settings, "model_provider", "litellm")
    monkeypatch.setitem(sys.modules, "autogen", MagicMock())
    for name in _AUTOGEN_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    groups = importlib.import_module("pydantic_agents.core.pipeline.autogen.groups")
    try:
        (cfg,) = groups._get_manager_llm_config()["config_list"]
        assert cfg["model"] == settings.cto_model
        assert cfg["base_url"] == "http://litellm.test:4000/v1"
        assert cfg["api_key"] == "sk-test-virtual"
    finally:
        for name in _AUTOGEN_MODULES:
            sys.modules.pop(name, None)


def test_manager_llm_config_with_native_provider_omits_litellm_proxy(monkeypatch):
    from pydantic_ai import Agent

    from pydantic_agents.agents.registration import register_all_agents
    from pydantic_agents.core.models.factory import create_model
    from pydantic_agents.core.registry import registry
    from pydantic_agents.core.settings import settings

    register_all_agents()
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-native-openai")
    monkeypatch.setattr(settings, "model_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-native-openai")
    monkeypatch.setitem(sys.modules, "autogen", MagicMock())
    for name in _AUTOGEN_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    native_cto = Agent(create_model("openai:gpt-5.4"), system_prompt="cto", defer_model_check=True)
    monkeypatch.setattr(registry, "get", lambda name: native_cto if name == "cto" else None)

    groups = importlib.import_module("pydantic_agents.core.pipeline.autogen.groups")
    try:
        (cfg,) = groups._get_manager_llm_config()["config_list"]
        assert cfg["model"] == "gpt-5.4"
        assert cfg["api_type"] == "openai"
        assert cfg["temperature"] == 0.1
        assert cfg.get("base_url") == "https://api.openai.com/v1/"
    finally:
        for name in _AUTOGEN_MODULES:
            sys.modules.pop(name, None)


def test_build_llm_config_raises_configuration_error_when_litellm_key_missing_on_lazy_model(monkeypatch):
    from pydantic_ai import Agent

    from pydantic_agents.core.adapters.autogen_bridge import _build_llm_config
    from pydantic_agents.core.exceptions import ConfigurationError
    from pydantic_agents.core.models.factory import create_model
    from pydantic_agents.core.settings import settings

    monkeypatch.setenv("MODEL_PROVIDER", "litellm")
    monkeypatch.delenv("LITELLM_VIRTUAL_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setattr(settings, "model_provider", "litellm")
    monkeypatch.setattr(settings, "litellm_virtual_key", None)
    monkeypatch.setattr(settings, "litellm_master_key", None)

    agent = Agent(create_model("openai:deepseek-v4-pro"), system_prompt="hi", defer_model_check=True)
    with pytest.raises(ConfigurationError, match="LITELLM_VIRTUAL_KEY"):
        _build_llm_config(agent)
