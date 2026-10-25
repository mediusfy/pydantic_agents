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
