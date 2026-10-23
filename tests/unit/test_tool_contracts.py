"""Contracts between prompts, tool registration, tool assignments and settings."""

import re

import pytest

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core import tool_assignments
from pydantic_agents.core.registry import registry
from pydantic_agents.core.settings import settings
from pydantic_agents.core.tools.registry import get_tools_for_agent, register_tool, tool_registry


@pytest.fixture(scope="module", autouse=True)
def _agents():
    register_all_agents()


def _toolset(name):
    return set(registry.get(name)._function_toolset.tools)


def _all_known_tool_names():
    names = set(tool_registry)
    for name in registry.list_agents():
        names |= _toolset(name)
    return names


def _prompt(name):
    return "\n".join(str(p) for p in registry.get(name)._system_prompts)


def test_every_tool_named_in_a_prompt_is_attached():
    known = _all_known_tool_names()
    missing, checked = [], 0
    for name in registry.list_agents():
        for mention in set(re.findall(r"`([a-z][a-z0-9_]*)`", _prompt(name))):
            if mention in known:
                checked += 1
                if mention not in _toolset(name):
                    missing.append(f"{name} -> {mention}")
    assert checked >= 5, "the contract found almost no tool mentions; the pattern is probably broken"
    assert not missing, f"prompts name tools the agent does not have: {missing}"


def test_assignments_cover_exactly_the_registered_agents():
    assert set(tool_assignments.AGENT_TOOL_CATEGORIES) == set(registry.list_agents())


def test_every_assigned_category_exists():
    known = {c for d in tool_registry.values() for c in d.categories}
    unknown = {c for cats in tool_assignments.AGENT_TOOL_CATEGORIES.values() for c in cats} - known
    # fs_write is intentionally granted to nobody; everything assigned must be provided by some tool.
    assert not unknown, f"categories assigned but provided by no tool: {unknown}"


def test_every_agent_has_its_own_model_setting():
    # Agents that share a model on purpose (test_gen/go_test use test_model, frontend_component uses frontend_model).
    shared = {
        "test_eng": "test_model",
        "test_gen": "test_model",
        "go_test": "test_model",
        "frontend_component": "frontend_model",
        "sre": "sre_model",
        "ux": "ux_model",
    }
    for name in registry.list_agents():
        field = shared.get(name, f"{name}_model")
        assert hasattr(settings, field), f"{name}: settings has no {field}"


def test_agent_name_is_not_an_implicit_category():
    register_tool("probe_tool_xyz", lambda: None, "probe", ["devops"])
    try:
        assert "probe_tool_xyz" not in {n for n, _ in get_tools_for_agent("devops", ["web"])}
    finally:
        tool_registry.pop("probe_tool_xyz")


def test_duplicate_tool_name_is_an_error():
    with pytest.raises(ValueError, match="already registered"):
        register_tool("run_go_build", lambda: None, "dup", ["x"])


def test_reregistering_the_same_function_is_allowed():
    tool = tool_registry["run_go_build"]
    register_tool("run_go_build", tool.fn, tool.description, tool.categories)


@pytest.mark.parametrize(
    "agent, absent",
    [
        ("backend", {"validate_frontend_code", "execute_test_suite"}),
        ("sre", {"run_go_build", "run_go_vet", "execute_test_suite"}),
        ("test_eng", {"validate_frontend_code"}),
    ],
)
def test_tool_sets_are_trimmed(agent, absent):
    assert not absent & _toolset(agent)


@pytest.mark.parametrize("agent, present", [("frontend", {"validate_frontend_code"}), ("devops", {"validate_yaml"})])
def test_needed_tools_are_still_attached(agent, present):
    assert present <= _toolset(agent)


@pytest.mark.parametrize(
    "module, attr",
    [("test_eng", "_test_writer"), ("test_eng", "_go_test_writer"), ("frontend", "_component_writer")],
)
def test_inner_generators_are_private_and_tool_less(module, attr):
    import importlib

    agent = getattr(importlib.import_module(f"pydantic_agents.agents.{module}"), attr)
    assert not agent._function_toolset.tools
    assert all(agent is not a for _, (a, *_rest) in registry.items())


def test_ba_has_no_nested_llm_tools():
    assert not _toolset("ba") & {"define_user_stories", "map_event_triggers"}
