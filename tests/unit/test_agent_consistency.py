"""Cross-checks the three places an agent is declared: its model field, its registry entry and its tool categories.

Nothing else ties them together, so adding an agent to one and forgetting another fails silently (an agent with no
tools, a model field nobody reads, a tool category nothing provides).

Conventions this relies on: agents read their model as ``settings.<name>_model`` (the repo-wide ``settings`` singleton,
no aliasing or ``getattr``), and every ``*_model`` field on ``Settings`` belongs to an agent. A new non-agent field
ending in ``_model`` (an embedding model, say) must be exempted in ``NON_AGENT_MODEL_FIELDS`` below.
"""

import re
from pathlib import Path

import pytest

import pydantic_agents.agents as agents_pkg
from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.registry import registry
from pydantic_agents.core.settings import Settings
from pydantic_agents.core.tool_assignments import AGENT_TOOL_CATEGORIES
from pydantic_agents.core.tools import registry as tools_registry

AGENTS_DIR = Path(agents_pkg.__file__).parent
# The lookahead skips the settings.resolve_model(...) method call.
MODEL_FIELD = re.compile(r"settings\.(\w+_model)\b(?!\()")
NON_AGENT_MODEL_FIELDS: set[str] = set()


@pytest.fixture(scope="module", autouse=True)
def _agents():
    register_all_agents()


def _model_fields_in_settings() -> set[str]:
    return {name for name in Settings.model_fields if name.endswith("_model")} - NON_AGENT_MODEL_FIELDS


def _model_fields_in_source(text: str) -> set[str]:
    return set(MODEL_FIELD.findall(text))


def _model_fields_used_by_agents() -> set[str]:
    return {m for path in AGENTS_DIR.rglob("*.py") for m in _model_fields_in_source(path.read_text())}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("create_model(settings.resolve_model(settings.review_model))", {"review_model"}),
        ("settings.resolve_model(name)", set()),
        ("a = settings.ba_model; b = settings.cto_model", {"ba_model", "cto_model"}),
        ("self.settings.review_model", {"review_model"}),
        ("cfg.review_model", set()),
        ("review_model = 'x'", set()),
    ],
    ids=[
        "field inside a call",
        "method call is not a field",
        "two fields",
        "attribute chain",
        "other object",
        "bare name",
    ],
)
def test_the_field_scan_finds_settings_fields_and_not_method_calls(source, expected):
    assert _model_fields_in_source(source) == expected


def test_every_registered_agent_has_tool_categories_and_none_are_orphaned():
    registered = set(registry.list_agents())
    assigned = set(AGENT_TOOL_CATEGORIES)
    assert registered - assigned == set(), "registered agents with no entry in tool_assignments"
    assert assigned - registered == set(), "tool_assignments entries for agents that are not registered"


def test_every_assigned_category_is_provided_by_a_registered_tool():
    provided = {c for tool in tools_registry.tool_registry.values() for c in tool.categories}
    for agent, categories in AGENT_TOOL_CATEGORIES.items():
        missing = set(categories) - provided
        assert not missing, f"{agent} is assigned categories no tool provides: {sorted(missing)}"


def test_every_model_field_an_agent_reads_exists_in_settings():
    unknown = _model_fields_used_by_agents() - _model_fields_in_settings()
    assert unknown == set(), f"agents read settings fields that do not exist: {sorted(unknown)}"


def test_every_model_field_in_settings_is_read_by_an_agent():
    unused = _model_fields_in_settings() - _model_fields_used_by_agents()
    assert unused == set(), f"settings model fields no agent reads: {sorted(unused)}"


def test_every_registered_agent_has_the_tools_its_categories_grant():
    for name in registry.list_agents():
        granted = {t for t, _ in tools_registry.get_tools_for_agent(AGENT_TOOL_CATEGORIES[name])}
        attached = set(registry.get(name)._function_toolset.tools)
        # Agents may add tools of their own, but never lack a granted one.
        assert granted <= attached, f"{name} is missing granted tools: {sorted(granted - attached)}"
