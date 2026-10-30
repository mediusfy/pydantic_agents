"""Deps reach the model through instructions, and the schemas carry what the prompts ask for."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from pydantic_agents.agents.backend import ProjectState, backend_agent
from pydantic_agents.agents.cto import OrgContext, cto_agent
from pydantic_agents.agents.review import ReviewCriteria, review_agent
from pydantic_agents.core.models.core import (
    AgentAdvice,
    ArchitecturalConstraint,
    EventTrigger,
    PipelineState,
    UserStory,
)
from pydantic_agents.core.pipeline.config import PipelineConfig


def _captured_instructions(agent, deps):
    """Run the agent on a stub model and return the instructions it sent.

    Reads the instructions off the first request message, which is pydantic-ai behaviour; the dependency is capped
    at <2.0 in pyproject.toml, so revisit this helper when that cap moves.
    """
    # call_tools=[]: only the first request is needed, and the default would execute every tool the agent has.
    with agent.override(model=TestModel(call_tools=[])):
        result = agent.run_sync("go", deps=deps)
    return result.all_messages()[0].instructions or ""


@pytest.mark.parametrize(
    "agent, deps, expected",
    [
        (
            cto_agent,
            OrgContext(tech_stack=["Go", "Vue"], compliance_tier="PCI", monthly_cloud_budget=1500.0),
            ["Go, Vue", "PCI", "1,500.00"],
        ),
        (
            backend_agent,
            ProjectState(base_path="/app/shop", environment="staging", team_owner="payments-team"),
            ["/app/shop", "staging", "payments-team"],
        ),
        (
            review_agent,
            ReviewCriteria(bounded_contexts=["orders", "billing"]),
            ["orders, billing", "Strict mode is ENABLED"],
        ),
    ],
)
def test_deps_are_rendered_into_instructions(agent, deps, expected):
    text = _captured_instructions(agent, deps)
    for fragment in expected:
        assert fragment in text


def test_review_prompt_has_no_hardcoded_contexts():
    text = _captured_instructions(review_agent, None)
    assert "payments" not in text
    assert "notifications" not in text


def test_schemas_carry_what_the_ba_and_cto_prompts_ask_for():
    trigger = EventTrigger(
        id="E1",
        name="orders.OrderPlaced",
        description="d",
        owner="orders",
        consumers=["billing"],
        payload_schema="{order_id: str}",
    )
    story = UserStory(id="ORDERS-01", title="t", gherkin="g", bounded_context="orders", event_triggers=[trigger])
    assert story.event_triggers[0].consumers == ["billing"]
    advice = AgentAdvice(
        architectural_constraints=[ArchitecturalConstraint(description="No global state", affected_agents=["backend"])]
    )
    assert advice.architectural_constraints[0].affected_agents == ["backend"]


async def test_advisory_stage_surfaces_cto_constraints():
    from pydantic_agents.core.stages.advisory_stage import AdvisoryStage

    cto = AgentAdvice(
        approved_to_proceed=True,
        architectural_constraints=[ArchitecturalConstraint(description="Use errgroup", affected_agents=["backend"])],
    )
    provider = AsyncMock()

    async def run_agent(sem, name, agent, prompt, deps=None):  # noqa: ARG001
        return MagicMock(output=cto if name == "cto" else AgentAdvice())

    provider.run_agent.side_effect = run_agent
    state = PipelineState(run_id="r", stage="ba", user_stories=[UserStory(id="U", title="t", gherkin="g")])
    with (
        patch("pydantic_agents.core.cto_gate.apply", AsyncMock(return_value=True)),
        patch("pydantic_agents.core.stages.advisory_stage.registry.get", return_value=MagicMock()),
    ):
        result = await AdvisoryStage(agent_provider=provider).run(state, PipelineConfig(project="p", requirement="r"))
    assert "[constraint: backend] Use errgroup" in result.advisory_results["cto"]


async def test_registry_query_without_deps_factory_passes_no_deps_and_names_agent():
    from pydantic_agents.core.registry import AgentRegistry
    from pydantic_agents.core.session_recorder import current_run_context

    reg = AgentRegistry()
    seen = {}

    async def run_agent(**kwargs):
        seen["deps"] = kwargs["deps"]
        seen["agent_name"] = current_run_context.get().get("agent_name")
        return MagicMock(output="ok")

    reg._provider = MagicMock(run_agent=run_agent)
    reg.register("database", MagicMock(), ["db"])
    await reg.query("db", "hi")
    assert seen == {"deps": None, "agent_name": "database"}


async def test_registry_query_restores_run_context_after_nested_queries():
    """A delegated (nested) query must not clobber the outer run's context keys."""
    from pydantic_agents.core.registry import AgentRegistry
    from pydantic_agents.core.session_recorder import current_run_context

    reg = AgentRegistry()
    seen: list[dict] = []

    async def outer_run(**kwargs):
        seen.append(dict(current_run_context.get()))
        await reg.query("inner", "x")
        seen.append(dict(current_run_context.get()))  # outer context again after the nested call
        return MagicMock(output="ok")

    async def inner_run(**kwargs):
        seen.append(dict(current_run_context.get()))
        return MagicMock(output="ok")

    run_for = {"outer": outer_run, "inner": inner_run}
    reg._provider = MagicMock(run_agent=lambda **kwargs: run_for[kwargs["agent_name"]](**kwargs))
    reg.register("outer", MagicMock(), ["outer"])
    reg.register("inner", MagicMock(), ["inner"])

    token = current_run_context.set({"run_id": "r1"})
    try:
        await reg.query("outer", "go")
        after = dict(current_run_context.get())
    finally:
        current_run_context.reset(token)

    assert [c["agent_name"] for c in seen] == ["outer", "inner", "outer"]
    assert all(c["run_id"] == "r1" for c in seen)
    assert after == {"run_id": "r1"}


@pytest.mark.parametrize("agent, heading", [(cto_agent, "## Organisation Context"), (backend_agent, "## Project")])
def test_agents_without_deps_get_no_context_section(agent, heading):
    assert heading not in _captured_instructions(agent, None)


async def test_registry_query_uses_the_deps_factory():
    from pydantic_agents.core.registry import AgentRegistry

    reg = AgentRegistry()
    seen = {}

    async def run_agent(**kwargs):
        seen["deps"] = kwargs["deps"]
        return MagicMock(output="ok")

    reg._provider = MagicMock(run_agent=run_agent)
    reg.register("a", MagicMock(), ["cap"], lambda: {"from": "factory"})
    await reg.query("cap", "go")
    assert seen["deps"] == {"from": "factory"}


async def test_review_prompt_asks_to_verify_previous_feedback():
    from pydantic_agents.core.models.core import Artifact, QualityGateResult, ReviewResult, RiskLevel, SecurityReport
    from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

    prompts = {}
    outputs = {
        "sre": Artifact(agent_name="sre", files={"a.yaml": "x"}),
        "security": SecurityReport(passed=True, score=90),
        "quality_gate": QualityGateResult(complexity_score=10, feedback=[]),
        "review": ReviewResult(approved=True, feedback=[], security_score=90, risk_level=RiskLevel.LOW),
    }
    provider = AsyncMock()

    async def run_agent(sem, name, agent, prompt, deps=None):  # noqa: ARG001
        prompts[name] = prompt
        return MagicMock(output=outputs[name])

    provider.run_agent.side_effect = run_agent
    state = PipelineState(
        run_id="r",
        stage="implementation",
        feedback="Missing error handling",
        artifacts=[Artifact(agent_name="backend", files={"README.md": "x"})],
    )
    with (
        patch("pydantic_agents.core.stages.quality_gate_stage.registry.get", return_value=MagicMock()),
        patch(
            "pydantic_agents.core.stages.quality_gate_stage.CodeValidator.run_all_async", AsyncMock(return_value=True)
        ),
        patch("pydantic_agents.core.context_engine.retrieve_context", return_value=""),
    ):
        result = await QualityGateStage(agent_provider=provider).run(
            state, PipelineConfig(project="p", requirement="r")
        )

    assert result.approved is True
    assert "verify as resolved" in prompts["review"]
    assert "Missing error handling" in prompts["review"]
    assert "Address previous feedback" not in prompts["review"]


async def test_log_activity_prefers_deps_over_run_context():
    from dataclasses import dataclass

    from pydantic_agents.core.tools import memory

    @dataclass
    class Deps:
        agent_name: str = "from_deps"
        pass_num: int = 3

    with patch.object(memory.vector_db, "async_add_documents", AsyncMock()) as add:
        await memory.log_activity(MagicMock(deps=Deps()), "act", "summary")
    assert add.await_args.args[1][0].startswith("During pass 3, the from_deps agent")


async def test_search_codebase_reports_missing_ripgrep():
    from pydantic_agents.core.tools.search import search_codebase

    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError)):
        result = await search_codebase(MagicMock(), "x", ".")
    assert "ripgrep (rg) not installed" in result[0]


def test_search_pattern_in_content_returns_line_matches():
    from pydantic_agents.core.tools.search import search_pattern_in_content

    assert search_pattern_in_content(MagicMock(), r"^b", "a\n  bee\nb") == [{"line": 3, "content": "b"}]


@pytest.mark.parametrize(
    "returncode, stdout, stderr, expected",
    [(0, b"a.go\nb.go\n", b"", ["a.go", "b.go"]), (1, b"", b"", []), (2, b"", b"bad regex", ["rg error: bad regex"])],
)
async def test_search_codebase_maps_ripgrep_exit_codes(returncode, stdout, stderr, expected):
    from pydantic_agents.core.tools.search import search_codebase

    proc = MagicMock(returncode=returncode, communicate=AsyncMock(return_value=(stdout, stderr)))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        assert await search_codebase(MagicMock(), "x", ".") == expected
