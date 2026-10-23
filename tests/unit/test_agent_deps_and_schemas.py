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
    """Run the agent on a stub model and return the instructions it sent."""
    with agent.override(model=TestModel()):
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
    assert "payments" not in text and "notifications" not in text


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
