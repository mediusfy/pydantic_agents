from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig


@pytest.fixture
def mock_provider():
    p = AsyncMock()
    p.run_agent = AsyncMock(return_value=MagicMock(data=MagicMock()))
    return p


@pytest.fixture(autouse=True)
def register_agents():
    from pydantic_agents.agents.registration import register_all_agents

    register_all_agents()


@pytest.fixture
def state():
    return PipelineState(run_id="test-1", stage="init", requirement="test requirement")


@pytest.fixture
def config():
    return PipelineConfig(blueprint="langgraph_full", framework="langgraph", requirement="test requirement")


class TestLangGraphNodes:
    @pytest.mark.asyncio
    async def test_ba_node_returns_delta(self, state, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.nodes import ba_node

        with patch("pydantic_agents.core.pipeline.langgraph.nodes.recorder") as mock_rec:
            mock_rec.record_span.return_value.__aenter__.return_value = AsyncMock()
            result = await ba_node(state, config, mock_provider)
        assert isinstance(result, dict)
        assert result["stage"] == "ba"

    @pytest.mark.asyncio
    async def test_merge_node_returns_delta(self, state, config):
        from pydantic_agents.core.pipeline.langgraph.nodes import merge_node

        with patch("pydantic_agents.core.pipeline.langgraph.nodes.recorder") as mock_rec:
            mock_rec.record_span.return_value.__aenter__.return_value = AsyncMock()
            result = await merge_node(state, config)
        assert isinstance(result, dict)
        assert "stage" in result

    def test_error_delta_contains_failure_report(self, state, config):
        from pydantic_agents.core.pipeline.langgraph.nodes import _make_error

        result = _make_error(state, "test_agent", "something broke")
        assert "errors" in result
        assert "failure_report" in result
        assert result["failure_report"].source == "test_agent"
        assert result["failure_report"].error_type == "runtime_error"

    @pytest.mark.asyncio
    async def test_quality_gate_node_includes_approved(self, state, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.nodes import quality_gate_node

        with patch("pydantic_agents.core.pipeline.langgraph.nodes.recorder") as mock_rec:
            mock_rec.record_span.return_value.__aenter__.return_value = AsyncMock()
            result = await quality_gate_node(state, config, mock_provider)
        assert isinstance(result, dict)
        assert "approved" in result

    @pytest.mark.asyncio
    async def test_implementation_node_increments_attempts(self, state, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.nodes import implementation_node

        with patch("pydantic_agents.core.pipeline.langgraph.nodes.recorder") as mock_rec:
            mock_rec.record_span.return_value.__aenter__.return_value = AsyncMock()
            result = await implementation_node(state, config, mock_provider)
        assert result.get("refinement_attempts") == 1

    @pytest.mark.asyncio
    async def test_advisory_node_returns_cto_approved(self, state, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.nodes import advisory_node

        with patch("pydantic_agents.core.pipeline.langgraph.nodes.recorder") as mock_rec:
            mock_rec.record_span.return_value.__aenter__.return_value = AsyncMock()
            result = await advisory_node(state, config, mock_provider)
        assert isinstance(result, dict)
        assert "cto_approved" in result


class TestLangGraphRouting:
    def test_route_after_ba_with_stories(self):
        from pydantic_agents.core.models.core import UserStory
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_ba

        s = PipelineState(
            run_id="r1",
            stage="ba",
            user_stories=[UserStory(id="US-1", title="Test", gherkin="Given...")],
        )
        assert route_after_ba(s) == "advisory"

    def test_route_after_ba_with_errors(self):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_ba

        s = PipelineState(run_id="r1", stage="ba", errors=["ba: failed"])
        assert route_after_ba(s) == "end"

    def test_route_after_ba_no_stories(self):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_ba

        s = PipelineState(run_id="r1", stage="ba")
        assert route_after_ba(s) == "end"

    def test_route_after_advisory_approved(self):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_advisory

        s = PipelineState(run_id="r1", stage="advisory", cto_approved=True)
        assert route_after_advisory(s) == "implementation"

    def test_route_after_advisory_rejected(self):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_advisory

        s = PipelineState(run_id="r1", stage="advisory", cto_approved=False)
        assert route_after_advisory(s) == "end"

    def test_route_after_implementation_test_failure(self):
        from pydantic_agents.core.models.core import FailureReport
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_implementation

        s = PipelineState(
            run_id="r1",
            stage="implementation",
            failure_report=FailureReport(source="test_suite", error_type="test_failure", trace="tests failed"),
        )
        assert route_after_implementation(s) == "retry_implementation"

    def test_route_after_implementation_no_failure(self):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_implementation

        s = PipelineState(run_id="r1", stage="implementation")
        assert route_after_implementation(s) == "quality_gate"

    def test_route_after_quality_gate_approved(self):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_quality_gate

        s = PipelineState(run_id="r1", stage="quality_gate", approved=True, refinement_attempts=1)
        assert route_after_quality_gate(s) == "end"

    def test_route_after_implementation_subgraph_approved(self):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_implementation_subgraph

        s = PipelineState(run_id="r1", stage="quality_gate", approved=True)
        assert route_after_implementation_subgraph(s) == "merge"

    def test_route_after_implementation_subgraph_not_approved(self):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_implementation_subgraph

        s = PipelineState(run_id="r1", stage="quality_gate", approved=False)
        assert route_after_implementation_subgraph(s) == "end"


class TestLangGraphGraphBuilder:
    def test_build_implementation_subgraph(self, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.graphs import build_implementation_subgraph

        graph = build_implementation_subgraph(config, mock_provider)
        assert graph is not None

    def test_build_full_graph(self, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.graphs import build_full_graph

        graph = build_full_graph(config, mock_provider)
        assert graph is not None

    def test_full_graph_with_checkpointer(self, config, mock_provider):
        from langgraph.checkpoint.memory import MemorySaver

        from pydantic_agents.core.pipeline.langgraph.graphs import build_full_graph

        checkpointer = MemorySaver()
        graph = build_full_graph(config, mock_provider, checkpointer=checkpointer)
        assert graph is not None


class TestBuildFullGraphNodes:
    @staticmethod
    def _nodes(config, provider):
        from pydantic_agents.core.pipeline.langgraph.graphs import build_full_graph

        return set(build_full_graph(config, provider).get_graph().nodes) - {"__start__", "__end__"}

    def test_a_full_run_has_every_stage_node_and_merge(self, config, mock_provider):
        assert self._nodes(config, mock_provider) == {"ba", "advisory", "implementation", "quality_gate", "merge"}

    @pytest.mark.parametrize(
        ("step", "stages", "expected"),
        [
            ("implementation", ["implementation"], {"implementation", "merge"}),
            ("advisory", ["advisory", "implementation"], {"advisory", "implementation", "merge"}),
            ("ba", ["ba"], {"ba", "merge"}),
        ],
    )
    def test_a_single_step_run_has_only_the_blueprint_s_stages(
        self, monkeypatch, config, mock_provider, step, stages, expected
    ):
        from pydantic_agents.core.pipeline.langgraph import graphs

        monkeypatch.setattr(graphs, "BLUEPRINTS", {config.blueprint: {"stages": stages}})
        config.step = step

        assert self._nodes(config, mock_provider) == expected

    def test_a_single_step_run_starts_at_the_first_stage_in_pipeline_order(self, monkeypatch, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph import graphs

        monkeypatch.setattr(graphs, "BLUEPRINTS", {config.blueprint: {"stages": ["implementation", "advisory"]}})
        config.step = "advisory"

        edges = {(e.source, e.target) for e in graphs.build_full_graph(config, mock_provider).get_graph().edges}

        assert ("__start__", "advisory") in edges
        assert ("advisory", "__end__") in edges
