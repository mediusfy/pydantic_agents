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
            mock_rec.record_span.return_value.__aenter__.return_value = MagicMock()
            result = await ba_node(state, config, mock_provider)
        assert isinstance(result, dict)
        assert result["stage"] == "ba"

    @pytest.mark.asyncio
    async def test_merge_node_returns_delta(self, state, config):
        from pydantic_agents.core.pipeline.langgraph.nodes import merge_node

        with patch("pydantic_agents.core.pipeline.langgraph.nodes.recorder") as mock_rec:
            mock_rec.record_span.return_value.__aenter__.return_value = MagicMock()
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
            mock_rec.record_span.return_value.__aenter__.return_value = MagicMock()
            result = await quality_gate_node(state, config, mock_provider)
        assert isinstance(result, dict)
        assert "approved" in result

    @pytest.mark.asyncio
    async def test_implementation_node_increments_attempts(self, state, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.nodes import implementation_node

        with patch("pydantic_agents.core.pipeline.langgraph.nodes.recorder") as mock_rec:
            mock_rec.record_span.return_value.__aenter__.return_value = MagicMock()
            result = await implementation_node(state, config, mock_provider)
        assert result.get("refinement_attempts") == 1

    @pytest.mark.asyncio
    async def test_advisory_node_returns_cto_approved(self, state, config, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.nodes import advisory_node

        with patch("pydantic_agents.core.pipeline.langgraph.nodes.recorder") as mock_rec:
            mock_rec.record_span.return_value.__aenter__.return_value = MagicMock()
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

    def test_route_after_implementation_runtime_error(self):
        from pydantic_agents.core.models.core import FailureReport
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_implementation

        s = PipelineState(
            run_id="r1",
            stage="implementation",
            failure_report=FailureReport(source="implementation", error_type="runtime_error", trace="crash"),
        )
        assert route_after_implementation(s) == "end"

    def test_route_after_quality_gate_retry_and_exhaustion(self, monkeypatch):
        from pydantic_agents.core.pipeline.langgraph.routing import route_after_quality_gate
        from pydantic_agents.core.settings import settings

        monkeypatch.setattr(settings, "max_retries", 3)
        retrying = PipelineState(run_id="r1", stage="quality_gate", approved=False, refinement_attempts=1)
        assert route_after_quality_gate(retrying) == "retry_implementation"

        exhausted = PipelineState(run_id="r1", stage="quality_gate", approved=False, refinement_attempts=3)
        assert route_after_quality_gate(exhausted) == "end"

    def test_route_retry_within_and_at_limit(self, monkeypatch):
        from pydantic_agents.core.pipeline.langgraph.routing import route_retry
        from pydantic_agents.core.settings import settings

        monkeypatch.setattr(settings, "max_retries", 3)
        s_ok = PipelineState(run_id="r1", stage="implementation", refinement_attempts=1)
        assert route_retry(s_ok) == "implementation"
        assert route_retry(PipelineState(run_id="r1", stage="implementation", refinement_attempts=3)) == "end"


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

    @pytest.mark.parametrize(
        ("blueprint", "expected_nodes", "expected_start"),
        [
            ("langgraph_quick", {"implementation", "quality_gate", "merge"}, "implementation"),
            ("quick_backend", {"implementation", "quality_gate", "merge"}, "implementation"),
            ("docs_update", {"implementation", "quality_gate", "merge"}, "implementation"),
            ("security_review", {"quality_gate", "merge"}, "quality_gate"),
        ],
    )
    def test_a_full_run_honours_the_blueprint_s_stages(self, mock_provider, blueprint, expected_nodes, expected_start):
        from pydantic_agents.core.pipeline.langgraph.graphs import build_full_graph

        cfg = PipelineConfig(blueprint=blueprint, framework="langgraph", requirement="ship it")
        graph = build_full_graph(cfg, mock_provider).get_graph()
        nodes = set(graph.nodes) - {"__start__", "__end__"}
        edges = {(e.source, e.target) for e in graph.edges}

        assert nodes == expected_nodes
        assert ("__start__", expected_start) in edges

    def test_build_full_graph_rejects_blueprint_without_stages(self, mock_provider):
        from pydantic_agents.core.pipeline.langgraph.graphs import build_full_graph

        cfg = PipelineConfig(blueprint="crew_sdlc", framework="langgraph")
        with pytest.raises(ValueError, match="has no stages to run"):
            build_full_graph(cfg, mock_provider)


class TestLangGraphExecutionAndResume:
    @pytest.mark.asyncio
    async def test_quick_blueprint_skips_ba_and_advisory_and_seeds_story(self, tmp_path, monkeypatch):
        from pydantic_agents.core.models.core import Artifact
        from pydantic_agents.core.pipeline.factory import build_pipeline
        from pydantic_agents.core.settings import settings

        monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "ckpts")
        monkeypatch.chdir(tmp_path)

        ba_run = AsyncMock()
        adv_run = AsyncMock()
        seen_stories = []

        async def fake_impl(_self, state, _cfg):
            seen_stories.extend(state.user_stories)
            return state.model_copy(
                update={"artifacts": [Artifact(agent_name="backend", files={"main.go": "package main"})]}
            )

        async def fake_qg(_self, state, _cfg):
            return state.model_copy(update={"approved": True})

        async def fake_merge(_self, state, _cfg):
            return state.model_copy(update={"stage": "merge", "approved": True})

        with (
            patch("pydantic_agents.core.stages.ba_stage.BAStage.run", ba_run),
            patch("pydantic_agents.core.stages.advisory_stage.AdvisoryStage.run", adv_run),
            patch("pydantic_agents.core.stages.implementation_stage.ImplementationStage.run", fake_impl),
            patch("pydantic_agents.core.stages.quality_gate_stage.QualityGateStage.run", fake_qg),
            patch("pydantic_agents.core.stages.merge_stage.MergeStage.run", fake_merge),
        ):
            cfg = PipelineConfig(
                project="quickproj",
                blueprint="langgraph_quick",
                framework="langgraph",
                requirement="Add GET /healthz",
            )
            result = await build_pipeline(cfg).execute()

        ba_run.assert_not_awaited()
        adv_run.assert_not_awaited()
        assert len(seen_stories) == 1
        assert seen_stories[0].id == "US-001"
        assert seen_stories[0].gherkin == "Add GET /healthz"
        assert result.approved is True
        assert result.stage == "merge"

    @pytest.mark.asyncio
    async def test_composite_pipeline_runs_crewai_implementation_step(self, tmp_path, monkeypatch):
        from pydantic_agents.core.models.core import Artifact, UserStory
        from pydantic_agents.core.pipeline.factory import build_pipeline
        from pydantic_agents.core.settings import settings

        monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "ckpts")
        monkeypatch.chdir(tmp_path)

        story = UserStory(id="US-1", title="Health", gherkin="Given /healthz")

        async def fake_ba(_self, state, _cfg):
            return state.model_copy(update={"user_stories": [story]})

        async def fake_adv(_self, state, _cfg):
            return state.model_copy(update={"cto_approved": True, "advisory_results": {"cto": "use net/http"}})

        async def fake_qg(_self, state, _cfg):
            return state.model_copy(update={"approved": True})

        async def fake_merge(_self, state, _cfg):
            return state.model_copy(update={"stage": "merge", "approved": True})

        crew_spy = AsyncMock(
            side_effect=lambda state, _cfg: state.model_copy(
                update={"artifacts": [Artifact(agent_name="backend", files={"main.go": "package main"})]}
            )
        )
        import sys
        from types import SimpleNamespace

        monkeypatch.setitem(
            sys.modules,
            "pydantic_agents.core.pipeline.crewai.bridge",
            SimpleNamespace(run_crewai_pipeline=crew_spy),
        )

        with (
            patch("pydantic_agents.core.stages.ba_stage.BAStage.run", fake_ba),
            patch("pydantic_agents.core.stages.advisory_stage.AdvisoryStage.run", fake_adv),
            patch("pydantic_agents.core.pipeline.factory._ensure_dependency", lambda _f: None),
            patch("pydantic_agents.core.stages.quality_gate_stage.QualityGateStage.run", fake_qg),
            patch("pydantic_agents.core.stages.merge_stage.MergeStage.run", fake_merge),
        ):
            cfg = PipelineConfig(
                project="compproj",
                blueprint="composite",
                framework="composite",
                requirement="Add GET /healthz",
            )
            result = await build_pipeline(cfg).execute()

        assert result.approved is True
        assert result.stage == "merge"
        assert len(result.artifacts) == 1
        assert crew_spy.await_count == 1

    @pytest.mark.asyncio
    async def test_persistent_checkpointer_resumes_across_instances_without_rerunning_ba(self, tmp_path, monkeypatch):
        from pydantic_agents.core.models.core import Artifact, UserStory
        from pydantic_agents.core.pipeline.factory import build_pipeline
        from pydantic_agents.core.settings import settings

        monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "ckpts")
        monkeypatch.chdir(tmp_path)

        story = UserStory(id="US-1", title="Health", gherkin="Given /healthz")
        ba_calls = 0
        cto_approved = False

        async def fake_ba(_self, state, _cfg):
            nonlocal ba_calls
            ba_calls += 1
            return state.model_copy(update={"user_stories": [story]})

        async def fake_adv(_self, state, _cfg):
            return state.model_copy(update={"cto_approved": cto_approved})

        async def fake_impl(_self, state, _cfg):
            return state.model_copy(
                update={"artifacts": [Artifact(agent_name="backend", files={"main.go": "package main"})]}
            )

        async def fake_qg(_self, state, _cfg):
            return state.model_copy(update={"approved": True})

        async def fake_merge(_self, state, _cfg):
            return state.model_copy(update={"stage": "merge", "approved": True})

        with (
            patch("pydantic_agents.core.stages.ba_stage.BAStage.run", fake_ba),
            patch("pydantic_agents.core.stages.advisory_stage.AdvisoryStage.run", fake_adv),
            patch("pydantic_agents.core.stages.implementation_stage.ImplementationStage.run", fake_impl),
            patch("pydantic_agents.core.stages.quality_gate_stage.QualityGateStage.run", fake_qg),
            patch("pydantic_agents.core.stages.merge_stage.MergeStage.run", fake_merge),
        ):
            first_cfg = PipelineConfig(
                project="resproj",
                blueprint="langgraph_full",
                framework="langgraph",
                requirement="Add GET /healthz",
                run_id="pipe_112233445566",
            )
            first_result = await build_pipeline(first_cfg).execute()
            assert first_result.approved is False
            assert first_result.failure_report is not None
            assert "CTO gate" in first_result.failure_report.trace
            assert ba_calls == 1

            # Fresh DiskCheckpointSaver loads the saved state from disk and resumes at advisory
            cto_approved = True
            resume_cfg = PipelineConfig(
                project="resproj",
                blueprint="langgraph_full",
                framework="langgraph",
                requirement="Add GET /healthz",
                resume=True,
            )
            resumed_result = await build_pipeline(resume_cfg).execute()

        assert ba_calls == 1  # BA was not re-run on resume!
        assert resumed_result.approved is True
        assert resumed_result.stage == "merge"

    def test_disk_checkpoint_saver_delete_and_corrupt_file(self, tmp_path):
        from pydantic_agents.core.pipeline.langgraph.checkpointer import DiskCheckpointSaver

        saver = DiskCheckpointSaver("p1", base_dir=tmp_path)
        assert saver.find_latest_run_id() is None

        # Corrupt file is ignored cleanly without raising
        corrupt_path = tmp_path / "p1" / "bad__langgraph.ckpt"
        corrupt_path.write_text("{not-valid-json", encoding="utf-8")
        saver2 = DiskCheckpointSaver("p1", base_dir=tmp_path)
        assert saver2.find_latest_run_id() == "bad"
        saver2.delete_thread("bad")
        assert not corrupt_path.exists()
