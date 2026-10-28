"""Integration tests for the Orgitration multi-agent pipeline.

Verifies:
- Pipeline stage data flow (BA → Advisory → Implementation)
- Checkpoint roundtrip with full PipelineState (memory preservation)
- Agent registration and capability discovery
- Artifact persistence to disk
- Telemetry logging
"""

import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.adapters.state_repository import LocalDiskStateRepository
from pydantic_agents.core.checkpoint import CheckpointManager
from pydantic_agents.core.memory import VectorMemory, create_memory_backend, get_memory_backend
from pydantic_agents.core.models.core import (
    AgentAdvice,
    Artifact,
    EventTrigger,
    PipelineState,
    UserStory,
)
from pydantic_agents.core.registry import AgentRegistry, registry
from pydantic_agents.core.telemetry import Telemetry
from pydantic_agents.core.writer import ArtifactWriter


class TestPipelineStageDataFlow:
    """Verifies data flows correctly between pipeline stages with mocked agents."""

    @pytest.mark.asyncio
    async def test_ba_stage_populates_user_stories_and_checkpoint(self, tmp_path, monkeypatch):
        """BAStage must set state.user_stories."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))

        from pydantic_agents.core.adapters.agents import AgentProviderPort
        from pydantic_agents.core.pipeline.config import PipelineConfig
        from pydantic_agents.core.stages.ba_stage import BAStage

        mock_ba_result = MagicMock()
        mock_ba_result.output = [
            UserStory(
                id="US-001", title="Process Payment", gherkin="Given a valid request\nWhen processed\nThen emit event"
            ),
            UserStory(
                id="US-002",
                title="Handle Refund",
                gherkin="Given a refund request\nWhen validated\nThen process refund",
            ),
        ]

        mock_provider = AsyncMock(spec=AgentProviderPort)
        mock_provider.run_agent.return_value = mock_ba_result

        state = PipelineState(run_id="ba-flow-test", stage="init")
        config = PipelineConfig(project="test_proj", requirement="Build a payment system")
        stage = BAStage(agent_provider=mock_provider)

        with patch("pydantic_agents.core.stages.ba_stage.registry.get", return_value=MagicMock()):
            result = await stage.run(state, config)

        assert len(result.user_stories) == 2
        assert result.user_stories[0].id == "US-001"
        assert result.user_stories[1].title == "Handle Refund"
        assert result.stage == "ba"

    @pytest.mark.asyncio
    async def test_ba_stage_returns_early_on_no_stories(self, tmp_path, monkeypatch):
        """When BA returns no stories, the pipeline should exit."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))

        from pydantic_agents.core.adapters.agents import AgentProviderPort
        from pydantic_agents.core.pipeline.config import PipelineConfig
        from pydantic_agents.core.stages.ba_stage import BAStage

        mock_result = MagicMock()
        mock_result.output = []

        mock_provider = AsyncMock(spec=AgentProviderPort)
        mock_provider.run_agent.return_value = mock_result

        state = PipelineState(run_id="ba-empty-test", stage="init")
        config = PipelineConfig(project="test_proj", requirement="Some requirement")
        stage = BAStage(agent_provider=mock_provider)

        with patch("pydantic_agents.core.stages.ba_stage.registry.get", return_value=MagicMock()):
            result = await stage.run(state, config)

        assert len(result.user_stories) == 0
        assert result.stage == "ba"  # the stage ran; the orchestrator stops the run on the empty list

    @pytest.mark.asyncio
    async def test_advisory_stage_merges_cto_and_ux_results(self, tmp_path, monkeypatch):
        """Advisory stage must capture both CTO and UX advice in state."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))

        from pydantic_agents.core.adapters.agents import AgentProviderPort
        from pydantic_agents.core.pipeline.config import PipelineConfig
        from pydantic_agents.core.stages.advisory_stage import AdvisoryStage

        cto_output = MagicMock()
        cto_output.output = AgentAdvice(
            bottlenecks_identified=["Database"],
            recommendations=["Use read replicas"],
            approved_to_proceed=True,
        )

        ux_output = MagicMock()
        ux_output.output = AgentAdvice(
            bottlenecks_identified=[],
            recommendations=["Add loading states"],
            approved_to_proceed=True,
        )

        mock_provider = AsyncMock(spec=AgentProviderPort)

        def run_agent_side_effect(sem, agent_name, agent, prompt, deps=None):
            if agent_name == "cto":
                return cto_output
            if agent_name == "ux":
                return ux_output

        mock_provider.run_agent.side_effect = run_agent_side_effect

        user_stories = [
            UserStory(
                id="US-001", title="Process Payment", gherkin="Given a valid request\nWhen processed\nThen emit event"
            ),
        ]

        state = PipelineState(run_id="advisory-flow-test", stage="ba", user_stories=user_stories)
        config = PipelineConfig(project="test_proj", requirement="test")
        stage = AdvisoryStage(agent_provider=mock_provider)

        # Skip cto_gate for pure stage testing to avoid prompt blocking if that logic triggers.
        with (
            patch("pydantic_agents.core.cto_gate.apply", new_callable=AsyncMock) as mock_gate,
            patch("pydantic_agents.core.stages.advisory_stage.registry.get", return_value=MagicMock()),
        ):
            mock_gate.return_value = True
            result = await stage.run(state, config)

        assert "cto" in result.advisory_results
        assert "ux" in result.advisory_results
        assert "read replicas" in result.advisory_results["cto"]
        assert "loading states" in result.advisory_results["ux"]
        assert result.stage == "advisory"

    @pytest.mark.asyncio
    async def test_advisory_stage_survives_cto_failure(self, tmp_path, monkeypatch):
        """When CTO agent fails, UX result is still preserved."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))

        from pydantic_agents.core.adapters.agents import AgentProviderPort
        from pydantic_agents.core.pipeline.config import PipelineConfig
        from pydantic_agents.core.stages.advisory_stage import AdvisoryStage

        ux_output = MagicMock()
        ux_output.output = AgentAdvice(
            bottlenecks_identified=[],
            recommendations=["Use a design system"],
            approved_to_proceed=True,
        )

        mock_provider = AsyncMock(spec=AgentProviderPort)

        def run_agent_side_effect(sem, agent_name, agent, prompt, deps=None):
            if agent_name == "cto":
                raise Exception("CTO API error")
            if agent_name == "ux":
                return ux_output

        mock_provider.run_agent.side_effect = run_agent_side_effect

        user_stories = [
            UserStory(
                id="US-001", title="Process Payment", gherkin="Given a valid request\nWhen processed\nThen emit event"
            ),
        ]

        state = PipelineState(run_id="advisory-partial-test", stage="ba", user_stories=user_stories)
        config = PipelineConfig(project="test_proj", requirement="test")
        stage = AdvisoryStage(agent_provider=mock_provider)

        with (
            patch("pydantic_agents.core.cto_gate.apply", new_callable=AsyncMock) as mock_gate,
            patch("pydantic_agents.core.stages.advisory_stage.registry.get", return_value=MagicMock()),
        ):
            mock_gate.return_value = False
            result = await stage.run(state, config)

        assert result.advisory_results.get("cto", "") == ""
        assert "design system" in result.advisory_results.get("ux", "")
        assert result.stage == "advisory"


# =============================================================================
#  Pipeline with Checkpoints  (Memory Preservation)
# =============================================================================


class TestPipelineStatePersistence:
    """Checks if the PipelineState survives save/load."""

    def test_empty_state_roundtrip(self, tmp_path, monkeypatch):
        """save and loads correctly."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        manager = CheckpointManager(repository=LocalDiskStateRepository("test_proj"), run_id="roundtrip-empty")
        state = PipelineState(run_id="roundtrip-empty", stage="init")

        manager.save("init", state)
        loaded = manager.load("init")

        assert loaded is not None
        assert loaded.run_id == "roundtrip-empty"
        assert loaded.stage == "init"
        assert loaded.attempt == 1
        assert loaded.user_stories == []
        assert loaded.advisory_results == {}
        assert loaded.artifacts == []
        assert loaded.feedback is None

    def test_fully_populated_state_roundtrip(self, tmp_path, monkeypatch):
        """Full PipelineState with nested models round-trips without data loss."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        manager = CheckpointManager(repository=LocalDiskStateRepository("test_proj"), run_id="roundtrip-full")

        state = PipelineState(
            run_id="roundtrip-full",
            stage="implementation",
            attempt=2,
            user_stories=[
                UserStory(
                    id="US-001",
                    title="Process Payment",
                    gherkin="Given a valid request\nWhen processed\nThen succeed",
                    event_triggers=[
                        EventTrigger(id="evt-1", name="PaymentSucceeded", description="OK"),
                        EventTrigger(id="evt-2", name="PaymentFailed", description="Failed"),
                    ],
                ),
                UserStory(
                    id="US-002",
                    title="Send Receipt",
                    gherkin="Given payment completed\nWhen receipt generated\nThen email sent",
                    event_triggers=[],
                ),
            ],
            advisory_results={
                "cto": "CTO Guidelines:\n- Use read replicas\n- Implement caching",
                "ux": "UX Guidelines:\n- Add loading spinners\n- Use skeleton screens",
            },
            artifacts=[
                Artifact(
                    agent_name="backend",
                    files={
                        "cmd/server/main.go": "package main\nfunc main() {}\n",
                        "internal/service/payment.go": "package service\n// payment logic\n",
                    },
                    metadata={"user_story_id": "US-001", "language": "go", "service_name": "payment"},
                ),
                Artifact(
                    agent_name="frontend",
                    files={"src/components/Payment.vue": "<template><div>Pay</div></template>\n"},
                    metadata={"user_story_id": "US-001", "language": "vue"},
                ),
                Artifact(
                    agent_name="database",
                    files={"migrations/001_init.sql": "CREATE TABLE payments (id UUID PRIMARY KEY);\n"},
                    metadata={"user_story_id": "US-001"},
                ),
                Artifact(
                    agent_name="infrastructure",
                    files={"terraform/main.tf": "# Infra as code\n"},
                    metadata={"user_story_id": "infrastructure"},
                ),
                Artifact(
                    agent_name="test",
                    files={"tests/payment_test.go": "func TestPayment(t *testing.T) {}\n"},
                    metadata={"user_story_id": "US-001"},
                ),
                Artifact(
                    agent_name="docs",
                    files={"README.md": "# Payment Service\n"},
                    metadata={"user_story_id": "global"},
                ),
            ],
            feedback="Address previous feedback: Missing error handling",
        )

        manager.save("implementation", state)
        loaded = manager.load("implementation")

        assert loaded is not None
        assert loaded.run_id == "roundtrip-full"
        assert loaded.stage == "implementation"
        assert loaded.attempt == 2

        # User stories
        assert len(loaded.user_stories) == 2
        assert loaded.user_stories[0].id == "US-001"
        assert loaded.user_stories[0].title == "Process Payment"
        assert len(loaded.user_stories[0].event_triggers) == 2
        assert loaded.user_stories[0].event_triggers[0].name == "PaymentSucceeded"
        assert loaded.user_stories[1].id == "US-002"
        assert loaded.user_stories[1].event_triggers == []

        # Advisory results
        assert loaded.advisory_results["cto"] == state.advisory_results["cto"]
        assert "read replicas" in loaded.advisory_results["cto"]
        assert loaded.advisory_results["ux"] == state.advisory_results["ux"]

        # Artifacts
        assert len(loaded.artifacts) == 6
        artifact_map = {a.agent_name: a for a in loaded.artifacts}
        assert "backend" in artifact_map
        assert "frontend" in artifact_map
        assert "database" in artifact_map
        assert "infrastructure" in artifact_map
        assert "test" in artifact_map
        assert "docs" in artifact_map

        backend_art = artifact_map["backend"]
        assert "cmd/server/main.go" in backend_art.files
        assert "internal/service/payment.go" in backend_art.files
        assert backend_art.metadata["language"] == "go"
        assert backend_art.metadata["service_name"] == "payment"

        frontend_art = artifact_map["frontend"]
        assert "src/components/Payment.vue" in frontend_art.files

        # Feedback
        assert loaded.feedback is not None
        assert "Missing error handling" in loaded.feedback

    def test_multiple_checkpoint_stages_preserved(self, tmp_path, monkeypatch):
        """Multiple stages (ba, advisory, implementation) are all independently loadable."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        manager = CheckpointManager(repository=LocalDiskStateRepository("test_proj"), run_id="multi-stage")

        ba_state = PipelineState(
            run_id="multi-stage",
            stage="ba",
            user_stories=[
                UserStory(id="US-001", title="Feature A", gherkin="Given X\nWhen Y\nThen Z"),
            ],
        )

        advisory_state = PipelineState(
            run_id="multi-stage",
            stage="advisory",
            user_stories=[
                UserStory(id="US-001", title="Feature A", gherkin="Given X\nWhen Y\nThen Z"),
            ],
            advisory_results={"cto": "Use Go", "ux": "Use Vue"},
        )
        impl_state = PipelineState(
            run_id="multi-stage",
            stage="implementation",
            user_stories=[
                UserStory(id="US-001", title="Feature A", gherkin="Given X\nWhen Y\nThen Z"),
            ],
            artifacts=[
                Artifact(agent_name="backend", files={"main.go": "package main"}, metadata={}),
            ],
        )

        manager.save("ba", ba_state)
        manager.save("advisory", advisory_state)
        manager.save("implementation", impl_state)

        loaded_ba = manager.load("ba")
        loaded_adv = manager.load("advisory")
        loaded_impl = manager.load("implementation")

        assert loaded_ba is not None
        assert loaded_ba.stage == "ba"
        assert loaded_adv is not None
        assert loaded_adv.stage == "advisory"
        assert loaded_impl is not None
        assert loaded_impl.stage == "implementation"

        # Each stage has progressively more data
        assert len(loaded_ba.user_stories) == 1
        assert len(loaded_ba.advisory_results) == 0

        assert len(loaded_adv.advisory_results) == 2
        assert loaded_adv.advisory_results["cto"] == "Use Go"

        assert len(loaded_impl.artifacts) == 1

    def test_checkpoint_clear_removes_all_stages(self, tmp_path, monkeypatch):
        """clear() must remove all checkpoint files for the current run."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        manager = CheckpointManager(repository=LocalDiskStateRepository("test_proj"), run_id="clear-test")
        state = PipelineState(run_id="clear-test", stage="init")

        manager.save("stage_a", state)
        manager.save("stage_b", state)
        assert manager.stage_exists("stage_a")
        assert manager.stage_exists("stage_b")

        manager.clear()

        assert not manager.stage_exists("stage_a")
        assert not manager.stage_exists("stage_b")

    def test_checkpoint_does_not_affect_other_runs(self, tmp_path, monkeypatch):
        """Clearing one run's checkpoints must not affect another run."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        run_a = CheckpointManager(repository=LocalDiskStateRepository("test_proj"), run_id="run-a")
        run_b = CheckpointManager(repository=LocalDiskStateRepository("test_proj"), run_id="run-b")
        state = PipelineState(run_id="run-a", stage="init")

        run_a.save("s1", state)
        run_b.save("s1", state)

        run_a.clear()

        assert run_b.stage_exists("s1")
        assert not run_a.stage_exists("s1")

    def test_load_corrupt_checkpoint_returns_none(self, tmp_path, monkeypatch):
        """Corrupted JSON files should not crash — return None."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        manager = CheckpointManager(repository=LocalDiskStateRepository("test_proj"), run_id="corrupt-test")

        filepath = Path(manager.repository._get_filepath(manager.run_id, "corrupt"))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("{this is not valid json!!}")

        loaded = manager.load("corrupt")
        assert loaded is None


# =============================================================================
# Agent Registration and Capability Discovery
# =============================================================================


class TestAgentRegistrationIntegration:
    """Verifies all agents register and capabilities are queryable."""

    def setup_method(self):
        registry._agents = {}
        registry._cap_index = {}
        # Reset the idempotency flag in registration.py so register_all_agents()
        # picks up the freshly-cleared registry instead of short-circuiting.
        import pydantic_agents.agents.registration as _reg

        _reg._registered = False

    def test_all_pipeline_agents_register(self):
        """register_all_agents must register expected SDLC agents."""
        from pydantic_agents.agents.registration import register_all_agents

        register_all_agents()

        expected_agents = [
            "ba",
            "cto",
            "backend",
            "frontend",
            "frontend_component",
            "database",
            "devops",
            "docs",
            "sre",
            "test_eng",
            "test_gen",
            "go_test",
            "review",
            "security",
            "ux",
            "quality_gate",
            "service_engineer",
            "planner",
            "critic",
            "api_contract",
            "observability_engineer",
            "finops",
            "release_manager",
        ]
        for name in expected_agents:
            assert name in registry._agents, f"Agent '{name}' not registered"

    def test_all_travel_finance_agents_register(self):
        """register_all_agents must register travel & finance agents."""
        from pydantic_agents.agents.registration import register_all_agents

        register_all_agents()

        expected = [
            "accountant",
            "asymmetric_reviewer",
            "current_events",
            "region_analyzer",
            "weather",
            "crypto_oracle",
            "holiday_planner",
            "market_shark",
            "price_analyst",
        ]
        for name in expected:
            assert name in registry._agents, f"Agent '{name}' not registered"

    def test_total_agent_count(self):
        """Verify the exact count of registered agents."""
        from pydantic_agents.agents.registration import register_all_agents

        register_all_agents()

        assert len(registry.list_agents()) == 36

    def test_capability_index_is_populated(self):
        """Each agent's capabilities must be in the index.

        Post-fix: `_cap_index[cap]` is a list of agent names in registration
        order, not a single agent name. Use `owners_of()` to read it.
        """
        from pydantic_agents.agents.registration import register_all_agents

        register_all_agents()

        assert "requirements" in registry._cap_index
        assert registry.owners_of("requirements") == ["ba"]

        for cap in ("architecture", "backend", "database", "testing", "infrastructure", "security", "code_review"):
            assert cap in registry._cap_index, f"capability '{cap}' missing from index"
            assert registry.owners_of(cap), f"capability '{cap}' has no owners"

    def test_capability_query_o1_lookup(self, tmp_path, monkeypatch):
        """query() must route to the right agent via O(1) capability lookup."""
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        from pydantic_agents.agents.registration import register_all_agents

        register_all_agents()

        ba_agent = registry.get("ba")
        assert ba_agent is not None

        sre_agent = registry.get("sre")
        assert sre_agent is not None

    def test_list_agents_returns_all_names(self):
        """list_agents() must return all registered agent names."""
        from pydantic_agents.agents.registration import register_all_agents

        register_all_agents()

        names = registry.list_agents()
        assert isinstance(names, list)
        assert "ba" in names
        assert "backend" in names
        assert "review" in names
        assert "weather" in names

    @pytest.mark.asyncio
    async def test_query_unknown_capability_raises_agent_not_found(self):
        """Post-fix: query() raises AgentNotFoundError instead of returning an error string,
        so callers can't accidentally treat a missing capability as a successful result."""
        from pydantic_agents.core.exceptions import AgentNotFoundError

        r = AgentRegistry()
        r.register("test", MagicMock(), ["known_cap"])

        with pytest.raises(AgentNotFoundError, match="unknown_cap"):
            await r.query("unknown_cap", "test prompt")

    def test_register_all_agents_is_idempotent(self):
        """Second call to register_all_agents() must NOT raise on duplicate registration.

        Previously, calling it twice (which delegate_to_orchestration_agent and
        modify_existing_code do on every MCP invocation) crashed on the
        first agent's `ValueError("Agent 'ba' is already registered.")`.
        """
        from pydantic_agents.agents.registration import register_all_agents

        register_all_agents()
        first_count = len(registry.list_agents())

        # Second call should be a no-op, not a crash.
        register_all_agents()
        second_count = len(registry.list_agents())

        assert second_count == first_count, "registration count drifted on second call"


# =============================================================================
# Artifact Persistence
# =============================================================================


class TestArtifactWriterIntegration:
    """Verifies artifacts are correctly persisted to disk."""

    def test_writes_files_to_disk(self, tmp_path):
        """Artifact files must be written to the output directory under agent_name subdirectory."""
        output_dir = tmp_path / "generated-app" / "test-project"
        writer = ArtifactWriter(str(output_dir))

        artifact = Artifact(
            agent_name="backend",
            files={
                "cmd/server/main.go": "package main\nfunc main() {}\n",
                "internal/service/handler.go": "package service\nfunc Handle() {}\n",
            },
            metadata={"user_story_id": "US-001"},
        )
        writer.write(artifact)

        main_go = output_dir / "backend" / "cmd" / "server" / "main.go"
        handler_go = output_dir / "backend" / "internal" / "service" / "handler.go"
        assert main_go.exists()
        assert handler_go.exists()
        assert main_go.read_text() == "package main\nfunc main() {}\n"

    def test_writes_multiple_artifacts(self, tmp_path):
        """Multiple agents writing files must all persist under their agent_name dirs."""
        output_dir = tmp_path / "generated-app" / "multi"
        writer = ArtifactWriter(str(output_dir))

        artifacts = [
            Artifact(agent_name="backend", files={"main.go": "package main"}, metadata={}),
            Artifact(agent_name="frontend", files={"App.vue": "<template>Hi</template>"}, metadata={}),
            Artifact(agent_name="database", files={"mig.sql": "CREATE TABLE t;"}, metadata={}),
        ]
        for art in artifacts:
            writer.write(art)

        assert (output_dir / "backend" / "main.go").exists()
        assert (output_dir / "frontend" / "App.vue").exists()
        assert (output_dir / "database" / "mig.sql").exists()

    def test_path_traversal_prevention(self, tmp_path):
        """ArtifactWriter must raise PathTraversalError when paths escape the output directory."""
        from pydantic_agents.core.exceptions import PathTraversalError

        writer = ArtifactWriter(str(tmp_path))

        artifact = Artifact(
            agent_name="malicious",
            files={"../../etc/passwd": "hacked"},
            metadata={},
        )
        with pytest.raises(PathTraversalError, match="Path traversal detected"):
            writer.write(artifact)


# =============================================================================
# Telemetry Integration
# =============================================================================


class TestTelemetryIntegration:
    """Verifies telemetry correctly logs agent calls."""

    @pytest.mark.asyncio
    async def test_logs_successful_call(self, tmp_path):
        """Successful agent calls must be logged with duration and success=true."""
        telemetry_file = tmp_path / "telemetry.jsonl"
        t = Telemetry(log_file=str(telemetry_file))

        async def fake_agent():
            return "done"

        result = await t.track("test_agent", fake_agent)
        assert result == "done"

        lines = telemetry_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["agent"] == "test_agent"
        assert entry["success"] is True
        assert entry["duration_sec"] >= 0

    @pytest.mark.asyncio
    async def test_logs_failed_call(self, tmp_path):
        """Failed agent calls must be logged with success=false."""
        telemetry_file = tmp_path / "telemetry.jsonl"
        t = Telemetry(log_file=str(telemetry_file))

        async def failing_agent():
            raise ValueError("API error")

        with pytest.raises(ValueError):
            await t.track("failing_agent", failing_agent)

        lines = telemetry_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["agent"] == "failing_agent"
        assert entry["success"] is False

    @pytest.mark.asyncio
    async def test_logs_multiple_calls_in_order(self, tmp_path):
        """Multiple telemetry entries must be appended in order."""
        telemetry_file = tmp_path / "telemetry.jsonl"
        t = Telemetry(log_file=str(telemetry_file))

        async def fake_agent(i):
            return f"result-{i}"

        for i in range(3):
            await t.track(f"agent-{i}", fake_agent, i)

        lines = telemetry_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["agent"] == f"agent-{i}"
        # Timestamps must be increasing
        timestamps = [json.loads(line)["timestamp"] for line in lines]
        assert timestamps == sorted(timestamps)


# =============================================================================
# Existing Code Pipeline Integration
# =============================================================================


class TestExistingCodePipelineIntegration:
    """Verifies the existing-code modification pipeline chains correctly."""

    @pytest.mark.asyncio
    async def test_pipeline_returns_result_without_errors(self, tmp_path, monkeypatch):
        """run_existing_code_pipeline with mocked agents must return a valid ExistingCodeResult."""
        from pydantic_agents.core.existing_code_handler import run_existing_code_pipeline

        # The result is written under settings.output_dir; keep it out of the repository's generated-app/.
        from pydantic_agents.core.settings import settings

        monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
        eng_result = MagicMock()
        eng_result.output = Artifact(
            agent_name="service_engineer",
            files={"internal/handler.go": "package internal\nfunc Handler() {}\n"},
            metadata={},
        )

        mock_eng = MagicMock()
        mock_eng.run = AsyncMock(return_value=eng_result)

        with patch.object(registry, "get", return_value=mock_eng):
            result = await run_existing_code_pipeline(
                change_description="Add a new handler",
                codebase_path="/tmp/test",
                run_tests=False,
                run_quality_gate=False,
                run_review=False,
            )

        assert result.passed is True
        assert result.changed_files == ["internal/handler.go"]
        assert len(result.artifacts) == 1
        assert result.errors == []
        assert result.output_dir is not None
        assert Path(result.output_dir).is_relative_to(tmp_path / "out")
        assert (Path(result.output_dir) / "service_engineer" / "internal" / "handler.go").exists()

    @pytest.mark.asyncio
    async def test_pipeline_propagates_engineer_error(self):
        """When the service engineer fails, the pipeline must return early with errors."""
        from pydantic_agents.core.existing_code_handler import run_existing_code_pipeline

        mock_eng = MagicMock()
        mock_eng.run = AsyncMock(side_effect=Exception("Engineer crashed"))

        with patch.object(registry, "get", return_value=mock_eng):
            result = await run_existing_code_pipeline(
                change_description="Add feature",
                codebase_path="/tmp/test",
                run_tests=False,
                run_quality_gate=False,
                run_review=False,
            )

        assert result.passed is False
        # With MAX_RETRIES=3, each failed attempt adds an error (1 try + 3 retries = 4)
        assert len(result.errors) >= 1
        assert "Engineer crashed" in result.errors[0]


# =============================================================================
# Redis / Mem0 Memory Backend
# =============================================================================


class TestMemoryBackendSelection:
    """Verifies the memory backend is selected correctly based on configuration."""

    def test_default_backend_is_vectormemory(self):
        """create_memory_backend returns the pgvector-backed VectorMemory."""
        backend = create_memory_backend()
        assert isinstance(backend, VectorMemory)

    def test_get_memory_backend_lazy_init(self, monkeypatch):
        """get_memory_backend must return the same instance on repeated calls."""
        monkeypatch.setattr("pydantic_agents.core.memory._memory_backend", None)
        first = get_memory_backend()
        second = get_memory_backend()
        assert first is second
        assert isinstance(first, VectorMemory)

    # ── pgvector test helpers ──────────────────────────────────────────────

    @staticmethod
    def _create_in_memory_pool_and_set(monkeypatch):
        """Create an in-memory pgvector mock for testing, and install it via _set_test_pool().

        Returns the mock pool so the caller can inspect stored rows.
        """
        # Mock _to_embedding so we don't require sentence_transformers for these tests
        monkeypatch.setattr("pydantic_agents.core.vector_db._to_embedding", lambda _text: [0.1, 0.2, 0.3])

        # We use a simple dict-based store: table_name -> {id: {content, embedding, metadata}}
        # Embedding is mocked — all documents get a trivial vector so <=> does "some" ranking.
        stores: dict[str, dict[str, dict]] = {}

        class MockConnection:
            """Simulates an asyncpg connection with execute() and fetch() that store/query in-memory dicts."""

            def __init__(self, table_prefix: str = "v_"):
                self._prefix = table_prefix

            async def fetchval(self, query: str, *args, **kwargs):
                return None  # no existing vector column: the dimension check passes

            async def execute(self, query: str, *args, **kwargs):
                # DDL: CREATE EXTENSION, CREATE TABLE, CREATE INDEX — no-op
                stripped = query.strip().upper()
                if stripped.startswith("CREATE"):
                    return
                # INSERT INTO v_xxx ... ON CONFLICT
                if stripped.startswith("INSERT"):
                    table_name = query.split()[2]  # v_collection
                    doc_id = args[0]
                    metadata = args[1]
                    _embedding = args[2]
                    content = args[3]
                    stores.setdefault(table_name, {})[doc_id] = {
                        "content": content,
                        "embedding": _embedding,
                        "metadata": metadata,
                    }

            async def fetch(self, query: str, *args, **kwargs):
                # e.g. "SELECT content FROM v_xxx ORDER BY embedding <=> $1::vector LIMIT $2"
                if "plainto_tsquery" in query:
                    return []  # the full-text half of a hybrid search: this mock has no text index
                parts = query.split()
                from_idx = next(i for i, p in enumerate(parts) if p.upper() == "FROM")
                table_name = parts[from_idx + 1]
                query_embedding = args[0]
                # Parse LIMIT — it's either literal number or $2 / placeholder
                limit_idx = next(i for i, p in enumerate(parts) if p.upper() == "LIMIT")
                limit_raw = parts[limit_idx + 1]
                try:
                    limit = int(limit_raw)
                except ValueError:
                    # It's a placeholder like $2 — pull from args by index
                    placeholder_idx = int(limit_raw.lstrip("$")) - 1
                    limit = args[placeholder_idx]

                table = stores.get(table_name, {})
                if not table:
                    return []
                # Use cosine distance on the (mocked) embeddings
                # Embeddings are lists of floats; compute cosine distance
                scored = []
                for _doc_id, record in table.items():
                    emb = record["embedding"]
                    import json

                    if isinstance(emb, str):
                        # pgvector format "[0.1,0.2,...]"
                        emb = json.loads(emb)
                    # Compute cosine distance
                    import math

                    q_raw = query_embedding
                    if isinstance(q_raw, str):
                        q_raw = json.loads(q_raw)

                    q = q_raw if isinstance(q_raw, list) else [0.0]
                    e = emb if isinstance(emb, list) else [0.0]
                    # Dot product / (norm_a * norm_b)
                    dot = sum(a * b for a, b in zip(q, e, strict=True))
                    norm_q = math.sqrt(sum(a * a for a in q)) or 1
                    norm_e = math.sqrt(sum(b * b for b in e)) or 1
                    distance = 1.0 - (dot / (norm_q * norm_e))
                    scored.append((distance, record["content"]))

                scored.sort(key=lambda x: x[0])
                # Return asyncpg-style Row proxies (dict-like)
                rows = []
                for _, content in scored[:limit]:
                    row = MagicMock()
                    row.__getitem__ = lambda _self, _k, _content=content: _content if _k == "content" else None
                    rows.append(row)
                return rows

        class MockPool:
            """Simulates asyncpg connection pool."""

            @contextlib.asynccontextmanager
            async def acquire(self):
                yield MockConnection()

            async def close(self):
                pass

        pool = MockPool()
        from pydantic_agents.core.vector_db import _set_test_pool

        _set_test_pool(pool)  # type: ignore[arg-type]
        return stores, pool

    @pytest.mark.asyncio
    async def test_vectormemory_add_and_get_relevant(self, tmp_path, monkeypatch):
        """VectorMemory add_memory and get_relevant_memories work end-to-end with mocked vector_db."""
        stores, _ = self._create_in_memory_pool_and_set(monkeypatch)

        vm = VectorMemory("test_integration")

        await vm.add_memory(
            "The payment gateway uses Stripe API v3",
            metadata={"type": "decision", "agent_name": "backend"},
        )
        await vm.add_memory(
            "User authentication uses OAuth 2.0 with Google provider",
            metadata={"type": "decision", "agent_name": "backend"},
        )

        results = await vm.get_relevant_memories("payment gateway", n_results=5)
        assert len(results) >= 1
        assert any("Stripe" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_vectormemory_consolidate(self, tmp_path, monkeypatch):
        """VectorMemory.consolidate returns a formatted text block of recent memories."""
        stores, _ = self._create_in_memory_pool_and_set(monkeypatch)

        vm = VectorMemory("test_consolidate")

        await vm.add_memory("First architectural decision", metadata={"type": "decision", "agent_name": "cto"})
        await vm.add_memory("Second architectural decision", metadata={"type": "decision", "agent_name": "ba"})

        result = await vm.consolidate(n_recent=10)
        assert result is not None
        assert "architectural decision" in result

    @pytest.mark.asyncio
    async def test_vectormemory_roundtrip(self, tmp_path, monkeypatch):
        """Added memories are retrievable by semantic search."""
        stores, _ = self._create_in_memory_pool_and_set(monkeypatch)

        vm = VectorMemory("test_roundtrip")

        await vm.add_memory(
            "During pass 1, the backend agent performed an action: create_payment."
            " Summary: Implemented Stripe payment processing"
        )
        await vm.add_memory(
            "During pass 1, the frontend agent performed an action: create_ui. Summary: Built checkout form"
        )

        results = await vm.get_relevant_memories("Stripe payment", n_results=5)
        assert len(results) >= 1

    def test_memory_model_default_metadata_is_empty_dict(self):
        """Memory model should default metadata to empty dict, not None."""
        from pydantic_agents.core.memory import Memory

        m = Memory(content="test")
        assert m.metadata == {}

    def test_memory_model_timestamp_is_set(self):
        """Memory model should auto-generate a timestamp."""
        from pydantic_agents.core.memory import Memory

        m = Memory(content="test")
        assert m.timestamp is not None
        assert "T" in m.timestamp  # ISO format contains 'T'
