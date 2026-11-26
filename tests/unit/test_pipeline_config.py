import pytest

from pydantic_agents.core.pipeline.config import BLUEPRINTS, BlueprintSpec, PipelineConfig


class TestPipelineConfig:
    def test_default_blueprint_is_legacy(self):
        cfg = PipelineConfig()
        assert cfg.blueprint == "new_feature"
        assert cfg.framework == "legacy"

    def test_known_blueprints_are_valid(self):
        for _name, bp in BLUEPRINTS.items():
            assert "description" in bp
            assert "framework" in bp
            assert bp["framework"] in ("legacy", "langgraph")
            assert bp.get("stages")

    def test_langgraph_blueprint_selects_correct_framework(self):
        cfg = PipelineConfig(blueprint="langgraph_full")
        assert cfg.blueprint == "langgraph_full"
        assert cfg.framework == "langgraph"

    def test_retired_frameworks_and_blueprints_fail_closed(self):
        for retired_bp in (
            "crew_sdlc",
            "crew_qa",
            "crew_architecture_review",
            "crew_audit",
            "autogen_sdlc",
            "autogen_design_review",
            "composite",
        ):
            with pytest.raises(ValueError, match="Unknown blueprint"):
                PipelineConfig(blueprint=retired_bp)

        for retired_fw in ("crewai", "autogen", "composite"):
            with pytest.raises(ValueError, match="Unsupported pipeline framework"):
                PipelineConfig(blueprint="new_feature", framework=retired_fw)
            with pytest.raises(ValueError, match="retired/unsupported framework"):
                BlueprintSpec(name="bad", description="d", framework=retired_fw, stages=["ba"])

    def test_unknown_blueprint_raises(self):
        with pytest.raises(ValueError, match="Unknown blueprint"):
            PipelineConfig(blueprint="nonexistent")

    def test_output_dir_is_project_isolated(self):
        cfg = PipelineConfig(project="myproj")
        assert "myproj" in str(cfg.output_dir())

    def test_requirement_propagates_to_state(self):
        cfg = PipelineConfig(requirement="Build a payment service")
        assert cfg.requirement == "Build a payment service"


class TestFrameworkValidation:
    def test_mismatch_warns(self, caplog):
        with caplog.at_level("WARNING", logger="pydantic_agents.core.pipeline.config"):
            cfg = PipelineConfig(blueprint="new_feature", framework="langgraph")
        assert cfg.framework == "langgraph"
        assert "expects framework 'legacy' but 'langgraph' was requested" in caplog.text

    def test_blueprint_spec_requires_stages(self):
        with pytest.raises(ValueError, match="requires 'stages'"):
            BlueprintSpec(name="empty", description="d", framework="legacy", stages=[])
