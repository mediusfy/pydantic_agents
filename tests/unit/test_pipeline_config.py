import pytest

from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig


class TestPipelineConfig:
    def test_default_blueprint_is_legacy(self):
        cfg = PipelineConfig()
        assert cfg.blueprint == "new_feature"
        assert cfg.framework == "legacy"

    def test_known_blueprints_are_valid(self):
        for _name, bp in BLUEPRINTS.items():
            assert "description" in bp
            assert "framework" in bp
            assert bp["framework"] in ("legacy", "langgraph", "crewai", "autogen", "composite")

    def test_langgraph_blueprint_selects_correct_framework(self):
        cfg = PipelineConfig(blueprint="langgraph_full", framework="langgraph")
        assert cfg.blueprint == "langgraph_full"

    def test_crewai_blueprint_has_crew_class(self):
        bp = BLUEPRINTS["crew_sdlc"]
        assert "crew_class" in bp
        assert bp["crew_class"] == "SdlcCrew"

    def test_autogen_blueprint_has_group_class(self):
        bp = BLUEPRINTS["autogen_sdlc"]
        assert "group_class" in bp
        assert bp["group_class"] == "SdlcGroupChat"

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
    def test_mismatch_warns(self):
        cfg = PipelineConfig(blueprint="langgraph_full", framework="crewai")
        assert cfg.framework == "crewai"

    def test_all_blueprints_have_valid_stages_or_class(self):
        for name, bp in BLUEPRINTS.items():
            if bp["framework"] in ("legacy", "langgraph", "composite"):
                assert "stages" in bp, f"{name} missing stages"
            elif bp["framework"] == "crewai":
                assert "crew_class" in bp, f"{name} missing crew_class"
            elif bp["framework"] == "autogen":
                assert "group_class" in bp, f"{name} missing group_class"
