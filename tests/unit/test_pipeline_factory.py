import pytest

from pydantic_agents.core.pipeline.config import PipelineConfig


class TestPipelineFactory:
    def test_build_default_pipeline(self):
        from pydantic_agents.core.pipeline.factory import build_default_pipeline

        cfg = PipelineConfig()
        pipeline = build_default_pipeline(cfg)
        assert pipeline is not None
        assert hasattr(pipeline, "execute")

    def test_build_legacy(self):
        from pydantic_agents.core.pipeline.factory import build_pipeline

        cfg = PipelineConfig(framework="legacy")
        pipeline = build_pipeline(cfg)
        assert pipeline is not None
        assert hasattr(pipeline, "execute")

    def test_build_langgraph(self):
        from pydantic_agents.core.pipeline.factory import build_pipeline
        from pydantic_agents.core.pipeline.protocols import PipelineOrchestrator

        cfg = PipelineConfig(blueprint="langgraph_full", framework="langgraph")
        pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, PipelineOrchestrator)

    @pytest.mark.parametrize("retired", ["crewai", "autogen", "composite"])
    def test_build_pipeline_rejects_retired_frameworks(self, retired):
        from pydantic_agents.core.pipeline.factory import build_pipeline

        cfg = PipelineConfig()
        cfg.framework = retired
        with pytest.raises(ValueError, match="Unsupported pipeline framework"):
            build_pipeline(cfg)

    def test_protocol_runtime_checkable(self):
        from pydantic_agents.core.pipeline.protocols import PipelineOrchestrator

        class Fake:
            async def execute(self):
                pass

        assert isinstance(Fake(), PipelineOrchestrator)
