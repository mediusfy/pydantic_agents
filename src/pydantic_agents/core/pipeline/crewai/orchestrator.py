import logging
import uuid

from pydantic_agents.core.adapters.workspace import ArtifactStorageAdapter, WorkspaceContext
from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.pipeline.crewai.bridge import run_crewai_pipeline

logger = logging.getLogger(__name__)


class CrewAIPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.workspace = WorkspaceContext(root_dir=config.target_root)
        self.artifact_writer = ArtifactStorageAdapter(str(config.output_dir()))

    async def execute(self):
        state = PipelineState(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            stage="init",
            requirement=self.config.requirement or "",
        )

        logger.info("Starting CrewAI Pipeline for blueprint: %s", self.config.blueprint)
        state = await run_crewai_pipeline(state, self.config)

        self.artifact_writer.write_artifacts(state.artifacts)
        logger.info("CrewAI Pipeline complete. Approved: %s", state.approved)
        return state
