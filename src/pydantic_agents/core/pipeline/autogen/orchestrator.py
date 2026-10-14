import logging
import uuid

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.autogen.bridge import run_autogen_pipeline
from pydantic_agents.core.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class AutoGenPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = PipelineState(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            stage="init",
            requirement=self.config.requirement or "",
        )

    async def execute(self):
        logger.info("Starting AutoGen Pipeline for blueprint: %s", self.config.blueprint)
        self.state = await run_autogen_pipeline(self.state, self.config)
        logger.info("AutoGen Pipeline complete. Stage: %s", self.state.stage)
        return self.state
