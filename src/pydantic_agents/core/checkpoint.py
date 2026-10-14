import logging
import uuid

from pydantic_agents.core.adapters.state_repository import StateRepositoryPort
from pydantic_agents.core.models.core import PipelineState

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, repository: StateRepositoryPort, run_id: str | None = None):
        self.repository = repository
        self.run_id = run_id or str(uuid.uuid4())

    def save(self, stage: str, state: PipelineState) -> None:
        self.repository.save(self.run_id, stage, state)

    def load(self, stage: str) -> PipelineState | None:
        return self.repository.load(self.run_id, stage)

    def clear(self) -> None:
        self.repository.clear(self.run_id)

    def find_latest_run_id(self) -> str | None:
        return self.repository.find_latest_run_id()

    def stage_exists(self, stage: str) -> bool:
        return self.repository.exists(self.run_id, stage)
