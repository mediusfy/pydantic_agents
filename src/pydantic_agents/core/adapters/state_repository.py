import abc
import json
import logging
import os
import uuid

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)


class StateRepositoryPort(abc.ABC):
    """
    Port for saving and loading PipelineState. This decouples the CheckpointManager
    from local disk I/O, allowing for DB-backed state persistence in the future.
    """

    @abc.abstractmethod
    def save(self, run_id: str, stage: str, state: PipelineState) -> None:
        pass

    @abc.abstractmethod
    def load(self, run_id: str, stage: str) -> PipelineState | None:
        pass

    @abc.abstractmethod
    def exists(self, run_id: str, stage: str) -> bool:
        pass

    @abc.abstractmethod
    def clear(self, run_id: str) -> None:
        pass

    @abc.abstractmethod
    def find_latest_run_id(self) -> str | None:
        pass


class LocalDiskStateRepository(StateRepositoryPort):
    """
    Local filesystem implementation of the StateRepositoryPort.
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.checkpoint_dir = os.path.join(str(settings.checkpoint_dir), project_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_filepath(self, run_id: str, stage: str) -> str:
        safe_stage = stage.replace("/", "_").replace(" ", "_")
        return os.path.join(self.checkpoint_dir, f"{run_id}_{safe_stage}.json")

    def save(self, run_id: str, stage: str, state: PipelineState) -> None:
        filepath = self._get_filepath(run_id, stage)
        try:
            data = state.model_dump_json()
        except AttributeError:
            data = state.json()

        # Atomic write
        tmp_path = f"{filepath}.tmp"
        with open(tmp_path, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
        logger.info(f"State saved to disk: stage='{stage}' at {os.path.relpath(filepath)}")

    def load(self, run_id: str, stage: str) -> PipelineState | None:
        filepath = self._get_filepath(run_id, stage)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath) as f:
                data = json.load(f)
            try:
                return PipelineState.model_validate(data)
            except AttributeError:
                return PipelineState.parse_obj(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Corrupted checkpoint for stage '{stage}': {e}")
            return None

    def exists(self, run_id: str, stage: str) -> bool:
        return os.path.exists(self._get_filepath(run_id, stage))

    def clear(self, run_id: str) -> None:
        if not os.path.isdir(self.checkpoint_dir):
            return
        count = 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(run_id):
                os.remove(os.path.join(self.checkpoint_dir, filename))
                count += 1
        logger.info(f"State cleared: {count} files for run {run_id}")

    def find_latest_run_id(self) -> str | None:
        if not os.path.isdir(self.checkpoint_dir):
            return None
        try:
            files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".json")]
            if not files:
                return None
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
            latest = files[0]
            run_id = latest.split("_")[0]
            try:
                uuid.UUID(run_id)
                return run_id
            except ValueError:
                return None
        except Exception as e:
            logger.warning(f"Error finding latest run: {e}")
            return None
