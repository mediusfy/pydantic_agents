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
        self._migrate_legacy_files()

    def _migrate_legacy_files(self) -> None:
        """Rename checkpoints written before the ``__`` separator ("{uuid}_{stage}.json") to the current name.

        ``load`` builds "{run_id}__{stage}.json", so without this rename a run started by an older version could be
        found by ``find_latest_run_id`` but never resumed. Legacy run ids were UUIDs (no underscore), so the first
        underscore is the separator. A file that already has "__" or whose prefix is not a UUID is left alone.
        """
        migrated = 0
        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith(".json") or "__" in filename or "_" not in filename:
                continue
            run_id, _, rest = filename.partition("_")
            try:
                uuid.UUID(run_id)
            except ValueError:
                continue
            target = os.path.join(self.checkpoint_dir, f"{run_id}__{rest}")
            if os.path.exists(target):
                continue
            os.replace(os.path.join(self.checkpoint_dir, filename), target)
            migrated += 1
        if migrated:
            logger.info("Migrated %d legacy checkpoint file(s) in %s to the '__' name", migrated, self.checkpoint_dir)

    def _get_filepath(self, run_id: str, stage: str) -> str:
        safe_stage = stage.replace("/", "_").replace(" ", "_")
        return os.path.join(self.checkpoint_dir, f"{run_id}__{safe_stage}.json")

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
            # Only "{run_id}__{stage}.json" files are candidates: that is the name ``load`` builds. Legacy files were
            # renamed by _migrate_legacy_files; anything else without "__" is not a checkpoint of this repository.
            current = [f for f in files if "__" in f]
            if not current:
                return None
            current.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
            return current[0].split("__")[0]
        except Exception as e:
            logger.warning(f"Error finding latest run: {e}")
            return None


def build_state_repository(project_name: str) -> StateRepositoryPort:
    """Return the repository selected by ``settings.state_backend`` (default: disk)."""
    if settings.state_backend == "turso":
        from pydantic_agents.core.adapters.turso_state_repository import TursoStateRepository

        return TursoStateRepository(project_name)
    return LocalDiskStateRepository(project_name)
