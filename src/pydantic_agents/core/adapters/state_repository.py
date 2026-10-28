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
        self._migrated = False

    def _ensure_migrated(self) -> None:
        """Once per repository, before the first read: rename legacy checkpoint files to the current name.

        Checkpoints written before the ``__`` separator were "{uuid}_{stage}.json"; ``load`` builds
        "{run_id}__{stage}.json", so without the rename a run started by an older version could be found by
        ``find_latest_run_id`` but never resumed. Legacy run ids were always UUIDs (no underscore), so the first
        underscore is the separator; the stage part may contain underscores and is kept whole. A file that already
        has "__", or whose prefix is not a UUID, is left alone, and such a leftover is reported once. The rename is
        lazy (not in ``__init__``) and never raises: a read-only or odd directory logs a warning and reads proceed.
        """
        if self._migrated:
            return
        self._migrated = True
        try:
            migrated, unrecognised = self._migrate_legacy_files()
        except OSError as e:
            logger.warning("Could not migrate legacy checkpoint files in %s: %s", self.checkpoint_dir, e)
            return
        if migrated:
            logger.info("Migrated %d legacy checkpoint file(s) in %s to the '__' name", migrated, self.checkpoint_dir)
        if unrecognised:
            logger.warning(
                "%d .json file(s) in %s are not checkpoints of this repository (no '__' and no UUID run id) and "
                "are ignored: %s",
                len(unrecognised),
                self.checkpoint_dir,
                ", ".join(sorted(unrecognised)[:5]),
            )

    def _migrate_legacy_files(self) -> tuple[int, list[str]]:
        migrated = 0
        unrecognised: list[str] = []
        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith(".json") or "__" in filename:
                continue
            run_id, sep, rest = filename.partition("_")
            try:
                uuid.UUID(run_id)
            except ValueError:
                unrecognised.append(filename)
                continue
            if not sep:
                unrecognised.append(filename)
                continue
            target = os.path.join(self.checkpoint_dir, f"{run_id}__{rest}")
            if os.path.exists(target):
                continue
            os.replace(os.path.join(self.checkpoint_dir, filename), target)
            migrated += 1
        return migrated, unrecognised

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
        self._ensure_migrated()
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
        self._ensure_migrated()
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
        self._ensure_migrated()
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
