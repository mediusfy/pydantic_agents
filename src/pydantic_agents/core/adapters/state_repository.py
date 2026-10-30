import abc
import json
import logging
import os
import threading
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
        self._migration_warned = False
        self._migration_lock = threading.Lock()

    def _ensure_migrated(self) -> None:
        """Before a read: rename legacy "{uuid}_{stage}.json" checkpoints to the "{run_id}__{stage}.json" name
        ``load`` builds, so runs started by an older version resume. Runs until one pass succeeds (a failed pass is
        retried on the next read, warned about once per instance), never raises, and is serialised per instance;
        two instances on one directory both pass because a rename the other did first counts as done.
        """
        if self._migrated:
            return
        with self._migration_lock:
            if self._migrated:
                return
            try:
                migrated, unrecognised = self._migrate_legacy_files()
            except OSError as e:
                if not self._migration_warned:
                    self._migration_warned = True
                    logger.warning(
                        "Could not migrate legacy checkpoint files in %s (will retry on the next read): %s",
                        self.checkpoint_dir,
                        e,
                    )
                return
            self._migrated = True
        if migrated:
            logger.info("Migrated %d legacy checkpoint file(s) in %s to the '__' name", migrated, self.checkpoint_dir)
        if unrecognised:
            names = sorted(unrecognised)
            logger.warning(
                "%d .json file(s) in %s are not checkpoints (no '__' and no UUID run id) and are ignored: %s%s",
                len(names),
                self.checkpoint_dir,
                ", ".join(names[:3]),
                f" (+{len(names) - 3} more at DEBUG)" if len(names) > 3 else "",
            )
            logger.debug("Ignored files in %s: %s", self.checkpoint_dir, ", ".join(names))

    def _migrate_legacy_files(self) -> tuple[int, list[str]]:
        """Legacy run ids were always UUIDs (no underscore), so the first underscore is the separator and the stage
        part, which may contain underscores, is kept whole. Returns (renamed count, files that are neither form)."""
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
            try:
                os.replace(os.path.join(self.checkpoint_dir, filename), target)
            except FileNotFoundError:
                # Another instance on the same directory renamed it between our listdir and replace.
                if not os.path.exists(target):
                    raise
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
        except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError and pydantic's ValidationError are ValueErrors
            logger.warning(f"Corrupted checkpoint for stage '{stage}': {e}")
            return None

    def exists(self, run_id: str, stage: str) -> bool:
        self._ensure_migrated()
        return os.path.exists(self._get_filepath(run_id, stage))

    def clear(self, run_id: str) -> None:
        if not os.path.isdir(self.checkpoint_dir):
            return
        self._ensure_migrated()
        prefix = f"{run_id}__"
        count = 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(prefix):
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
