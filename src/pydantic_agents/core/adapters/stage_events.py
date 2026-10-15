import abc
import logging
import os
import threading
import time
from dataclasses import dataclass

from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_stage_events (
    id          INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL,
    stage       TEXT NOT NULL,
    attempt     INTEGER NOT NULL,
    verdict     TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS idx_stage_events_run ON pipeline_stage_events(run_id, id)"


@dataclass(frozen=True)
class StageEvent:
    run_id: str
    stage: str
    attempt: int
    verdict: str
    duration_ms: int
    detail: str = ""


class StageEventRecorderPort(abc.ABC):
    """Port for recording per-stage outcomes (retries, gate verdicts) of a pipeline run.

    Implementations must never raise: recording is best-effort and must not break a run.
    """

    @abc.abstractmethod
    def record(self, run_id: str, stage: str, attempt: int, verdict: str, duration_ms: int, detail: str = "") -> None:
        pass


class NullStageEventRecorder(StageEventRecorderPort):
    def record(self, run_id: str, stage: str, attempt: int, verdict: str, duration_ms: int, detail: str = "") -> None:
        del run_id, stage, attempt, verdict, duration_ms, detail  # intentionally discarded


class TursoStageEventRecorder(StageEventRecorderPort):
    """Appends stage events to the project's embedded Turso database (same file as checkpoints)."""

    def __init__(self, project_name: str, db_path: str | None = None):
        import turso

        if db_path is None:
            directory = os.path.join(str(settings.checkpoint_dir), project_name)
            os.makedirs(directory, exist_ok=True)
            db_path = os.path.join(directory, "state.db")
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = turso.connect(db_path)
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.execute(_INDEX)
            self._conn.commit()

    def record(self, run_id: str, stage: str, attempt: int, verdict: str, duration_ms: int, detail: str = "") -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO pipeline_stage_events "
                    "(run_id, stage, attempt, verdict, duration_ms, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, stage, attempt, verdict, duration_ms, detail, time.time()),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record stage event {stage}/{attempt}: {e}")

    def list_events(self, run_id: str) -> list[StageEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, stage, attempt, verdict, duration_ms, detail "
                "FROM pipeline_stage_events WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [StageEvent(*row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def build_stage_event_recorder(project_name: str) -> StageEventRecorderPort:
    """Return the recorder selected by ``settings.state_backend`` (no-op unless "turso")."""
    if settings.state_backend == "turso":
        try:
            return TursoStageEventRecorder(project_name)
        except Exception as e:
            logger.warning(f"Stage event recording disabled, Turso unavailable: {e}")
    return NullStageEventRecorder()
