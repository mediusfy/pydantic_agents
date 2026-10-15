import logging
import os
import threading
import time

from pydantic import ValidationError

from pydantic_agents.core.adapters.state_repository import StateRepositoryPort
from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    run_id     TEXT NOT NULL,
    stage      TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (run_id, stage)
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON pipeline_checkpoints(updated_at)"


class TursoStateRepository(StateRepositoryPort):
    """
    Embedded Turso (SQLite-compatible) implementation of the StateRepositoryPort.

    One database file per project, holding every run's checkpoints. Requires the
    optional ``turso`` extra (``pyturso``).
    """

    def __init__(self, project_name: str, db_path: str | None = None):
        try:
            import turso
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "state_backend='turso' requires the 'turso' extra: pip install pydantic-agents[turso]"
            ) from e

        self.project_name = project_name
        if db_path is None:
            directory = os.path.join(str(settings.checkpoint_dir), project_name)
            os.makedirs(directory, exist_ok=True)
            db_path = os.path.join(directory, "state.db")
        self.db_path = db_path

        # A single connection shared across callers; the lock serialises access.
        self._lock = threading.Lock()
        self._conn = turso.connect(db_path)
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.execute(_INDEX)
            self._conn.commit()

    def save(self, run_id: str, stage: str, state: PipelineState) -> None:
        data = state.model_dump_json()
        with self._lock:
            self._conn.execute(
                "INSERT INTO pipeline_checkpoints (run_id, stage, state_json, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id, stage) DO UPDATE SET "
                "state_json = excluded.state_json, updated_at = excluded.updated_at",
                (run_id, stage, data, time.time()),
            )
            self._conn.commit()
        logger.info(f"State saved to turso: stage='{stage}' run='{run_id}' db={self.db_path}")

    def load(self, run_id: str, stage: str) -> PipelineState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM pipeline_checkpoints WHERE run_id = ? AND stage = ?",
                (run_id, stage),
            ).fetchone()
        if row is None:
            return None
        try:
            return PipelineState.model_validate_json(row[0])
        except (ValidationError, ValueError) as e:
            logger.warning(f"Corrupted checkpoint for stage '{stage}': {e}")
            return None

    def exists(self, run_id: str, stage: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM pipeline_checkpoints WHERE run_id = ? AND stage = ?",
                (run_id, stage),
            ).fetchone()
        return row is not None

    def clear(self, run_id: str) -> None:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM pipeline_checkpoints WHERE run_id = ?", (run_id,))
            count = cursor.rowcount
            self._conn.commit()
        logger.info(f"State cleared: {count} rows for run {run_id}")

    def find_latest_run_id(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id FROM pipeline_checkpoints ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
