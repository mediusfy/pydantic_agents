"""Merge stage: applies generated artifacts into the real target project.

After all pipeline stages complete and quality gate approves, this stage:
1. Reads artifacts from generated-app/{project}/
2. Maps files to the correct locations in the target project
3. Backs up existing files before overwriting
4. Writes merged output
5. Optionally runs make test / lint on the target project
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.settings import settings
from pydantic_agents.core.stages.base import StageRunner

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of a merge operation for user-facing summary."""

    stage: str = "merge"
    files_written: int = 0
    files_skipped: int = 0
    files_conflicted: int = 0
    backups_created: int = 0
    errors: list[str] = field(default_factory=list)


# Maps agent_name prefixes → subdirectory within the target project (first match wins)
_AGENT_TARGET_MAP: dict[str, str] = {
    "frontend": "src",
    "backend": "backend",
    "database": "database",
    "infrastructure": "deploy",
    "devops": "deploy",
    "docs": "docs",
    "sre": "deploy",
    "test": "tests",
    "test_eng": "tests",
}


def _target_subdir(agent: str) -> str:
    return next((subdir for prefix, subdir in _AGENT_TARGET_MAP.items() if agent.startswith(prefix)), ".")


def _approve(state: PipelineState) -> PipelineState:
    state.stage = "merge"
    state.approved = True
    return state


# Longest file body indexed: the embedding model reads about the first 1 KB anyway.
_INDEX_MAX_CHARS = 4_000


async def _store_merged_contexts(pending: list[tuple[str, str, str]], project: str) -> None:
    """Index the merged files in the context engine, in one batch; failures are only logged.

    The row id is the project and file path, so a file merged again replaces its earlier row (a random id added a
    duplicate on every run).
    """
    try:
        from pydantic_agents.core.context_engine import astore_artifact_contexts

        await astore_artifact_contexts(
            "project",
            [
                (
                    f"File: {path}\nAgent: {agent}\nContent:\n{content[:_INDEX_MAX_CHARS]}",
                    {"agent": agent, "file": path, "project": project},
                    f"merge:{project}:{path}",
                )
                for agent, path, content in pending
            ],
        )
    except Exception as ce:
        logger.debug("Failed to store merged file context: %s", ce)


def _is_server_tree(target_root: Path) -> bool:
    """Whether the target is the server's own working directory."""
    return target_root.resolve() == Path.cwd().resolve()


class MergeStage(StageRunner):
    """Applies generated artifacts from `generated-app/{project}/` into the real project tree."""

    def __init__(self):
        self.result = MergeResult()
        self._pending_index: list[tuple[str, str, str]] = []

    def _merge_file(
        self, target_root: Path, backup_dir: Path, subdir: str, filepath: str, content: str, agent: str
    ) -> None:
        """Write one artifact file into the project, backing up a different existing file first."""
        clean_path = _clean_artifact_path(filepath)
        target_root_resolved = target_root.resolve()
        dest = (target_root_resolved / subdir / clean_path).resolve()

        # Safety: never write outside target root
        if not dest.is_relative_to(target_root_resolved):
            self.result.errors.append(f"Path escape blocked: {clean_path} → {dest}")
            return

        if dest.exists():
            if dest.read_text(encoding="utf-8", errors="replace").strip() == content.strip():
                self.result.files_skipped += 1  # content identical — skip writing
                return
            backup_path = backup_dir / dest.relative_to(target_root_resolved)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(dest), str(backup_path))
            self.result.backups_created += 1

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        self.result.files_written += 1
        logger.info(f"  Merged: {dest.relative_to(target_root_resolved)}")
        self._pending_index.append((agent, clean_path, content))

    def _merge_all(self, artifacts: list, target_root: Path, backup_dir: Path) -> None:
        """Write every artifact file (blocking file I/O, so run in a worker thread)."""
        for artifact in artifacts:
            subdir = _target_subdir(artifact.agent_name)
            for filepath, content in (artifact.files or {}).items():
                self._merge_file(target_root, backup_dir, subdir, filepath, content, artifact.agent_name)

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        logger.info("--- Running Merge Stage ---")

        if config.dry_run:
            logger.info("DRY RUN: Would merge artifacts into target project.")
            return _approve(state)

        target_root = Path(config.target_root)
        if not target_root.exists():
            logger.warning(f"Target project root does not exist: {target_root}")
            return _approve(state)

        artifacts = state.artifacts or []
        if not artifacts:
            logger.info("No artifacts to merge.")
            return _approve(state)

        if _is_server_tree(target_root) and not settings.merge_into_server_tree:
            logger.warning(
                "Target root %s is the server's own working directory: not merging into it "
                "(set MERGE_INTO_SERVER_TREE=true to allow). The generated files stay under %s.",
                target_root,
                config.output_dir(),
            )
            return _approve(state)

        backup_dir = target_root / ".merge-backups"
        await asyncio.to_thread(self._merge_all, artifacts, target_root, backup_dir)
        await _store_merged_contexts(self._pending_index, config.project)

        logger.info(
            f"Merge complete: {self.result.files_written} written, "
            f"{self.result.files_skipped} skipped (identical), "
            f"{self.result.backups_created} backups in {backup_dir}"
        )
        for err in self.result.errors:
            logger.warning(f"  Merge error: {err}")

        return _approve(state)


def _clean_artifact_path(filepath: str) -> str:
    """Normalize artifact file paths into project-relative paths.

    Artifact paths can come in many forms from different agents:
    - src/components/Foo.vue        → src/components/Foo.vue  (keep)
    - cmd/server/main.go            → cmd/server/main.go       (keep)
    - frontend/src/App.vue          → src/App.vue             (agent dir is a known prefix)
    - generated-app/default/frontend/src/App.vue  → src/App.vue
    - default/frontend/src/App.vue  → src/App.vue

    Paths that start with an agent name (frontend/, backend/, test/, etc.)
    have the outermost known prefix stripped. Paths that start with a
    real directory like cmd/ or src/ are kept as-is.

    The exception: if the path starts with a known prefix but then has
    a second known prefix, only strip the outermost layer(s) above the
    project-specific subdir.
    """
    parts = Path(filepath).parts

    # Known agent/container prefixes that should be stripped from the path
    known_prefixes = {"generated-app", "default", "frontend", "backend", "infrastructure"}

    # Strip leading known prefixes, but only up to the first "real" directory
    stripped: list[str] = []
    for p in parts:
        if not stripped and p in known_prefixes:
            continue
        stripped.append(p)

    if not stripped:
        return filepath
    return str(Path(*stripped))
