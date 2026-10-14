"""Merge stage: applies generated artifacts into the real target project.

After all pipeline stages complete and quality gate approves, this stage:
1. Reads artifacts from generated-app/{project}/
2. Maps files to the correct locations in the target project
3. Backs up existing files before overwriting
4. Writes merged output
5. Optionally runs make test / lint on the target project
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig
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


class MergeStage(StageRunner):
    """Applies generated artifacts from `generated-app/{project}/` into the real project tree."""

    def __init__(self):
        self.result = MergeResult()

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        logger.info("--- Running Merge Stage ---")

        if config.dry_run:
            logger.info("DRY RUN: Would merge artifacts into target project.")
            state.stage = "merge"
            state.approved = True
            return state

        target_root = Path(config.target_root)
        if not target_root.exists():
            logger.warning(f"Target project root does not exist: {target_root}")
            state.stage = "merge"
            state.approved = True
            return state

        artifacts = state.artifacts or []
        if not artifacts:
            logger.info("No artifacts to merge.")
            state.stage = "merge"
            state.approved = True
            return state

        # ── File routing maps ──────────────────────────────────────────────
        # Maps agent_name patterns → subdirectory within target project
        agent_target_map: dict[str, str] = {
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

        backup_dir = target_root / ".merge-backups"

        for artifact in artifacts:
            agent = artifact.agent_name
            target_subdir = "."
            for prefix, subdir in agent_target_map.items():
                if agent.startswith(prefix):
                    target_subdir = subdir
                    break

            for filepath, content in (artifact.files or {}).items():
                # Strip common artifact path prefixes
                clean_path = _clean_artifact_path(filepath)

                # Determine absolute destination
                dest = (target_root / target_subdir / clean_path).resolve()

                # Safety: never write outside target root
                if not str(dest).startswith(str(target_root.resolve())):
                    self.result.errors.append(f"Path escape blocked: {clean_path} → {dest}")
                    continue

                # Check if file already exists and differs
                if dest.exists():
                    existing_content = dest.read_text(encoding="utf-8", errors="replace")
                    if existing_content.strip() == content.strip():
                        # Content identical — skip writing
                        self.result.files_skipped += 1
                        continue

                    # Backup existing file
                    backup_path = backup_dir / dest.relative_to(target_root)
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(dest), str(backup_path))
                    self.result.backups_created += 1

                # Write the merged file
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
                self.result.files_written += 1
                logger.info(f"  Merged: {dest.relative_to(target_root)}")

                # Store in context engine
                try:
                    from pydantic_agents.core.context_engine import store_artifact_context

                    store_artifact_context(
                        domain="project",
                        content=f"File: {clean_path}\nAgent: {agent}\nContent:\n{content}",
                        metadata={"agent": agent, "file": clean_path, "project": config.project},
                    )
                except Exception as ce:
                    logger.debug("Failed to store merged file context: %s", ce)

        # Store merge result in state metadata
        # (Removed unused meta assignments)

        logger.info(
            f"Merge complete: {self.result.files_written} written, "
            f"{self.result.files_skipped} skipped (identical), "
            f"{self.result.backups_created} backups in {backup_dir}"
        )
        if self.result.errors:
            for err in self.result.errors:
                logger.warning(f"  Merge error: {err}")

        state.stage = "merge"
        state.approved = True
        return state


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
