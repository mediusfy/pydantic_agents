import logging
import re
import uuid

from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
from pydantic_agents.core.adapters.state_repository import build_state_repository
from pydantic_agents.core.adapters.workspace import ArtifactStorageAdapter, WorkspaceContext
from pydantic_agents.core.checkpoint import CheckpointManager
from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.models.core import FailureReport, PipelineState
from pydantic_agents.core.pipeline.config import STAGES, PipelineConfig
from pydantic_agents.core.stages.base import StageRunner

logger = logging.getLogger(__name__)


# ── Step routing: single-agent names → their containing pipeline stage ─────
_STEP_TO_STAGE = {
    STAGES.BA: STAGES.BA,
    STAGES.ADVISORY: STAGES.ADVISORY,
    STAGES.IMPLEMENTATION: STAGES.IMPLEMENTATION,
    "backend": STAGES.IMPLEMENTATION,
    "frontend": STAGES.IMPLEMENTATION,
    "database": STAGES.IMPLEMENTATION,
    "test": STAGES.IMPLEMENTATION,
    "devops": STAGES.IMPLEMENTATION,
    "docs": STAGES.IMPLEMENTATION,
    STAGES.QUALITY_GATE: STAGES.QUALITY_GATE,
    "sre": STAGES.QUALITY_GATE,
    "review": STAGES.QUALITY_GATE,
    "security": STAGES.QUALITY_GATE,
    "all": "all",
}


_ALL_STAGES = [STAGES.BA, STAGES.ADVISORY, STAGES.IMPLEMENTATION, STAGES.QUALITY_GATE]

# Checkpoint key of one implementation attempt (the loop saves after every attempt, not only at its end).
_ATTEMPT_STAGE = "implementation.attempt{n}"
_ATTEMPT_RE = re.compile(r"implementation\.attempt(\d+)")


class OrgitrationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize state repository with project isolation
        repository = build_state_repository(config.project)
        # One id for the run and its checkpoints: a resume names the run it continues (config.run_id), or falls
        # back to the project's latest checkpoint when the caller gave none.
        run_id = config.run_id
        if config.resume and run_id is None:
            run_id = repository.find_latest_run_id()
            if run_id is None:
                logger.warning(
                    "Resume requested for project %s but no checkpoint exists; starting a fresh run.", config.project
                )
        run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.checkpoint_mgr = CheckpointManager(repository=repository, run_id=run_id)

        # Workspace context pointing at the target project for repo-awareness
        self.workspace = WorkspaceContext(root_dir=config.target_root)
        self.artifact_writer = ArtifactStorageAdapter(str(config.output_dir()))

        self.agent_provider = PydanticAIAgentProvider()

        req = config.requirement or ""
        self.state = PipelineState(run_id=run_id, stage="init", requirement=req)

        # Stages mapping
        self.stages: dict[str, StageRunner] = {}

    def register_stage(self, name: str, stage: StageRunner):
        if stage.agent_provider is None:
            stage.agent_provider = self.agent_provider
        self.stages[name] = stage

    def _resume_attempt(self) -> int:
        """On resume, load the latest finished implementation attempt and return the attempt to continue from."""
        if not self.config.resume:
            return 0
        latest = -1
        for attempt in range(self.max_attempts()):
            if self.checkpoint_mgr.stage_exists(_ATTEMPT_STAGE.format(n=attempt)):
                latest = attempt
        if latest < 0:
            return 0
        loaded = self.checkpoint_mgr.load(_ATTEMPT_STAGE.format(n=latest))
        if loaded is None:
            return 0
        logger.info("Resuming the implementation loop after attempt %d.", latest + 1)
        self.state = loaded
        return latest + 1

    def max_attempts(self) -> int:
        from pydantic_agents.core.settings import settings

        return settings.max_retries if self.config.step == "all" else 1

    async def _run_implementation_loop(self) -> None:
        """Run implementation, retrying on test failures, with the quality gate after each clean attempt.

        Every attempt (implementation plus its gate) is checkpointed, so a crash in attempt 2 of 3 resumes there.
        """
        max_retries = self.max_attempts()

        for attempt in range(self._resume_attempt(), max_retries):
            logger.info(f"--- Implementation Loop Attempt {attempt + 1}/{max_retries} ---")
            self.state = await self.stages[STAGES.IMPLEMENTATION].run(self.state, self.config)

            # Test suite failed — retry implementation
            failure = getattr(self.state, "failure_report", None)
            if failure and failure.source == "test_suite":
                logger.info("Test failure detected — retrying implementation with feedback.")
                self.checkpoint_mgr.save(_ATTEMPT_STAGE.format(n=attempt), self.state)
                continue

            # Proceed to quality gate
            if STAGES.QUALITY_GATE in self.stages:
                self.state = await self.stages[STAGES.QUALITY_GATE].run(self.state, self.config)
            self.checkpoint_mgr.save(_ATTEMPT_STAGE.format(n=attempt), self.state)

            if getattr(self.state, "approved", False):
                logger.info("Quality gate passed — breaking retry loop.")
                break

    async def _run_stage(self, stage_name: str) -> bool:
        """Run one stage; False when it is handled elsewhere and so needs no checkpoint of its own."""
        if stage_name == STAGES.IMPLEMENTATION:
            await self._run_implementation_loop()
        elif stage_name == STAGES.QUALITY_GATE and self.config.step == "all":
            return False  # Handled inside the implementation retry loop
        else:
            self.state = await self.stages[stage_name].run(self.state, self.config)
        return True

    def _resume_from_checkpoint(self, stage_name: str) -> bool:
        """Load the stage's checkpoint instead of running it, when resuming and one exists."""
        if not (self.config.resume and self.checkpoint_mgr.stage_exists(stage_name)):
            return False
        loaded = self.checkpoint_mgr.load(stage_name)
        if loaded is None:
            logger.warning(f"Checkpoint for stage {stage_name} could not be loaded; running the stage instead.")
            return False
        logger.info(f"Resuming stage {stage_name} from checkpoint...")
        self.state = loaded
        return True

    async def _merge_artifacts(self) -> None:
        """Apply the approved artifacts into the real project tree."""
        from pydantic_agents.core.stages.merge_stage import MergeStage

        merge = MergeStage()
        merge.agent_provider = self.agent_provider
        logger.info("Pipeline approved — running merge stage to apply artifacts into target project.")
        self.state = await merge.run(self.state, self.config)

    async def _run_registered_stage(self, stage_name: str) -> None:
        """Resume the stage from its checkpoint or run it, then checkpoint the result."""
        if self._resume_from_checkpoint(stage_name):
            return

        logger.info(f"Executing stage: {stage_name}")

        try:
            ran = await self._run_stage(stage_name)
        except Exception as e:
            logger.exception("Pipeline crashed during stage '%s': %s", stage_name, e)
            raise

        if ran:
            self.checkpoint_mgr.save(stage_name, self.state)

    def _log_start(self) -> None:
        if run_id_var.get() is None:
            run_id_var.set(self.state.run_id)
        logger.info(f"Starting Orgitration Pipeline for project: {self.config.project}")
        logger.info(f"Target root: {self.config.target_root}")
        logger.info(f"Output dir: {self.config.output_dir()}")
        logger.info(f"Registered stages: {list(self.stages.keys())}")

    def _blocked_after(self, stage_name: str) -> str | None:
        """Why the run cannot continue past a stage, or None: no stories after BA, no CTO approval after advisory."""
        if stage_name == STAGES.BA and not self.state.user_stories:
            return "The BA stage produced no user stories; nothing to design or implement."
        if stage_name == STAGES.ADVISORY and not self.state.cto_approved:
            return "The CTO gate did not approve the run."
        return None

    async def execute(self) -> PipelineState:
        """Run the configured stages and return the final state.

        The state says how the run ended: ``approved`` for a run that passed its gates, ``failure_report`` for one
        that was stopped (no stories, CTO rejection, exhausted gate retries), both unset for a partial run.
        """
        self._log_start()

        # Resolve step name → pipeline stage name
        stage = _STEP_TO_STAGE.get(self.config.step)
        if stage is None:
            logger.error(f"Unknown step: {self.config.step}. Valid steps: {list(_STEP_TO_STAGE.keys())}")
            self.state.failure_report = FailureReport(
                source="pipeline", error_type="config", trace=f"Unknown step: {self.config.step}"
            )
            return self.state

        for stage_name in _ALL_STAGES if stage == "all" else [stage]:
            if stage_name not in self.stages:
                logger.warning(f"Stage {stage_name} not registered. Skipping.")
                continue

            await self._run_registered_stage(stage_name)

            reason = self._blocked_after(stage_name) if stage == "all" else None
            if reason:
                logger.error("Pipeline stopped after stage '%s': %s", stage_name, reason)
                self.state.approved = False
                self.state.failure_report = self.state.failure_report or FailureReport(
                    source=getattr(stage_name, "value", str(stage_name)), error_type="blocked", trace=reason
                )
                return self.state

        # Final artifact flush
        self.artifact_writer.write_artifacts(self.state.artifacts)

        # Run merge stage: apply artifacts into the real project tree
        if stage == "all" and self.state.approved:
            await self._merge_artifacts()
        return self.state
