import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_agents.core.limits import llm_semaphore
from pydantic_agents.core.models.core import Artifact, FailureReport, PipelineState, UserStory
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.registry import registry
from pydantic_agents.core.session_recorder import recorder
from pydantic_agents.core.settings import settings
from pydantic_agents.core.stages.base import StageRunner

logger = logging.getLogger(__name__)

# ── Emoji markers for progress telemetry ───────────────────────────────────
_PROGRESS = {
    "pending": "⏳",
    "running": "▶",
    "done": "✓",
    "failed": "✗",
    "skipped": "–",
}


class ProgressTracker:
    """Lightweight progress tracker that prints actionable status lines."""

    def __init__(self, total_agents: int, total_stories: int):
        self.total_agents = total_agents
        self.total_stories = total_stories
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.agent_map: dict[str, str] = {}  # label → status

    def start(self, label: str):
        self.agent_map[label] = "running"
        self._print()

    def done(self, label: str):
        self.agent_map[label] = "done"
        self.completed += 1
        self._print()

    def fail(self, label: str):
        self.agent_map[label] = "failed"
        self.failed += 1
        self._print()

    def _print(self):
        elapsed = int(time.time() - self.start_time)
        statuses = "  ".join(f"{s} {k}" for k, s in sorted(self.agent_map.items()))
        logger.info(f"[{elapsed}s] {self.completed}/{self.total_agents} agents done ({self.failed} failed)  {statuses}")


def _mock_artifacts() -> list[Artifact]:
    """The fixed artifacts a dry run produces."""
    return [
        Artifact(
            agent_name="backend",
            files={"cmd/server/main.go": "package main\nfunc main() {}\n"},
            metadata={"user_story_id": "US-001", "language": "go"},
        ),
        Artifact(
            agent_name="frontend",
            files={"src/App.vue": "<template><div>Hello</div></template>\n"},
            metadata={"user_story_id": "US-001", "language": "vue"},
        ),
        Artifact(
            agent_name="test",
            files={"tests/payment.spec.ts": "test('payment', () => {});\n"},
            metadata={"user_story_id": "US-001"},
        ),
    ]


@dataclass
class _Run:
    """What one implementation run shares between its helpers."""

    state: PipelineState
    config: PipelineConfig
    semaphore: asyncio.Semaphore
    is_retry: bool
    repo_context: str
    markdown_ctx: str
    cto_advice: str
    ux_advice: str
    project_state: Any
    progress: ProgressTracker
    # Story id -> what its failed test run printed, so a retry sees the failure and not just the story id.
    test_output: dict[str, str] = field(default_factory=dict)


async def _guidelines(agent: str, gherkin: str, heading: str, project: str | None = None) -> str:
    """Retrieved guidelines for an agent as a prompt section, or "" when there are none."""
    from pydantic_agents.core.context_engine import retrieve_context

    retrieved = await retrieve_context(agent, gherkin, project=project)
    return f"\n\nRetrieved {heading} Guidelines:\n{retrieved}" if retrieved else ""


def _retry_section(run: _Run) -> str:
    """The previous failure, for an agent that is asked to patch rather than rewrite; "" on a first run."""
    failure = run.state.failure_report
    if not (run.is_retry and failure is not None):
        return ""
    return (
        f"PREVIOUS FAILURE DETECTED:\n"
        f"Source: {failure.source}\n"
        f"Error Type: {failure.error_type}\n"
        f"Trace/Feedback:\n{failure.trace}\n\n"
        "TASK: Analyze the failure telemetry and generate a hyper-specific patch to fix "
        "the issue. Only output corrected files instead of rewriting from scratch."
    )


# Prompt layout: the blocks that are identical for every story of a run (repository context, local guidelines,
# CTO advice) come first and in the same order, so a provider-side prefix cache can reuse them across the
# per-story calls; the story and the retrieved snippets vary at the end.
def _backend_prompt(run: _Run, us: UserStory, guidelines: str) -> str:
    retry = _retry_section(run)
    if retry:
        return (
            f"{run.repo_context}\n\n{run.markdown_ctx}\n\n"
            f"CTO ADVICE:\n{run.cto_advice}\n\n"
            f"FULL GHERKIN:\n{us.gherkin}\n\n"
            f"{retry}{guidelines}"
        )
    return (
        f"{run.repo_context}\n\n{run.markdown_ctx}\n\n"
        f"CTO ADVICE:\n{run.cto_advice}\n\n"
        f"FULL GHERKIN:\n{us.gherkin}\n\n"
        f"TASK: Generate a complete Go backend for this user story using Hexagonal Architecture. "
        f"Include ALL of the following in one pass:\n"
        f"1. Domain structs and interfaces (domain/types.go)\n"
        f"2. Postgres database adapters (adapters/postgres_repo.go)\n"
        f"3. Pub/Sub event handlers and definitions (ports/events.go)\n"
        f"4. Core business logic service wiring (domain/service.go)\n"
        f"5. Entry point with HTTP/gRPC server ({settings.go_entrypoint})\n\n"
        f"All files must compile together. "
        f"Run `run_go_build` to self-validate before returning.{guidelines}"
    )


async def _frontend_prompt(run: _Run, us: UserStory) -> str:
    guidelines = await _guidelines("frontend", us.gherkin, "Frontend", run.config.project)
    retry = _retry_section(run)
    task = retry or f"TASK:\nImplement frontend for story {us.id}: {us.title}"
    return (
        f"{run.repo_context}\n\n{run.markdown_ctx}\n\n"
        f"CTO ADVICE:\n{run.cto_advice}\n\n"
        f"UX GUIDELINES:\n{run.ux_advice}\n\n"
        f"FULL GHERKIN:\n{us.gherkin}\n\n"
        f"{task}{guidelines}"
    )


async def _database_prompt(run: _Run, us: UserStory) -> str:
    guidelines = await _guidelines("database", us.gherkin, "Database", run.config.project)
    retry = _retry_section(run)
    task = retry or f"TASK:\nGenerate database schema/migrations for story {us.id}: {us.title}"
    return (
        f"{run.markdown_ctx}\n\n"
        f"CTO ADVICE:\n{run.cto_advice}\n\n"
        f"FULL GHERKIN:\n{us.gherkin}\n\n"
        "EVENTS:\n"
        + "\n".join(f" - {e.name}: {e.description}" for e in (us.event_triggers or []))
        + f"\n\n{task}{guidelines}"
    )


def _collect_artifacts(
    artifacts: list[Artifact], tasks: list[tuple[str, Any]], results: list[Any], is_retry: bool
) -> list[Artifact]:
    """Merge the artifacts the generation tasks returned; a task that raised is logged and skipped."""
    for (agent_type, _), r in zip(tasks, results, strict=False):
        if isinstance(r, Exception):
            logger.error(f"Implementation task ({agent_type}) raised: {r}")
            continue
        # A task yields one artifact, a list of them, or None (e.g. a skipped retry task), which is ignored.
        for item in r if isinstance(r, list) else [r]:
            if isinstance(item, Artifact):
                artifacts = _merge_artifact(artifacts, item, is_retry)
    return artifacts


def _story_sources(artifacts: list[Artifact], story_id: str) -> dict[str, str]:
    """Every non-empty source file the story's implementation artifacts produced, to run its tests against."""
    sources: dict[str, str] = {}
    for a in artifacts:
        if a.agent_name in ("backend", "frontend") and a.metadata.get("user_story_id") == story_id:
            sources.update({n: c for n, c in (a.files or {}).items() if c})
    return sources


def _replace_test_artifact(artifacts: list[Artifact], new_test: Artifact) -> list[Artifact]:
    """Add a story's test artifact, dropping the earlier one for the same story."""
    story_id = new_test.metadata["user_story_id"]
    kept = [a for a in artifacts if not (a.agent_name == "test_eng" and a.metadata.get("user_story_id") == story_id)]
    kept.append(new_test)
    return kept


_TEST_TRACE_CHARS = 3_000


def _test_trace(failed_stories: list[str], details: dict[str, str]) -> str:
    """The failure trace the next attempt reads: each failed story with the tail of its test output (the end of
    a ``go test`` run is where the failing assertions are)."""
    lines = [f"Failed stories: {', '.join(failed_stories)}"]
    for story_id in failed_stories:
        if output := details.get(story_id):
            lines.append(f"\n--- {story_id} ---\n{output[-_TEST_TRACE_CHARS:]}")
    return "\n".join(lines)


def _record_test_outcome(
    state: PipelineState, failed_stories: list[str], n_stories: int, details: dict[str, str] | None = None
) -> None:
    """Set the feedback and failure report the next implementation attempt (or the loop) will read."""
    if failed_stories:
        logger.warning(f"Test failures in {len(failed_stories)}/{n_stories} stories: {', '.join(failed_stories)}")
        state.feedback = (
            f"Test suite failed for stories: {', '.join(failed_stories)}. "
            "Rewrite the implementations to resolve these errors."
        )
        state.failure_report = FailureReport(
            source="test_suite",
            error_type="test_failure",
            trace=_test_trace(failed_stories, details or {}),
            relevant_files=failed_stories,
        )
    else:
        logger.info("All tests passed.")
        state.feedback = None
        state.failure_report = None
    state.approved = False


class ImplementationStage(StageRunner):
    def _new_run(self, state: PipelineState, config: PipelineConfig) -> _Run:
        """Gather the shared context for a run: repo awareness, advice, retry mode and the progress tracker."""
        from pydantic_agents.agents.backend import ProjectState
        from pydantic_agents.core.adapters.workspace import WorkspaceContext

        semaphore = llm_semaphore()
        user_stories = state.user_stories or []
        workspace = WorkspaceContext(root_dir=config.target_root)
        repo_context = workspace.get_repo_context()
        markdown_ctx = workspace.get_markdown_context()
        is_retry = bool(state.failure_report)

        # Agents to track: backend and test per story; on a first run also frontend and database per story, plus
        # devops and docs.
        n_stories = len(user_stories)
        n_agents = 2 * n_stories + (0 if is_retry else 2 * n_stories + 2)
        progress = ProgressTracker(total_agents=n_agents, total_stories=n_stories)

        return _Run(
            state=state,
            config=config,
            semaphore=semaphore,
            is_retry=is_retry,
            repo_context=repo_context,
            markdown_ctx=markdown_ctx,
            cto_advice=state.advisory_results.get("cto", ""),
            ux_advice=state.advisory_results.get("ux", ""),
            project_state=ProjectState(
                base_path=f"/app/{config.project}", environment="dev", team_owner="platform-engineering"
            ),
            progress=progress,
        )

    async def _run_tagged(
        self, run: _Run, agent: str, prompt: str, story_id: str | None = None, deps: Any = None
    ) -> Artifact:
        label = f"{story_id or 'global'}:{agent}"
        run.progress.start(label)
        async with recorder.record_span(
            run_id=run.state.run_id,
            agent_name=agent,
            pipeline_stage="implementation",
            prompt=prompt,
            story_id=story_id or "",
        ) as rec:
            res = await self.provider.run_agent(run.semaphore, agent, registry.get(agent), prompt, deps=deps)
            rec.set_output(res)
            res.output.metadata["user_story_id"] = story_id or "global"
            run.progress.done(label)
            return res.output

    # Backend: one consolidated prompt per story. Collapsed from 4 sequential phases into 1: the backend agent
    # already self-validates with run_go_build, so splitting doesn't buy quality.
    async def _run_backend(self, run: _Run, us: UserStory, story_idx: int) -> Artifact:
        prompt = _backend_prompt(run, us, await _guidelines("backend", us.gherkin, "Backend", run.config.project))
        res = await self._run_tagged(run, "backend", prompt, story_id=us.id, deps=run.project_state)
        res.metadata["story_index"] = story_idx
        return res

    async def _run_test(self, run: _Run, us: UserStory, story_idx: int) -> Artifact:
        label = f"{us.id}:test_eng"
        run.progress.start(label)
        test_prompt = f"Write tests for gherkin: {us.gherkin}"
        test_prompt += await _guidelines("test_eng", us.gherkin, "Test", run.config.project)
        async with recorder.record_span(
            run_id=run.state.run_id,
            agent_name="test_eng",
            pipeline_stage="implementation",
            prompt=test_prompt,
            story_id=us.id,
            story_gherkin=us.gherkin,
        ) as rec:
            res = await self.provider.run_agent(run.semaphore, "test_eng", registry.get("test_eng"), test_prompt)
            rec.set_output(res)
            res.output.metadata["user_story_id"] = us.id
            res.output.metadata["story_index"] = story_idx
            run.progress.done(label)
            return res.output

    async def _run_frontend(self, run: _Run, us: UserStory) -> Artifact:
        return await self._run_tagged(run, "frontend", await _frontend_prompt(run, us), story_id=us.id)

    async def _run_database(self, run: _Run, us: UserStory) -> Artifact:
        return await self._run_tagged(run, "database", await _database_prompt(run, us), story_id=us.id)

    def _generation_tasks(self, run: _Run) -> list[tuple[str, Any]]:
        """The (agent, coroutine) pairs to run concurrently: per-story work, then the global agents.

        A retry after a test failure re-runs the backend only; a retry after a gate rejection (review, security,
        quality gate, validation) re-runs the frontend and database agents too, since their files were judged.
        """
        failure = run.state.failure_report
        rerun_all = run.is_retry and failure is not None and failure.source != "test_suite"
        tasks: list[tuple[str, Any]] = []
        for idx, us in enumerate(run.state.user_stories or []):
            tasks.append(("backend", self._run_backend(run, us, idx)))
            if not run.is_retry or rerun_all:
                tasks.append(("frontend", self._run_frontend(run, us)))
                tasks.append(("database", self._run_database(run, us)))

        if not run.is_retry:
            stories = run.state.user_stories or []
            devops_prompt = "Generate CI/CD pipelines and Docker Compose for these stories:\n" + "\n".join(
                f"- {us.id}: {us.title}" for us in stories
            )
            docs_prompt = f"Generate API docs and README for: {run.config.requirement or ''}"
            tasks.append(("devops", self._run_tagged(run, "devops", devops_prompt)))
            tasks.append(("docs", self._run_tagged(run, "docs", docs_prompt)))
        return tasks

    @staticmethod
    async def _test_story(
        run: _Run, us: UserStory, res: Any, artifacts: list[Artifact], run_test_suite: Any
    ) -> tuple[list[Artifact], bool]:
        """One story's test outcome: (the artifacts with its tests added, whether the story failed).

        A crashed test agent, missing test files or missing sources fail the story (no tests is not a pass). Only Go
        suites are run: the vitest path has no package.json or node_modules.
        """
        if isinstance(res, BaseException):
            logger.error(f"Test agent for {us.id} failed: {res}")
            run.test_output[us.id] = f"The test agent crashed: {res}"
            return artifacts, True
        res.metadata["user_story_id"] = res.metadata.get("user_story_id", "?")
        artifacts = _replace_test_artifact(artifacts, res)

        sources = _story_sources(artifacts, res.metadata["user_story_id"])
        if not (res.files and sources):
            reason = "the test agent returned no test files." if not res.files else "no source files were generated."
            run.test_output[us.id] = f"No tests ran: {reason}"
            return artifacts, True
        if not any(name.endswith(".go") for name in res.files):
            logger.warning(f"Tests for {us.id} are not Go tests; they are kept but not executed.")
            return artifacts, False
        outcome = await run_test_suite(sources, dict(res.files))
        if outcome.passed:
            return artifacts, False
        run.test_output[us.id] = outcome.error or outcome.output
        logger.warning(f"Test suite failed for {us.id}: {(outcome.error or outcome.output)[:200]}...")
        return artifacts, True

    async def _run_test_phase(self, run: _Run, artifacts: list[Artifact]) -> tuple[list[Artifact], list[str]]:
        """Per-story test agents in parallel (failures isolated), then each story's suite against every source
        file of the story; returns the new artifact list and the ids of the stories whose suite failed or could
        not be run (a toolchain or setup error is a failure, not a pass)."""
        from pydantic_agents.core.tools.test_tools import run_test_suite

        logger.info("Running Test agents per story...")
        user_stories = run.state.user_stories or []
        test_results = await asyncio.gather(
            *(self._run_test(run, us, idx) for idx, us in enumerate(user_stories)), return_exceptions=True
        )

        failed_stories: list[str] = []
        for us, res in zip(user_stories, test_results, strict=False):
            artifacts, failed = await self._test_story(run, us, res, artifacts, run_test_suite)
            if failed:
                failed_stories.append(us.id)
        return artifacts, failed_stories

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        logger.info("--- Running Implementation Stage ---")

        if config.dry_run:
            logger.info("DRY RUN: Generating mock implementation artifacts.")
            state.artifacts = _mock_artifacts()
            state.stage = "implementation"
            return state

        run = self._new_run(state, config)
        artifacts = list(state.artifacts) if run.is_retry and state.artifacts else []

        # ── Run all generation tasks concurrently ──────────────────────────
        tasks = self._generation_tasks(run)
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        artifacts = _collect_artifacts(artifacts, tasks, results, run.is_retry)

        if not artifacts:
            # Every generation task failed: say so instead of testing nothing and reporting "All tests passed".
            state.failure_report = FailureReport(
                source="implementation",
                error_type="runtime_error",
                trace="No implementation agent produced an artifact (see the logs for each agent's error).",
            )
            state.approved = False
            state.stage = "implementation"
            return state

        # ── Test phase ──────────────────────────────────────────────────────
        artifacts, failed_stories = await self._run_test_phase(run, artifacts)
        _record_test_outcome(state, failed_stories, len(state.user_stories or []), run.test_output)

        state.artifacts = artifacts
        state.stage = "implementation"
        return state


def _merge_artifact(artifacts: list[Artifact], new: Artifact, is_retry: bool) -> list[Artifact]:
    """Merge an artifact into the list. On retry, merge files into matching existing artifacts."""
    if not is_retry:
        artifacts.append(new)
        return artifacts

    merged = False
    us_id = new.metadata.get("user_story_id", "")
    for art in artifacts:
        if art.agent_name == new.agent_name and art.metadata.get("user_story_id") == us_id:
            art.files.update(new.files)
            merged = True
            break
    if not merged:
        artifacts.append(new)
    return artifacts
