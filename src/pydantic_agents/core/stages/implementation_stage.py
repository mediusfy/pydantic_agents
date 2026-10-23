import asyncio
import logging
import time

from pydantic_agents.core.limits import llm_semaphore
from pydantic_agents.core.models.core import Artifact, FailureReport, PipelineState
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


class ImplementationStage(StageRunner):
    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        logger.info("--- Running Implementation Stage ---")

        if config.dry_run:
            logger.info("DRY RUN: Generating mock implementation artifacts.")
            state.artifacts = [
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
            state.stage = "implementation"
            return state

        semaphore = llm_semaphore()
        user_stories = state.user_stories or []
        cto_advice = state.advisory_results.get("cto", "")
        ux_advice = state.advisory_results.get("ux", "")
        raw_requirements = config.requirement or ""

        # ── Real project repo context ─────────────────────────────────────
        from pydantic_agents.core.adapters.workspace import WorkspaceContext

        workspace = WorkspaceContext(root_dir=config.target_root)
        repo_context = workspace.get_repo_context()
        markdown_ctx = workspace.get_markdown_context()

        is_retry = bool(state.failure_report)
        artifacts = list(state.artifacts) if is_retry and state.artifacts else []

        from pydantic_agents.agents.backend import ProjectState

        project_state = ProjectState(
            base_path=f"/app/{config.project}", environment="dev", team_owner="platform-engineering"
        )

        # Determine total agent count for progress tracker
        n_stories = len(user_stories)
        n_backend = n_stories  # 1 per story (collapsed from 4)
        n_frontend = n_stories if not is_retry else 0
        n_db = n_stories if not is_retry else 0
        n_test = n_stories
        n_global = 0 if is_retry else 2  # devops + docs
        progress = ProgressTracker(
            total_agents=n_backend + n_frontend + n_db + n_test + n_global,
            total_stories=n_stories,
        )

        # ── Instrumented helpers ───────────────────────────────────────────

        async def _instrumented_test(sem, us, story_idx):
            label = f"{us.id}:test_eng"
            progress.start(label)
            from pydantic_agents.core.context_engine import retrieve_context

            test_ctx = retrieve_context("test_eng", us.gherkin)
            test_prompt = f"Write tests for gherkin: {us.gherkin}"
            if test_ctx:
                test_prompt += f"\n\nRetrieved Test Guidelines:\n{test_ctx}"
            async with recorder.record_span(
                run_id=state.run_id,
                agent_name="test_eng",
                pipeline_stage="implementation",
                prompt=test_prompt,
                story_id=us.id,
                story_gherkin=us.gherkin,
            ) as rec:
                res = await self.provider.run_agent(sem, "test_eng", registry.get("test_eng"), test_prompt)
                rec.set_output(res)
                res.output.metadata["user_story_id"] = us.id
                res.output.metadata["story_index"] = story_idx
                progress.done(label)
                return res.output

        async def run_tagged(sem, agent, prompt, story_id=None, deps=None):
            label = f"{story_id or 'global'}:{agent}"
            progress.start(label)
            async with recorder.record_span(
                run_id=state.run_id,
                agent_name=agent,
                pipeline_stage="implementation",
                prompt=prompt,
                story_id=story_id or "",
            ) as rec:
                res = await self.provider.run_agent(sem, agent, registry.get(agent), prompt, deps=deps)
                rec.set_output(res)
                res.output.metadata["user_story_id"] = story_id or "global"
                progress.done(label)
                return res.output

        # ── Backend: single consolidated prompt per story ──────────────────
        # Collapsed from 4 sequential phases into 1. The backend agent already
        # self-validates with run_go_build, so splitting doesn't buy quality.
        async def run_backend(sem, us, repo_ctx, cto_adv, proj_state, story_idx):
            from pydantic_agents.core.context_engine import retrieve_context

            backend_ctx = retrieve_context("backend", us.gherkin)
            backend_ctx_str = f"\n\nRetrieved Backend Guidelines:\n{backend_ctx}" if backend_ctx else ""

            if is_retry:
                prompt = (
                    f"{repo_ctx}\n\n"
                    f"CTO ADVICE:\n{cto_adv}\n\n"
                    f"FULL GHERKIN:\n{us.gherkin}\n\n"
                    f"PREVIOUS FAILURE DETECTED:\n"
                    f"Source: {state.failure_report.source}\n"
                    f"Error Type: {state.failure_report.error_type}\n"
                    f"Trace/Feedback:\n{state.failure_report.trace}\n\n"
                    f"TASK: Analyze the failure telemetry and generate a hyper-specific patch to fix "
                    f"the issue. Only output corrected files instead of rewriting from scratch.{backend_ctx_str}"
                )
            else:
                prompt = (
                    f"{repo_ctx}\n\n"
                    f"CTO ADVICE:\n{cto_adv}\n\n"
                    f"FULL GHERKIN:\n{us.gherkin}\n\n"
                    f"TASK: Generate a complete Go backend for this user story using Hexagonal Architecture. "
                    f"Include ALL of the following in one pass:\n"
                    f"1. Domain structs and interfaces (domain/types.go)\n"
                    f"2. Postgres database adapters (adapters/postgres_repo.go)\n"
                    f"3. Pub/Sub event handlers and definitions (ports/events.go)\n"
                    f"4. Core business logic service wiring (domain/service.go)\n"
                    f"5. Entry point with HTTP/gRPC server ({settings.go_entrypoint})\n\n"
                    f"All files must compile together. "
                    f"Run `run_go_build` to self-validate before returning.{backend_ctx_str}"
                )
            res = await run_tagged(sem, "backend", prompt, story_id=us.id, deps=proj_state)
            res.metadata["story_index"] = story_idx
            return res

        # ── Build task list ────────────────────────────────────────────────
        per_story_tasks = []
        for idx, us in enumerate(user_stories):
            per_story_tasks.append(
                ("backend", run_backend(semaphore, us, repo_context, cto_advice, project_state, idx))
            )

            if not is_retry:
                from pydantic_agents.core.context_engine import retrieve_context

                fe_ctx = retrieve_context("frontend", us.gherkin)
                fe_ctx_str = f"\n\nRetrieved Frontend Guidelines:\n{fe_ctx}" if fe_ctx else ""

                fe_prompt = (
                    f"{repo_context}\n\n{markdown_ctx}\n\n"
                    f"CTO ADVICE:\n{cto_advice}\n\n"
                    f"UX GUIDELINES:\n{ux_advice}\n\n"
                    f"FULL GHERKIN:\n{us.gherkin}\n\n"
                    f"TASK:\nImplement frontend for story {us.id}: {us.title}{fe_ctx_str}"
                )
                per_story_tasks.append(("frontend", run_tagged(semaphore, "frontend", fe_prompt, story_id=us.id)))

                db_ctx = retrieve_context("database", us.gherkin)
                db_ctx_str = f"\n\nRetrieved Database Guidelines:\n{db_ctx}" if db_ctx else ""

                db_prompt = (
                    f"FULL GHERKIN:\n{us.gherkin}\n\n"
                    "EVENTS:\n"
                    + "\n".join(f" - {e.name}: {e.description}" for e in (us.event_triggers or []))
                    + f"\n\nTASK:\nGenerate database schema/migrations for story {us.id}: {us.title}{db_ctx_str}"
                )
                per_story_tasks.append(("database", run_tagged(semaphore, "database", db_prompt, story_id=us.id)))

        global_tasks = []
        if not is_retry:
            devops_prompt = "Generate CI/CD pipelines and Docker Compose for these stories:\n" + "\n".join(
                f"- {us.id}: {us.title}" for us in user_stories
            )
            docs_prompt = f"Generate API docs and README for: {raw_requirements}"
            global_tasks = [
                ("devops", run_tagged(semaphore, "devops", devops_prompt)),
                ("docs", run_tagged(semaphore, "docs", docs_prompt)),
            ]

        # ── Run all tasks concurrently ─────────────────────────────────────
        all_tasks = per_story_tasks + global_tasks
        results = await asyncio.gather(*[t[1] for t in all_tasks], return_exceptions=True)

        # ── Collect artifacts ──────────────────────────────────────────────
        for (agent_type, _), r in zip(all_tasks, results, strict=False):
            if isinstance(r, list):
                for sub_r in r:
                    if isinstance(sub_r, Artifact):
                        artifacts = _merge_artifact(artifacts, sub_r, is_retry)
            elif isinstance(r, Artifact):
                artifacts = _merge_artifact(artifacts, r, is_retry)
            elif isinstance(r, Exception):
                logger.error(f"Implementation task ({agent_type}) raised: {r}")
            # None results are silently skipped (e.g. skipped retry tasks)

        # ── Test phase: per-story, parallel, isolated failures ────────────
        logger.info("Running Test agents per story...")
        test_tasks = []
        for idx, us in enumerate(user_stories):
            test_tasks.append(_instrumented_test(semaphore, us, idx))

        test_results = await asyncio.gather(*test_tasks, return_exceptions=True)

        failed_stories: list[str] = []
        from pydantic_agents.core.tools.test_tools import execute_test_suite

        for us, res in zip(user_stories, test_results, strict=False):
            if isinstance(res, BaseException):
                logger.error(f"Test agent for {us.id} failed: {res}")
                continue
            res.metadata["user_story_id"] = res.metadata.get("user_story_id", "?")

            # Remove old test artifact for this story
            artifacts = [
                a
                for a in artifacts
                if not (a.agent_name == "test_eng" and a.metadata.get("user_story_id") == res.metadata["user_story_id"])
            ]
            artifacts.append(res)

            # Find matching backend artifact for this story
            target_source_code = ""
            target_code_file = "service.go"
            for a in artifacts:
                if (
                    a.agent_name == "backend"
                    and a.metadata.get("user_story_id") == res.metadata["user_story_id"]
                    and a.files
                ):
                    for fname, fcontent in a.files.items():
                        if fname.endswith(".go") or fname.endswith(".vue") or fname.endswith(".tsx"):
                            target_code_file = fname
                            target_source_code = fcontent
                            break
                    if target_source_code:
                        break

            if res.files and target_source_code:
                test_file_name = next(iter(res.files.keys()))
                test_source_code = res.files[test_file_name]
                trace = await execute_test_suite(test_file_name, test_source_code, target_code_file, target_source_code)
                if trace.startswith("FAILED"):
                    failed_stories.append(us.id)
                    logger.warning(f"Test suite failed for {us.id}: {trace[:200]}...")

        if failed_stories:
            logger.warning(
                f"Test failures in {len(failed_stories)}/{len(user_stories)} stories: {', '.join(failed_stories)}"
            )
            state.feedback = (
                f"Test suite failed for stories: {', '.join(failed_stories)}. "
                "Rewrite the implementations to resolve these errors."
            )
            state.failure_report = FailureReport(
                source="test_suite",
                error_type="test_failure",
                trace=f"Failed stories: {', '.join(failed_stories)}",
                relevant_files=failed_stories,
            )
            state.approved = False
        else:
            logger.info("All tests passed.")
            state.feedback = None
            state.failure_report = None
            state.approved = False

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
