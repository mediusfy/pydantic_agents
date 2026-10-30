"""Quality gate: deterministic verification first, then the SRE, security, quality-gate and review agents.

Order and fail-closed rules:

1. ``verify_artifacts`` builds, vets and tests the Go files with the real toolchain. Its report is handed to every
   LLM gate, and a failed build stops the stage with a ``quality_gate_validation`` failure (no model is asked to
   review code that does not compile).
2. The SRE agent writes the infrastructure for the requirement and stories (it used to get no context at all).
3. Security, quality gate and review judge the artifacts in batches that fit their character budget, so a large
   run is reviewed completely instead of being truncated and judged on what was left. A batch that fails stops the
   stage with that gate's failure report.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from pydantic_agents.agents.review import ReviewCriteria
from pydantic_agents.core.limits import llm_semaphore
from pydantic_agents.core.models.core import Artifact, FailureReport, PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.registry import registry
from pydantic_agents.core.session_recorder import recorder
from pydantic_agents.core.stages.base import StageRunner
from pydantic_agents.core.utils import format_artifacts
from pydantic_agents.core.validators import CodeValidator
from pydantic_agents.core.verifier import VerificationReport, verify_artifacts

logger = logging.getLogger(__name__)

# Character budget per gate prompt. Artifacts are batched so that no batch exceeds it; a single artifact larger
# than the budget is reviewed alone and the gate is told it is reading a truncated file.
GATE_BATCH_CHARS = 80_000


def _batches(artifacts: Sequence[Artifact], max_chars: int = GATE_BATCH_CHARS) -> list[list[Artifact]]:
    """Group artifacts into batches whose formatted size stays under ``max_chars`` (one oversized artifact alone)."""
    batches: list[list[Artifact]] = []
    current: list[Artifact] = []
    size = 0
    for art in artifacts:
        art_size = sum(len(p) + len(c) + 10 for p, c in art.files.items()) + 60
        if current and size + art_size > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(art)
        size += art_size
    if current:
        batches.append(current)
    return batches


def _reject(state: PipelineState, source: str, error_type: str, feedback: str) -> PipelineState:
    state.feedback = feedback
    state.failure_report = FailureReport(source=source, error_type=error_type, trace=feedback)
    state.approved = False
    return state


def _context_query(state: PipelineState) -> str:
    """What to retrieve context for: the requirement and the story titles, never the artifact dump."""
    titles = ", ".join(us.title for us in state.user_stories)
    return f"{state.requirement}\n{titles}".strip() or "generated service"


class QualityGateStage(StageRunner):
    async def _run_agent(self, state: PipelineState, agent: str, prompt: str, deps: Any = None) -> Any:
        semaphore = llm_semaphore()
        async with recorder.record_span(
            run_id=state.run_id, agent_name=agent, pipeline_stage="quality_gate", prompt=prompt
        ) as rec:
            res = await self.provider.run_agent(semaphore, agent, registry.get(agent), prompt, deps=deps)
            rec.set_output(res)
            return res.output

    async def _retrieved(self, agent: str, state: PipelineState, config: PipelineConfig, heading: str) -> str:
        from pydantic_agents.core.context_engine import retrieve_context

        # The same query is asked once per gate and batch; the answer does not change within a run.
        cache: dict[tuple[str, str, str], str] = self.__dict__.setdefault("_ctx_cache", {})
        query = _context_query(state)
        key = (agent, state.run_id, query)
        if key not in cache:
            cache[key] = await retrieve_context(agent, query, project=config.project)
        ctx = cache[key]
        return f"\n\nRetrieved {heading}:\n{ctx}" if ctx else ""

    async def _run_sre(self, state: PipelineState, config: PipelineConfig, artifacts: list[Artifact]) -> None:
        logger.info("Running SRE Agent...")
        stories = "\n".join(f"- {us.id}: {us.title}" for us in state.user_stories)
        services = sorted({a.metadata.get("service_name") or config.project for a in artifacts if a.files})
        sre_prompt = (
            f"Provision infrastructure and Helm charts for project '{config.project}'.\n\n"
            f"REQUIREMENT:\n{state.requirement}\n\nSTORIES:\n{stories}\n\n"
            f"SERVICES TO DEPLOY: {', '.join(services) or config.project}\n\n"
            f"CTO ADVICE:\n{state.advisory_results.get('cto', '(none)')}"
        ) + await self._retrieved("sre", state, config, "SRE Context/ADRs")
        sre_output = await self._run_agent(state, "sre", sre_prompt)
        # Create a clean copy to avoid mutating the agent's returned artifact
        sre_artifact = sre_output.model_copy(deep=True)
        sre_artifact.metadata["user_story_id"] = "infrastructure"
        sre_artifact.agent_name = "infrastructure"
        # A retry runs this gate again on the same list: replace the earlier infrastructure artifact, do not stack.
        artifacts[:] = [a for a in artifacts if a.agent_name != "infrastructure"]
        artifacts.append(sre_artifact)

    async def _security(self, state, config, text: str, verification: str) -> list[str] | None:
        """Findings when the audit fails, else None."""
        prompt = f"Audit the following code for vulnerabilities:\n{text}\n\n{verification}"
        prompt += await self._retrieved("security", state, config, "Security Context")
        result = await self._run_agent(state, "security", prompt)
        logger.info(f"Security: passed={result.passed}, score={result.score}")
        if result.passed:
            return None
        return list(result.feedback) or ["(the security agent gave no details; re-audit the artifacts)"]

    async def _quality(self, state, config, text: str, verification: str) -> list[str] | None:
        prompt = f"Perform quality gate check on these artifacts:\n{text}\n\n{verification}"
        prompt += await self._retrieved("quality_gate", state, config, "Quality Gate Rules")
        result = await self._run_agent(state, "quality_gate", prompt)
        logger.info(f"Quality Gate: passed={result.passed}, complexity={result.complexity_score}")
        return None if result.passed else list(result.feedback)

    async def _review(self, state, config, text: str, verification: str) -> list[str] | None:
        prompt = (
            f"Review the following generated artifacts:\n{text}\n\n{verification}\n"
            "Ensure code is robust, secure, and meets all criteria."
        )
        if state.feedback:
            prompt += f"\n\nPrevious feedback to verify as resolved (report it only if still present): {state.feedback}"
        prompt += await self._retrieved("review", state, config, "Review Context/Conventions")
        contexts = sorted({us.bounded_context for us in state.user_stories if us.bounded_context})
        result = await self._run_agent(state, "review", prompt, deps=ReviewCriteria(bounded_contexts=contexts))
        logger.info(
            f"Review: approved={result.approved}, risk={result.risk_level}, security_score={result.security_score}"
        )
        return None if result.approved else list(result.feedback)

    @staticmethod
    async def _run_gate(gate: Any, state: Any, config: Any, batches: list[list[Artifact]], verification: str):
        """One gate over every batch: the first batch's findings, or None when all batches pass."""
        for batch in batches:
            findings = await gate(state, config, format_artifacts(batch, max_chars=GATE_BATCH_CHARS), verification)
            if findings is not None:
                return findings
        return None

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        logger.info("--- Running Quality Gate Stage ---")

        if config.dry_run:
            logger.info("DRY RUN: Skipping Quality Gate checks.")
            state.approved = True
            state.stage = "quality_gate"
            return state

        artifacts = state.artifacts
        state.stage = "quality_gate"

        if not artifacts:
            # Nothing was generated (every implementation agent failed): there is nothing to approve, and the SRE
            # agent's own output must not stand in for the code.
            return _reject(
                state,
                "quality_gate_validation",
                "empty",
                "No implementation artifacts were produced; nothing to review. Re-run the implementation.",
            )

        logger.info("Running Code Validation...")
        if not await CodeValidator.run_all_async(artifacts):
            logger.warning("Code validation failed.")
            return _reject(
                state,
                "quality_gate_validation",
                "compilation",
                "Code validation failed during the Quality Gate. Please fix compilation issues.",
            )

        logger.info("Running deterministic verification (go build / vet / test)...")
        report: VerificationReport = await verify_artifacts(artifacts)
        if not report.skipped and not report.passed:
            logger.warning("Verification failed: %s", report.failures())
            return _reject(state, "quality_gate_validation", "compilation", "\n".join(report.failures()))
        verification = report.as_prompt()

        await self._run_sre(state, config, artifacts)

        gates = (
            (
                "security",
                "security_audit",
                "security",
                "Security audit failed. Resolve these findings: \n",
                self._security,
            ),
            ("quality_gate", "quality_gate_agent", "quality_check", "", self._quality),
            ("review", "review_agent", "review_rejection", "", self._review),
        )
        batches = _batches(artifacts)
        logger.info("Reviewing %d artifacts in %d batch(es)", len(artifacts), len(batches))
        # The three gates are independent, so they run side by side (each still walks its batches in order); the
        # first rejection in gate order is reported, as before. Trade-off: a rejection no longer saves the calls
        # of the gates after it.
        outcomes = await asyncio.gather(
            *(self._run_gate(gate, state, config, batches, verification) for *_, gate in gates)
        )
        for (name, source, error_type, prefix, _), findings in zip(gates, outcomes, strict=True):
            if findings is not None:
                logger.warning("%s rejected.", name)
                return _reject(state, source, error_type, prefix + "\n".join(findings))

        state.approved = True
        state.feedback = None
        state.failure_report = None
        return state
