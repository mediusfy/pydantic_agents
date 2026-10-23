import asyncio
import logging

from pydantic_agents.agents.review import ReviewCriteria
from pydantic_agents.core.models.core import FailureReport, PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.registry import registry
from pydantic_agents.core.session_recorder import recorder
from pydantic_agents.core.settings import settings
from pydantic_agents.core.stages.base import StageRunner
from pydantic_agents.core.utils import format_artifacts
from pydantic_agents.core.validators import CodeValidator

logger = logging.getLogger(__name__)


class QualityGateStage(StageRunner):
    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        logger.info("--- Running Quality Gate Stage ---")

        if config.dry_run:
            logger.info("DRY RUN: Skipping Quality Gate checks.")
            state.approved = True
            state.stage = "quality_gate"
            return state

        semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
        artifacts = state.artifacts or []

        logger.info("Running SRE Agent...")
        from pydantic_agents.core.context_engine import retrieve_context

        sre_ctx = retrieve_context("sre", "Provision infrastructure and apply Helm charts.")
        sre_prompt = "Provision infrastructure and apply Helm charts."
        if sre_ctx:
            sre_prompt += f"\n\nRetrieved SRE Context/ADRs:\n{sre_ctx}"

        async with recorder.record_span(
            run_id=state.run_id,
            agent_name="sre",
            pipeline_stage="quality_gate",
            prompt=sre_prompt,
        ) as rec:
            sre_res = await self.provider.run_agent(semaphore, "sre", registry.get("sre"), sre_prompt)
            rec.set_output(sre_res)
            # Create a clean copy to avoid mutating the agent's returned artifact
            sre_artifact = sre_res.output.model_copy(deep=True)
            sre_artifact.metadata["user_story_id"] = "infrastructure"
            sre_artifact.agent_name = "infrastructure"
            artifacts.append(sre_artifact)

        full_text = format_artifacts(artifacts, max_chars=80_000)

        logger.info("Running Code Validation...")
        if not await CodeValidator.run_all_async(artifacts):
            logger.warning("Code validation failed.")
            state.feedback = "Code validation failed during the Quality Gate. Please fix compilation issues."
            state.failure_report = FailureReport(
                source="quality_gate_validation",
                error_type="compilation",
                trace=state.feedback,
            )
            state.approved = False
            return state

        logger.info("Running Security Audit...")
        from pydantic_agents.core.context_engine import retrieve_context

        sec_ctx = retrieve_context("security", full_text)
        sec_prompt = f"Audit the following code for vulnerabilities:\n{full_text}"
        if sec_ctx:
            sec_prompt += f"\n\nRetrieved Security Context:\n{sec_ctx}"

        async with recorder.record_span(
            run_id=state.run_id,
            agent_name="security",
            pipeline_stage="quality_gate",
            prompt=sec_prompt,
        ) as rec:
            sec_res = await self.provider.run_agent(
                semaphore,
                "security",
                registry.get("security"),
                sec_prompt,
            )
            rec.set_output(sec_res)
            sec_result = sec_res.output
            logger.info(f"Security: passed={sec_result.passed}, score={sec_result.score}")
            if not sec_result.passed:
                logger.warning("Security audit failed.")
                findings = (
                    "\n".join(sec_result.feedback) or "(the security agent gave no details; re-audit the artifacts)"
                )
                state.feedback = "Security audit failed. Resolve these findings: \n" + findings
                state.failure_report = FailureReport(
                    source="security_audit",
                    error_type="security",
                    trace=state.feedback,
                )
                state.approved = False
                return state

        logger.info("Running Quality Gate...")
        qg_ctx = retrieve_context("quality_gate", full_text)
        qg_prompt = f"Perform quality gate check on these artifacts:\n{full_text}"
        if qg_ctx:
            qg_prompt += f"\n\nRetrieved Quality Gate Rules:\n{qg_ctx}"

        async with recorder.record_span(
            run_id=state.run_id,
            agent_name="quality_gate",
            pipeline_stage="quality_gate",
            prompt=qg_prompt,
        ) as rec:
            q_res = await self.provider.run_agent(
                semaphore,
                "quality_gate",
                registry.get("quality_gate"),
                qg_prompt,
            )
            rec.set_output(q_res)
            q_result = q_res.output
            logger.info(f"Quality Gate: passed={q_result.passed}, complexity={q_result.complexity_score}")
            if not q_result.passed:
                logger.warning("Quality Gate rejected.")
                state.feedback = "\n".join(q_result.feedback)
                state.failure_report = FailureReport(
                    source="quality_gate_agent",
                    error_type="quality_check",
                    trace=state.feedback,
                )
                state.approved = False
                return state

        logger.info("Running Board of Review...")
        review_prompt = (
            f"Review the following generated artifacts:\n{full_text}\n"
            "Ensure code is robust, secure, and meets all criteria."
        )
        if state.feedback:
            review_prompt += (
                f"\n\nPrevious feedback to verify as resolved (report it only if still present): {state.feedback}"
            )

        review_ctx = retrieve_context("review", full_text)
        if review_ctx:
            review_prompt += f"\n\nRetrieved Review Context/Conventions:\n{review_ctx}"

        async with recorder.record_span(
            run_id=state.run_id,
            agent_name="review",
            pipeline_stage="quality_gate",
            prompt=review_prompt,
        ) as rec:
            contexts = sorted({us.bounded_context for us in state.user_stories if us.bounded_context})
            review_res = await self.provider.run_agent(
                semaphore,
                "review",
                registry.get("review"),
                review_prompt,
                deps=ReviewCriteria(bounded_contexts=contexts),
            )
            rec.set_output(review_res)
            result = review_res.output
            logger.info(
                f"Review: approved={result.approved}, risk={result.risk_level}, security_score={result.security_score}"
            )

            if result.approved:
                state.approved = True
                state.feedback = None
                state.failure_report = None
            else:
                logger.warning("Review rejected. Setting feedback for implementation retry.")
                state.feedback = "\n".join(result.feedback)
                state.failure_report = FailureReport(
                    source="review_agent",
                    error_type="review_rejection",
                    trace=state.feedback,
                )
                state.approved = False

        state.stage = "quality_gate"
        return state
