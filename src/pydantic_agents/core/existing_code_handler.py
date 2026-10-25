"""Entry point for modifying EXISTING Go codebases.

REFINED ARCHITECTURE: Autonomous Remediation Loop + Parallel Verification Gates

Pipeline: service_engineer → [parallel: quality_gate + code_reviewer] → auto-remediation (up to 3 retries)
  Instead of one-way sequential, we run:
  1. service_engineer — implements the change
  2. test_eng — generates tests
  3. Parallel: quality_gate (deterministic: lint, test, build)
          + code_reviewer (heuristic: logic, architecture, security)
  4. If either gate fails → auto-route back to service_engineer with specific error context
  5. Max 3 retries, then escalate to Lead Orchestrator with full failure report

This keeps the Lead Orchestrator out of the loop until the code actually passes all gates.
"""

import asyncio
import contextlib
import logging
import os
import time
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import logfire

from pydantic_agents.core.adapters.stage_events import StageEventRecorderPort, build_stage_event_recorder
from pydantic_agents.core.limits import usage_limits
from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.registry import registry
from pydantic_agents.core.telemetry import telemetry
from pydantic_agents.core.utils import format_artifacts

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

_T = TypeVar("_T")


@dataclass
class ExistingCodeResult:
    """Result of an existing code modification pipeline run."""

    passed: bool = False
    changed_files: list[str] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    quality_gate_score: int = 0
    quality_gate_feedback: list[str] = field(default_factory=list)
    review_approved: bool = False
    review_risk: str = ""
    errors: list[str] = field(default_factory=list)
    retries_used: int = 0
    run_id: str = ""


async def _run_stage(
    stage_name: str,
    agent_name: str,
    prompt: str,
    span_attrs: dict | None = None,
) -> tuple[Any, bool]:
    """Run a single pipeline stage with Logfire span + JSONL telemetry.

    Returns:
        (result, success) — result is the agent output or None, success is a bool.
    """
    span_attrs = span_attrs or {}
    agent = registry.get(agent_name)
    start = time.time()
    success = False
    try:
        with logfire.span(
            f"existing_code.{stage_name}",
            agent_name=agent_name,
            prompt_length=len(prompt),
            **span_attrs,
            _level="info",
        ):
            result = await agent.run(prompt, usage_limits=usage_limits(agent_name))
            success = True
            return result, success
    finally:
        duration = time.time() - start
        with contextlib.suppress(Exception):
            await telemetry.log_call(agent_name, duration, success, metadata=span_attrs)


async def _timed(coro: Awaitable[_T]) -> tuple[_T, int]:
    """Await ``coro`` and return its value with the elapsed milliseconds."""
    start = time.time()
    value = await coro
    return value, int((time.time() - start) * 1000)


async def _run_quality_gate(artifacts: list[Artifact]) -> tuple[bool, int, list[str]]:
    """Run the quality gate (deterministic: lint, test, build)."""
    preview = format_artifacts(artifacts, max_chars=30_000, include_agent_header=False)
    prompt = (
        f"Perform a quality gate check on these code changes. "
        f"Check for over-engineering, premature abstraction, "
        f"reinventing the wheel, and unnecessary complexity.\n\n"
        f"{preview}"
    )
    try:
        result, _ = await _run_stage("quality_gate", "quality_gate", prompt, {"artifact_count": len(artifacts)})
        data = result.output
        return data.passed, data.complexity_score, data.feedback
    except Exception as e:
        logger.warning("Quality Gate failed: %s", e)
        return False, 0, [f"Quality Gate error: {e}"]


async def _run_code_review(artifacts: list[Artifact]) -> tuple[bool, str, list[str]]:
    """Run the code review (heuristic: logic, architecture, security)."""
    preview = format_artifacts(artifacts, max_chars=40_000, include_agent_header=False)
    prompt = (
        f"Review these code changes for production readiness. "
        f"Check for correctness, security issues, architecture fit, "
        f"and CODING_STANDARDS compliance.\n\n{preview}"
    )
    try:
        result, _ = await _run_stage("review", "review", prompt, {"artifact_count": len(artifacts)})
        data = result.output
        return data.approved, data.risk_level.value, data.feedback
    except Exception as e:
        logger.warning("Code Review failed: %s", e)
        return False, "unknown", [f"Code Review error: {e}"]


def _record(
    recorder: StageEventRecorderPort,
    result: ExistingCodeResult,
    stage: str,
    attempt: int,
    verdict: str,
    started: float,
    detail: str = "",
) -> None:
    recorder.record(result.run_id, stage, attempt, verdict, int((time.time() - started) * 1000), detail)


def _build_remediation_prompt(
    change_description: str,
    codebase_path: str,
    previous_errors: list[str],
    retry_count: int,
) -> str:
    """Build an enhanced prompt that includes previous gate failure context."""
    error_context = "\n".join(f"  - {e}" for e in previous_errors)
    return (
        f"ATTEMPT {retry_count + 1}/{MAX_RETRIES} — Previous attempt failed.\n\n"
        f"Original change: {change_description}\n\n"
        f"Gate failures from previous attempt:\n{error_context}\n\n"
        f"Address each failure above. Read the files again with read_file, fix the issues, "
        f"then write the corrected versions. Ensure all gates pass this time.\n\n"
        f"Codebase: {codebase_path}"
    )


async def run_existing_code_pipeline(
    change_description: str,
    codebase_path: str,
    run_tests: bool = True,
    run_quality_gate: bool = True,
    run_review: bool = True,
) -> ExistingCodeResult:
    """Modify existing Go code through a multi-agent pipeline with autonomous remediation.

    REFINED FLOW:
      1. service_engineer — implements the change
      2. test_eng — generates/updates tests
      3. PARALLEL: quality_gate + code_reviewer (run concurrently)
      4. If either fails → auto-route back to service_engineer with error context
      5. Max {MAX_RETRIES} retries, then escalate

    Args:
        change_description: What to change and why.
        codebase_path: Absolute path to the existing codebase root.
        run_tests: Whether to generate/update tests.
        run_quality_gate: Whether to check for over-engineering.
        run_review: Whether to run final review.

    Returns:
        ExistingCodeResult with pass/fail, changed files, and check results.
    """
    result = ExistingCodeResult(run_id=str(uuid.uuid4()))
    project = os.path.basename(os.path.realpath(codebase_path)) if codebase_path else ""
    recorder = build_stage_event_recorder(project or "existing-code")

    logger.info("=== Existing Code Pipeline (Autonomous Remediation Loop) ===")
    logger.info("Change: %s", change_description)

    if codebase_path:
        os.environ["ORCHESTRATION_CODEBASE_ROOT"] = os.path.realpath(codebase_path)
    else:
        os.environ.pop("ORCHESTRATION_CODEBASE_ROOT", None)

    for attempt in range(MAX_RETRIES + 1):
        # ── Stage 1: Service Engineer ──────────────────────────────────
        logger.info("Attempt %d/%d — Service Engineer...", attempt + 1, MAX_RETRIES + 1)

        if attempt == 0:
            eng_prompt = (
                f"Modify the codebase at {codebase_path} to implement this change:\n\n"
                f"{change_description}\n\n"
                f"Read the relevant existing files first using the read_file tool. "
                f"Understand the patterns before modifying. Return ONLY the changed files."
            )
        else:
            eng_prompt = _build_remediation_prompt(change_description, codebase_path, result.errors, attempt)

        eng_start = time.time()
        try:
            eng_result, _ = await _run_stage(
                f"service_engineer.attempt_{attempt}",
                "service_engineer",
                eng_prompt,
                {"codebase_path": codebase_path or "auto-detect", "attempt": attempt},
            )
        except Exception as e:
            msg = f"Service Engineer failed on attempt {attempt}: {e}"
            logger.error(msg)
            result.errors.append(msg)
            _record(recorder, result, "service_engineer", attempt, "error", eng_start, str(e))
            continue  # Try again if retries remain

        eng_artifact = eng_result.output
        result.artifacts = [eng_artifact]
        result.changed_files = list(eng_artifact.files.keys())
        logger.info("Changed files: %s", result.changed_files)

        _record(recorder, result, "service_engineer", attempt, "ok", eng_start, f"{len(result.changed_files)} files")

        if not result.changed_files:
            logger.warning("Service Engineer returned no changed files.")
            result.passed = True
            _record(recorder, result, "pipeline", attempt, "passed", eng_start, "no changes")
            return result

        # ── Stage 2: Test Engineer (if enabled) ────────────────────────
        test_failures: list[str] = []
        if run_tests:
            logger.info("Attempt %d/%d — Test Engineer...", attempt + 1, MAX_RETRIES + 1)
            test_start = time.time()
            try:
                test_preview = format_artifacts(result.artifacts, max_chars=15_000, include_agent_header=False)
                test_prompt = (
                    f"Using the generate_go_table_driven_tests tool, write tests for "
                    f"the changed functions in these files:\n\n{test_preview}\n\n"
                    f"Read each changed source file to identify the function signatures, "
                    f"then generate table-driven tests."
                )
                test_result, _ = await _run_stage(
                    f"test_engineer.attempt_{attempt}",
                    "test_eng",
                    test_prompt,
                    {"changed_files_count": len(result.changed_files)},
                )
                if hasattr(test_result.output, "files"):
                    result.artifacts.append(test_result.output)
                    result.test_files = list(test_result.output.files.keys())
                    logger.info("Test files: %s", result.test_files)
                _record(recorder, result, "test_engineer", attempt, "ok", test_start)
            except Exception as e:
                msg = f"Test Engineer failed on attempt {attempt}: {e}"
                logger.warning(msg)
                test_failures.append(msg)
                result.errors.append(msg)
                _record(recorder, result, "test_engineer", attempt, "error", test_start, str(e))

        # ── Stage 3: Parallel Verification Gates ───────────────────────
        if not run_quality_gate and not run_review:
            result.passed = True
            _record(recorder, result, "pipeline", attempt, "passed", eng_start, "gates disabled")
            return result

        logger.info(
            "Attempt %d/%d — Running parallel gates (quality_gate + code_review)...",
            attempt + 1,
            MAX_RETRIES + 1,
        )

        qg_passed = True
        cv_passed = True
        qg_feedback: list[str] = []
        cv_feedback: list[str] = []

        with logfire.span(
            "existing_code.parallel_gates",
            attempt=attempt,
            artifact_count=len(result.artifacts),
            _level="info",
        ):
            tasks: list[Any] = []
            if run_quality_gate:
                tasks.append(_timed(_run_quality_gate(result.artifacts)))
            else:
                tasks.append(_timed(asyncio.sleep(0, result=(True, 0, []))))

            if run_review:
                tasks.append(_timed(_run_code_review(result.artifacts)))
            else:
                tasks.append(_timed(asyncio.sleep(0, result=(True, "low", []))))

            gate_results = await asyncio.gather(*tasks, return_exceptions=True)

            gate_idx = 0
            qr = gate_results[gate_idx]
            if isinstance(qr, BaseException):
                qg_passed = False
                qg_feedback = [f"Quality Gate exception: {qr}"]
                recorder.record(result.run_id, "quality_gate", attempt, "error", 0, str(qr))
            else:
                (qg_passed, score, qg_feedback), qg_ms = qr
                result.quality_gate_score = score
                result.quality_gate_feedback = qg_feedback
                recorder.record(
                    result.run_id,
                    "quality_gate",
                    attempt,
                    "passed" if qg_passed else "failed",
                    qg_ms,
                    "; ".join(qg_feedback),
                )
            gate_idx += 1

            rr = gate_results[gate_idx]
            if isinstance(rr, BaseException):
                cv_passed = False
                cv_feedback = [f"Code Review exception: {rr}"]
                recorder.record(result.run_id, "review", attempt, "error", 0, str(rr))
            else:
                (cv_passed, risk, cv_feedback), cv_ms = rr
                result.review_risk = risk
                recorder.record(
                    result.run_id, "review", attempt, "approved" if cv_passed else "rejected", cv_ms, f"risk={risk}"
                )
                # Collect code review feedback as errors too
                result.errors.extend(cv_feedback)
            gate_idx += 1

        # ── Remediation Decision ────────────────────────────────────────
        logger.info(
            "Gates: quality_gate=%s, review=%s, test_errors=%s",
            qg_passed,
            cv_passed,
            bool(test_failures),
        )

        if qg_passed and cv_passed and not test_failures:
            # All gates passed!
            result.passed = True
            result.retries_used = attempt
            logger.info("All gates passed on attempt %d.", attempt + 1)
            _record(recorder, result, "pipeline", attempt, "passed", eng_start)
            _sync_memory_after_pipeline()
            return result

        # Append gate errors to result errors for next attempt context
        for err in qg_feedback + cv_feedback:
            if err not in result.errors:
                result.errors.append(err)

        if attempt < MAX_RETRIES:
            logger.info("Gates failed — auto-remediation attempt %d/%d", attempt + 1, MAX_RETRIES)
        else:
            logger.warning("Max retries (%d) reached. Escalating to Lead Orchestrator.", MAX_RETRIES)

    _sync_memory_after_pipeline()

    # All retries exhausted
    result.passed = False
    result.retries_used = MAX_RETRIES
    recorder.record(result.run_id, "pipeline", MAX_RETRIES, "failed", 0, f"{len(result.errors)} errors")
    logger.error(
        "Pipeline failed after %d retries. Errors: %s",
        MAX_RETRIES,
        result.errors,
    )
    return result


def _bullet_section(title: str, items: list[str]) -> list[str]:
    """A title line followed by one bullet per item; nothing when there are no items."""
    return [title, *(f"  - {item}" for item in items)] if items else []


def format_result_summary(result: ExistingCodeResult) -> str:
    """Format the pipeline result into a human-readable summary."""
    lines = [
        "## Existing Code Pipeline Results",
        "",
        f"**Status**: {'PASSED ✅' if result.passed else 'FAILED 🛑'}",
        f"**Attempts**: {result.retries_used + 1} / {MAX_RETRIES + 1}",
        *_bullet_section("\n**Changed files**:", result.changed_files),
        *_bullet_section("\n**Test files**:", result.test_files),
        *_bullet_section(f"\n**Quality Gate**: score={result.quality_gate_score}", result.quality_gate_feedback),
        f"\n**Review**: {'approved ✅' if result.passed else 'rejected 🛑'} (risk: {result.review_risk})",
        # Show only the last 5 errors to avoid noise
        *_bullet_section("\n**Errors blocking completion**:", result.errors[-5:]),
    ]
    if len(result.errors) > 5:
        lines.append(f"  - (... and {len(result.errors) - 5} more)")

    return "\n".join(lines)


def _sync_memory_after_pipeline() -> None:
    """Auto-sync pipeline results to Claude Code and Antigravity memory."""
    _logger = logging.getLogger(__name__)
    try:
        from pydantic_agents.core.memory_bridge import sync_memory_to_claude

        sync_memory_to_claude(max_entries=25)
        _logger.info("Memory bridge synced after pipeline run")
    except Exception as e:
        _logger.warning("Memory bridge sync after pipeline failed: %s", e)
