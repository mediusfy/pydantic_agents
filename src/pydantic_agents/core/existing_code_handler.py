"""Entry point for modifying EXISTING Go codebases.

REFINED ARCHITECTURE: Autonomous Remediation Loop + Verification Gates

Pipeline: service_engineer → test_eng → verifier → [parallel: quality_gate + code_reviewer] → auto-remediation
  1. service_engineer — implements the change (reads the codebase through sandboxed tools)
  2. test_eng — generates tests
  3. verifier — deterministic: go build / go vet / go test on a copy of the codebase with the changes overlaid
  4. Parallel: quality_gate (over-engineering) + code_reviewer (logic, architecture, security), both reading
     the verifier's report
  5. If anything fails → auto-route back to service_engineer with the specific error context
  6. Max 3 retries, then escalate to the caller with the full failure report

The changed files are written under generated-app/<codebase name>/ (``ExistingCodeResult.output_dir``); nothing is
applied to the codebase itself. Every agent call goes through ``registry.run`` (permit, timeout, retry, usage caps,
trace tags), and the sandbox root is a context variable, so concurrent pipelines on different codebases are
isolated.
"""

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import logfire

from pydantic_agents.core.adapters.stage_events import StageEventRecorderPort, build_stage_event_recorder
from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.registry import registry
from pydantic_agents.core.utils import format_artifacts, sandbox_root_var
from pydantic_agents.core.verifier import VerificationReport, verify_artifacts, verify_tree

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
    verification: str = ""
    output_dir: str = ""


async def _run_stage(
    stage_name: str,
    agent_name: str,
    prompt: str,
    span_attrs: dict | None = None,
) -> tuple[Any, bool]:
    """Run a single pipeline stage inside a span, through the registry (permit, timeout, retry, telemetry).

    Returns:
        (result, success) — result is the agent run result, success is True (a failure raises).
    """
    span_attrs = span_attrs or {}
    with logfire.span(
        f"existing_code.{stage_name}",
        agent_name=agent_name,
        prompt_length=len(prompt),
        **span_attrs,
        _level="info",
    ):
        result = await registry.run(agent_name, prompt)
        return result, True


async def _timed(coro: Awaitable[_T]) -> tuple[_T, int]:
    """Await ``coro`` and return its value with the elapsed milliseconds."""
    start = time.time()
    value = await coro
    return value, int((time.time() - start) * 1000)


async def _run_quality_gate(artifacts: list[Artifact], verification: str = "") -> tuple[bool, int, list[str]]:
    """Run the quality gate agent (over-engineering, premature abstraction) over the changes and the verifier report."""
    preview = format_artifacts(artifacts, max_chars=30_000, include_agent_header=False)
    prompt = (
        f"Perform a quality gate check on these code changes. "
        f"Check for over-engineering, premature abstraction, "
        f"reinventing the wheel, and unnecessary complexity.\n\n"
        f"{preview}\n\n{verification}"
    )
    try:
        result, _ = await _run_stage("quality_gate", "quality_gate", prompt, {"artifact_count": len(artifacts)})
        data = result.output
        return data.passed, data.complexity_score, data.feedback
    except Exception as e:
        logger.warning("Quality Gate failed: %s", e)
        return False, 0, [f"Quality Gate error: {e}"]


async def _run_code_review(artifacts: list[Artifact], verification: str = "") -> tuple[bool, str, list[str]]:
    """Run the code review (heuristic: logic, architecture, security) over the changes and the verifier's report."""
    preview = format_artifacts(artifacts, max_chars=40_000, include_agent_header=False)
    prompt = (
        f"Review these code changes for production readiness. "
        f"Check for correctness, security issues, architecture fit, "
        f"and CODING_STANDARDS compliance.\n\n{preview}\n\n{verification}"
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


# Returned by _run_engineer when the stage raised (the error is already recorded on the result).
_STAGE_FAILED: Any = object()


def _engineer_prompt(attempt: int, change_description: str, codebase_path: str, previous_errors: list[str]) -> str:
    if attempt > 0:
        return _build_remediation_prompt(change_description, codebase_path, previous_errors, attempt)
    return (
        f"Modify the codebase at {codebase_path} to implement this change:\n\n"
        f"{change_description}\n\n"
        f"Read the relevant existing files first using the read_file tool. "
        f"Understand the patterns before modifying. Return ONLY the changed files."
    )


async def _run_engineer(
    result: ExistingCodeResult,
    recorder: StageEventRecorderPort,
    attempt: int,
    change_description: str,
    codebase_path: str,
) -> tuple[Any, float]:
    """Run the service engineer: (its artifact or _STAGE_FAILED, the start time)."""
    prompt = _engineer_prompt(attempt, change_description, codebase_path, result.errors)
    start = time.time()
    try:
        eng_result, _ = await _run_stage(
            f"service_engineer.attempt_{attempt}",
            "service_engineer",
            prompt,
            {"codebase_path": codebase_path or "auto-detect", "attempt": attempt},
        )
    except Exception as e:
        msg = f"Service Engineer failed on attempt {attempt}: {e}"
        logger.error(msg)
        result.errors.append(msg)
        _record(recorder, result, "service_engineer", attempt, "error", start, str(e))
        return _STAGE_FAILED, start
    return eng_result.output, start


async def _run_test_engineer(result: ExistingCodeResult, recorder: StageEventRecorderPort, attempt: int) -> list[str]:
    """Generate or update tests for the changed files; returns the failure messages (empty on success)."""
    logger.info("Attempt %d/%d — Test Engineer...", attempt + 1, MAX_RETRIES + 1)
    start = time.time()
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
        _record(recorder, result, "test_engineer", attempt, "ok", start)
    except Exception as e:
        msg = f"Test Engineer failed on attempt {attempt}: {e}"
        logger.warning(msg)
        result.errors.append(msg)
        _record(recorder, result, "test_engineer", attempt, "error", start, str(e))
        return [msg]
    return []


def _apply_quality_gate_outcome(
    result: ExistingCodeResult, recorder: StageEventRecorderPort, attempt: int, outcome: Any
) -> tuple[bool, list[str]]:
    """Fold the quality gate's outcome (a value with its duration, or the exception it raised) into the result."""
    if isinstance(outcome, BaseException):
        recorder.record(result.run_id, "quality_gate", attempt, "error", 0, str(outcome))
        return False, [f"Quality Gate exception: {outcome}"]
    (passed, score, feedback), duration_ms = outcome
    result.quality_gate_score = score
    result.quality_gate_feedback = feedback
    recorder.record(
        result.run_id, "quality_gate", attempt, "passed" if passed else "failed", duration_ms, "; ".join(feedback)
    )
    return passed, feedback


def _apply_review_outcome(
    result: ExistingCodeResult, recorder: StageEventRecorderPort, attempt: int, outcome: Any
) -> tuple[bool, list[str]]:
    """Fold the code review's outcome into the result; its feedback is kept as errors even when it approves."""
    if isinstance(outcome, BaseException):
        recorder.record(result.run_id, "review", attempt, "error", 0, str(outcome))
        return False, [f"Code Review exception: {outcome}"]
    (approved, risk, feedback), duration_ms = outcome
    result.review_risk = risk
    recorder.record(
        result.run_id, "review", attempt, "approved" if approved else "rejected", duration_ms, f"risk={risk}"
    )
    if not approved:
        result.errors.extend(feedback)  # an approved review's remarks are not errors for the next attempt
    return approved, feedback


async def _run_gates(
    result: ExistingCodeResult,
    recorder: StageEventRecorderPort,
    attempt: int,
    run_quality_gate: bool,
    run_review: bool,
) -> tuple[bool, bool, list[str], list[str]]:
    """Run the quality gate and the code review concurrently (a disabled one counts as passed).

    Returns (quality gate passed, review approved, quality gate feedback, review feedback).
    """
    with logfire.span(
        "existing_code.parallel_gates",
        attempt=attempt,
        artifact_count=len(result.artifacts),
        _level="info",
    ):
        quality = (
            _timed(_run_quality_gate(result.artifacts, result.verification))
            if run_quality_gate
            else _timed(asyncio.sleep(0, result=(True, 0, [])))
        )
        review = (
            _timed(_run_code_review(result.artifacts, result.verification))
            if run_review
            else _timed(asyncio.sleep(0, result=(True, "low", [])))
        )
        quality_outcome, review_outcome = await asyncio.gather(quality, review, return_exceptions=True)

        qg_passed, qg_feedback = _apply_quality_gate_outcome(result, recorder, attempt, quality_outcome)
        cv_passed, cv_feedback = _apply_review_outcome(result, recorder, attempt, review_outcome)
    return qg_passed, cv_passed, qg_feedback, cv_feedback


@dataclass(frozen=True)
class _PipelineOptions:
    change_description: str
    codebase_path: str
    run_tests: bool
    run_quality_gate: bool
    run_review: bool


async def _run_verifier(
    result: ExistingCodeResult, recorder: StageEventRecorderPort, attempt: int, codebase: str
) -> list[str]:
    """Build, vet and test the changes with the real toolchain; returns the failures (empty when they pass or the
    check could not apply, e.g. no Go code)."""
    changed: dict[str, str] = {}
    for art in result.artifacts:
        changed.update(art.files or {})
    start = time.time()
    try:
        report: VerificationReport = (
            await verify_tree(codebase, changed) if codebase else await verify_artifacts(result.artifacts)
        )
    except Exception as e:  # the verifier is defensive, but a bug in it must not end the run
        logger.warning("Verifier failed on attempt %d: %s", attempt, e)
        _record(recorder, result, "verifier", attempt, "error", start, str(e))
        result.verification = f"VERIFICATION: error ({e})"
        return []
    result.verification = report.as_prompt()
    if report.skipped:
        _record(recorder, result, "verifier", attempt, "skipped", start, report.skipped)
        return []
    failures = report.failures()
    _record(recorder, result, "verifier", attempt, "passed" if report.passed else "failed", start, "; ".join(failures))
    return failures


async def _run_attempt(
    result: ExistingCodeResult, recorder: StageEventRecorderPort, attempt: int, options: _PipelineOptions
) -> bool:
    """One pass of engineer → tests → gates. True when the pipeline is finished and passed; False to try again."""
    # ── Stage 1: Service Engineer ──────────────────────────────────
    logger.info("Attempt %d/%d — Service Engineer...", attempt + 1, MAX_RETRIES + 1)
    eng_artifact, eng_start = await _run_engineer(
        result, recorder, attempt, options.change_description, options.codebase_path
    )
    if eng_artifact is _STAGE_FAILED:
        return False

    result.artifacts = [eng_artifact]
    result.changed_files = list(eng_artifact.files.keys())
    logger.info("Changed files: %s", result.changed_files)
    _record(recorder, result, "service_engineer", attempt, "ok", eng_start, f"{len(result.changed_files)} files")

    if not result.changed_files:
        logger.warning("Service Engineer returned no changed files.")
        result.passed = True
        _record(recorder, result, "pipeline", attempt, "passed", eng_start, "no changes")
        return True

    # ── Stage 2: Test Engineer (if enabled) ────────────────────────
    test_failures = await _run_test_engineer(result, recorder, attempt) if options.run_tests else []

    # ── Stage 3: deterministic verification (build / vet / test) ──
    verify_failures = await _run_verifier(result, recorder, attempt, options.codebase_path)
    test_failures = test_failures + verify_failures

    # ── Stage 4: Parallel LLM Gates ────────────────────────────────
    if not options.run_quality_gate and not options.run_review:
        if test_failures:
            result.errors.extend(err for err in test_failures if err not in result.errors)
            return False
        result.passed = True
        _record(recorder, result, "pipeline", attempt, "passed", eng_start, "gates disabled")
        return True

    logger.info(
        "Attempt %d/%d — Running parallel gates (quality_gate + code_review)...",
        attempt + 1,
        MAX_RETRIES + 1,
    )
    qg_passed, cv_passed, qg_feedback, cv_feedback = await _run_gates(
        result, recorder, attempt, options.run_quality_gate, options.run_review
    )

    # ── Remediation Decision ────────────────────────────────────────
    logger.info("Gates: quality_gate=%s, review=%s, test_errors=%s", qg_passed, cv_passed, bool(test_failures))

    if qg_passed and cv_passed and not test_failures:
        result.passed = True
        result.retries_used = attempt
        logger.info("All gates passed on attempt %d.", attempt + 1)
        _record(recorder, result, "pipeline", attempt, "passed", eng_start)
        _sync_memory_after_pipeline()
        return True

    # Append gate errors to result errors for next attempt context
    result.errors.extend(
        err for err in dict.fromkeys(verify_failures + qg_feedback + cv_feedback) if err not in result.errors
    )

    if attempt < MAX_RETRIES:
        logger.info("Gates failed — auto-remediation attempt %d/%d", attempt + 1, MAX_RETRIES)
    else:
        logger.warning("Max retries (%d) reached. Escalating to Lead Orchestrator.", MAX_RETRIES)
    return False


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
    options = _PipelineOptions(change_description, codebase_path, run_tests, run_quality_gate, run_review)

    logger.info("=== Existing Code Pipeline (Autonomous Remediation Loop) ===")
    logger.info("Change: %s", change_description)

    # The sandbox root is per task: a concurrent pipeline on another codebase keeps its own.
    from pydantic_agents.core.limits import run_id_var

    root_token = sandbox_root_var.set(Path(os.path.realpath(codebase_path)) if codebase_path else None)
    run_token = run_id_var.set(run_id_var.get() or f"modify_{result.run_id[:12]}")
    try:
        for attempt in range(MAX_RETRIES + 1):
            if await _run_attempt(result, recorder, attempt, options):
                _write_result(result, project or "existing-code")
                return result

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
        _write_result(result, project or "existing-code")
        return result
    finally:
        if hasattr(recorder, "close"):
            recorder.close()
        run_id_var.reset(run_token)
        sandbox_root_var.reset(root_token)


def _write_result(result: ExistingCodeResult, project: str) -> None:
    """Write the changed files and tests under the output directory; the caller gets the path in the result."""
    if not any(art.files for art in result.artifacts):
        return
    from pydantic_agents.core.adapters.workspace import ArtifactStorageAdapter
    from pydantic_agents.core.settings import settings

    output_dir = Path(settings.output_dir) / project / result.run_id[:8]
    try:
        ArtifactStorageAdapter(str(output_dir)).write_artifacts(result.artifacts)
        result.output_dir = str(output_dir)
    except Exception as e:  # the pipeline result stands even if the disk write fails
        logger.warning("Could not write the changed files to %s: %s", output_dir, e)
        result.errors.append(f"Could not write the changed files to {output_dir}: {e}")


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
        *([f"\n**Written to**: {result.output_dir}"] if result.output_dir else []),
        *([f"\n**Verification**:\n{result.verification}"] if result.verification else []),
        *_bullet_section(f"\n**Quality Gate**: score={result.quality_gate_score}", result.quality_gate_feedback),
        f"\n**Review**: {'approved ✅' if result.passed else 'rejected 🛑'} (risk: {result.review_risk})",
        # Show only the last 5 errors to avoid noise
        *_bullet_section("\n**Errors blocking completion**:", result.errors[-5:]),
    ]
    if len(result.errors) > 5:
        lines.append(f"  - (... and {len(result.errors) - 5} more)")

    return "\n".join(lines)


def _sync_memory_after_pipeline() -> None:
    """Auto-sync pipeline results to Claude Code and Antigravity memory, when MEMORY_BRIDGE_SYNC is on."""
    from pydantic_agents.core.settings import settings

    if not settings.memory_bridge_sync:
        return
    _logger = logging.getLogger(__name__)
    try:
        from pydantic_agents.core.memory_bridge import sync_memory_to_claude

        sync_memory_to_claude(max_entries=25)
        _logger.info("Memory bridge synced after pipeline run")
    except Exception as e:
        _logger.warning("Memory bridge sync after pipeline failed: %s", e)
