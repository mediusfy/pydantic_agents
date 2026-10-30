import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from pydantic_agents.core.limits import run_blocking_with_permit
from pydantic_agents.core.models.core import (
    AgentAdvice,
    Artifact,
    FailureReport,
    Finding,
    PipelineState,
    ReviewResult,
    TestRunReport,
)
from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig
from pydantic_agents.core.pipeline.crewai.approval import finalize_approval as _finalize_approval
from pydantic_agents.core.pipeline.crewai.crews import (
    ArchitectureReviewCrew,
    AuditCrew,
    QaCrew,
    ResearchCrew,
    SdlcCrew,
    SecurityReviewCrew,
)
from pydantic_agents.core.session_recorder import recorder
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)


_CREW_MAP: dict[str, Any] = {
    "SdlcCrew": SdlcCrew,
    "ResearchCrew": ResearchCrew,
    "SecurityReviewCrew": SecurityReviewCrew,
    "QaCrew": QaCrew,
    "ArchitectureReviewCrew": ArchitectureReviewCrew,
    "AuditCrew": AuditCrew,
}


async def run_crewai_pipeline(
    state: PipelineState,
    config: PipelineConfig,
) -> PipelineState:
    bp = BLUEPRINTS.get(config.blueprint, {})
    crew_class_name = bp.get("crew_class", "SdlcCrew")

    if config.dry_run:
        logger.info("DRY RUN: CrewAI crew execution skipped.")
        state.stage = "complete"
        return state

    crew_cls = _CREW_MAP.get(crew_class_name)
    if crew_cls is None:
        msg = f"Unknown CrewAI crew class: {crew_class_name}"
        logger.error(msg)
        state.errors.append(msg)
        state.failure_report = FailureReport(source="crewai", error_type="config", trace=msg)
        return state

    async with recorder.record_span(
        run_id=state.run_id,
        agent_name=f"crewai_{crew_class_name}",
        pipeline_stage="implementation",
        prompt=config.requirement or "",
    ) as rec:
        try:
            from pydantic_agents.core.pipeline.context import ContextFrame

            logger.info("Running %s...", crew_class_name)
            staged_runner = _STAGED_RUNNERS.get(crew_cls)
            if staged_runner is not None:
                summary = await staged_runner(state)
                rec.set_output({"crew": crew_class_name, "result": summary[:500]})
            else:
                crew = crew_cls().build(state)
                result = await _run_crew_async(crew)
                rec.set_output({"crew": crew_class_name, "result": str(result)[:500]})
                _apply_result(state, result)

            files: list[str] = []
            for art in state.artifacts or []:
                if art.files:
                    files.extend(art.files.keys())
            cf = ContextFrame(
                output=f"CrewAI execution for {crew_class_name} completed.",
                modified_files=files,
            )
            state.active_context = cf
            state.context_history.append(cf)

            state.stage = "implementation"
        except BaseException as e:
            logger.exception("CrewAI pipeline failed: %s", e)
            state.errors.append(f"crewai: {e}")
            state.failure_report = FailureReport(source="crewai", error_type="runtime_error", trace=str(e))
            state.approved = False
            rec.abort_run(str(e) or type(e).__name__)  # the span would otherwise be recorded as a success
            if not isinstance(e, Exception):
                raise

    _finalize_approval(state, crew_class_name)
    return state


async def _run_crew_async(crew: Any) -> Any:
    return await run_blocking_with_permit(crew.kickoff, settings.agent_timeout_seconds)


def _qa_gate_reason(report: TestRunReport | None) -> str | None:
    """Why the QA gate is not met, or None when it is: no parsed report, a failing run, or coverage under the floor."""
    if report is None:
        return "qa_executor's test-execution report did not parse"
    if not report.all_passed:
        return "test run still failing"
    if report.coverage_percent < settings.qa_min_coverage_percent:
        return f"coverage {report.coverage_percent}% below floor {settings.qa_min_coverage_percent}%"
    return None


async def _run_qa_staged_gate(state: PipelineState) -> str:
    """QaCrew's staged run: tests, then a bounded execute-and-fix loop, then sign-off behind a fail-closed gate.

    ``Process.sequential`` runs each task exactly once, so the loop can't live inside one ``Crew``: this runs a
    bounded sequence of single-purpose crews instead, re-running ``qa_executor``/``go_test`` up to
    ``settings.qa_max_fix_rounds`` times before the review sign-off runs regardless. The final ``TestRunReport``
    (not the reviewer's own verdict) gates ``state.approved``: a reviewer can't approve what the test run itself
    still says is failing or under the coverage floor. ``state.test_run_report`` is the last parsed report, or
    None when ``qa_executor``'s output never parsed, so "no report" and "an empty report" stay distinguishable.
    """
    qa = QaCrew()

    tests_output = await _run_qa_tests(state, qa)
    report, fix_rounds, tests_output = await _qa_execute_and_fix(state, qa, tests_output)
    state.test_run_report = report

    signoff_result = await _run_crew_async(qa.build_signoff(tests_output, report))
    signoff_outputs = list(getattr(signoff_result, "tasks_output", None) or [])
    if signoff_outputs:
        _apply_verdict(state, _output_text(signoff_outputs[0]))

    gate_reason = _qa_gate_reason(report)
    if gate_reason is not None:
        state.approved = False
        if report is None:
            state.errors.append(f"crewai QaCrew: {gate_reason}; no fix round was run and approval is blocked")
        else:
            state.errors.append(f"crewai QaCrew: fix loop exhausted after {fix_rounds} round(s), {gate_reason}")

    if report is None:
        return f"QaCrew finished: report=unparsed fix_rounds={fix_rounds} approved={state.approved}"
    return (
        f"QaCrew finished: report=parsed fix_rounds={fix_rounds} all_passed={report.all_passed} "
        f"coverage={report.coverage_percent} approved={state.approved}"
    )


async def _run_qa_tests(state: PipelineState, qa: QaCrew) -> str:
    """Run the test-writing crew, merge its artifacts into the state, and return the last task's output."""
    tests_result = await _run_crew_async(qa.build_tests(state))
    tests_outputs = list(getattr(tests_result, "tasks_output", None) or [])
    for task_out in tests_outputs:
        target = _owner_and_kind(task_out)
        if target and target[1] == "artifact":
            _merge_artifact(state, _artifact_from_output(target[0], _output_text(task_out)))
    return _output_text(tests_outputs[-1]) if tests_outputs else ""


async def _qa_execute_and_fix(
    state: PipelineState, qa: QaCrew, tests_output: str
) -> tuple[TestRunReport | None, int, str]:
    """The bounded execute-and-fix loop: (the last parsed report or None, fix rounds run, the latest test output)."""
    report: TestRunReport | None = None
    fix_rounds = 0
    for round_num in range(settings.qa_max_fix_rounds + 1):
        exec_result = await _run_crew_async(qa.build_execution(tests_output))
        exec_outputs = list(getattr(exec_result, "tasks_output", None) or [])
        raw_report = _output_text(exec_outputs[0]) if exec_outputs else ""
        report, reason = _parse_json_model(TestRunReport, raw_report)
        if report is None:
            state.errors.append(f"crewai qa_executor: {reason}")
            logger.warning("QaCrew execution report rejected on round %d: %s", round_num, reason)
            # qa_executor's output didn't parse, not the tests failing: a fix round would hand go_test a
            # meaningless empty failing-id list, so stop retrying and let the fail-closed gate block.
            break
        if _qa_gate_reason(report) is None or round_num >= settings.qa_max_fix_rounds:
            break

        fix_rounds += 1
        fix_result = await _run_crew_async(qa.build_fix(tests_output, report))
        fix_outputs = list(getattr(fix_result, "tasks_output", None) or [])
        if fix_outputs:
            tests_output = _output_text(fix_outputs[0])
            _merge_artifact(state, _artifact_from_output("test_eng", tests_output))
    return report, fix_rounds, tests_output


# Crews whose stages the bridge orchestrates itself instead of one ``build(state)`` + ``kickoff``. Keyed by class,
# not by the blueprint's ``crew_class`` name, so renaming a ``_CREW_MAP`` entry cannot silently drop the runner.
_STAGED_RUNNERS: dict[type, Callable[[PipelineState], Awaitable[str]]] = {QaCrew: _run_qa_staged_gate}


# ── Mapping crew output back onto the pipeline state ──────────────────────


def _output_text(task_output: Any) -> str:
    raw = getattr(task_output, "raw", None)
    if isinstance(raw, str) and raw:
        return raw
    output = getattr(task_output, "output", None)
    return str(output) if output is not None else str(task_output)


def _role_of(task_output: Any) -> str:
    agent = getattr(task_output, "agent", None)
    if isinstance(agent, str):
        return agent
    return str(getattr(agent, "role", None) or getattr(agent, "name", None) or "")


def _owner_and_kind(task_output: Any) -> tuple[str, str] | None:
    """The ``<owner>/<kind>`` a crew task declared, or a guess from the agent role for tasks that did not."""
    name = getattr(task_output, "name", None)
    if isinstance(name, str) and "/" in name:
        owner, _, kind = name.partition("/")
        return owner, kind
    role = _role_of(task_output).lower()
    if not role:
        return None
    if "business analyst" in role or role == "ba":
        return "ba", "note"
    if "cto" in role:
        return "cto", "note"
    if "backend" in role:
        return "backend", "artifact"
    if "frontend" in role:
        return "frontend", "artifact"
    if "qa" in role or "test" in role:
        return "test_eng", "artifact"
    if "review" in role:
        return "review", "verdict"
    return role, "note"


def _apply_result(state: PipelineState, result: Any) -> None:
    for task_out in getattr(result, "tasks_output", None) or []:
        target = _owner_and_kind(task_out)
        if target is None:
            continue
        owner, kind = target
        output = _output_text(task_out)
        if kind == "artifact":
            _merge_artifact(state, _artifact_from_output(owner, output))
        elif kind == "verdict":
            _apply_verdict(state, output)
        elif kind == "advice":
            _apply_advice(state, owner, output)
        elif kind == "feedback":
            state.feedback = output
        elif kind == "findings":
            _apply_findings(state, owner, output)
        else:
            state.advisory_results[owner] = output


def _first_json_array(text: str) -> str | None:
    """The first complete JSON array in ``text`` (models often wrap it in prose or a code fence)."""
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return text[idx:end]
    return None


def _apply_findings(state: PipelineState, owner: str, output: str) -> None:
    """Drop findings missing a file:line/resource id; reject the whole batch if any other finding is invalid or
    lacks an owner or due date (the consolidating review task is required to fill both in, never leave the gate to
    downstream)."""
    payload = _first_json_array(output)
    if payload is None:
        state.errors.append(f"crewai {owner} findings: no JSON array found in output")
        logger.warning("CrewAI %s findings rejected: no JSON array found", owner)
        return
    try:
        raw_items = json.loads(payload)
    except json.JSONDecodeError as exc:
        state.errors.append(f"crewai {owner} findings: invalid JSON ({exc})")
        logger.warning("CrewAI %s findings rejected: invalid JSON (%s)", owner, exc)
        return

    kept: list[Finding] = []
    dropped = 0
    for item in raw_items:
        # Only a missing location is a reason to drop; any other schema violation rejects the batch, so a real
        # finding is never discarded under the "no location" count.
        if isinstance(item, dict) and not str(item.get("location") or "").strip():
            dropped += 1
            continue
        try:
            kept.append(Finding.model_validate(item))
        except ValidationError as exc:
            state.errors.append(f"crewai {owner} findings: rejected, invalid finding ({exc.error_count()} error(s))")
            logger.warning("CrewAI %s findings rejected: invalid finding (%s)", owner, exc)
            return

    missing = [f.id for f in kept if not f.owner or not f.due_date]
    if missing:
        state.errors.append(f"crewai {owner} findings: rejected, missing owner/due_date for {missing}")
        logger.warning("CrewAI %s findings rejected: missing owner/due_date for %s", owner, missing)
        return

    lines = [f"- [{f.severity.value}] {f.location}: {f.description} (owner={f.owner}, due={f.due_date})" for f in kept]
    if dropped:
        lines.append(f"({dropped} finding(s) dropped for missing a file:line or resource id)")
    state.feedback = "\n".join(lines) if lines else "No findings."


def _first_json_object(text: str) -> str | None:
    """The first complete JSON object in ``text`` (models often wrap it in prose or a code fence)."""
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return text[idx:end]
    return None


def _parse_json_model(model: type[Any], text: str) -> tuple[Any | None, str | None]:
    """Parse the first JSON object of ``text`` as ``model``; returns (instance, None) or (None, reason)."""
    payload = _first_json_object(text)
    if payload is None:
        return None, f"no JSON object found in {model.__name__} output"
    try:
        return model.model_validate_json(payload), None
    except ValidationError as exc:
        return None, f"{model.__name__} output failed validation: {exc.errors()[0].get('msg', exc)}"


def _apply_verdict(state: PipelineState, output: str) -> None:
    verdict, reason = _parse_json_model(ReviewResult, output)
    if verdict is None:
        state.approved = False
        state.feedback = output
        state.errors.append(f"crewai review: {reason}")
        logger.warning("CrewAI review verdict rejected: %s", reason)
        return
    state.approved = verdict.approved
    state.feedback = "\n".join(verdict.feedback) if verdict.feedback else output


def _apply_advice(state: PipelineState, owner: str, output: str) -> None:
    state.advisory_results[owner] = output
    advice, reason = _parse_json_model(AgentAdvice, output)
    if advice is None:
        state.cto_approved = False
        state.errors.append(f"crewai {owner} advice: {reason}")
        logger.warning("CrewAI %s advice rejected: %s", owner, reason)
        return
    state.cto_approved = advice.approved_to_proceed


def _merge_artifact(state: PipelineState, artifact: Artifact) -> None:
    """Keep one artifact per owner: a later task of the same owner (QaCrew's fix task after its test task) updates
    the files it re-emits and adds the rest, instead of leaving two artifacts for consumers keyed by owner."""
    for idx, existing in enumerate(state.artifacts):
        if existing.agent_name == artifact.agent_name:
            # A new instance, not an in-place update: an Artifact may already be referenced elsewhere.
            state.artifacts[idx] = existing.model_copy(update={"files": {**existing.files, **artifact.files}})
            return
    state.artifacts.append(artifact)


def _artifact_from_output(agent_name: str, output: str) -> Artifact:
    from pydantic_agents.core.utils import parse_code_blocks_to_files

    return Artifact(
        agent_name=agent_name,
        files=parse_code_blocks_to_files(output),
        metadata={"agent_name": agent_name},
    )
