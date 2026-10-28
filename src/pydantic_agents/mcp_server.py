import asyncio
import contextlib
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import logfire
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core import platform_metrics
from pydantic_agents.core.context_engine import retrieve_context
from pydantic_agents.core.exceptions import AgentExecutionError, AgentNotFoundError, PathTraversalError
from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.memory_bridge import get_bridge_status, sync_claude_to_pgvector, sync_memory_to_claude
from pydantic_agents.core.observability import configure_observability
from pydantic_agents.core.registry import registry
from pydantic_agents.core.settings import settings
from pydantic_agents.core.utils import serialize_output
from pydantic_agents.core.vector_db import vector_db

logger = logging.getLogger(__name__)

configure_observability()

logging.basicConfig(level=logging.INFO)

mcp = FastMCP(
    "Orgitration",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=os.environ.get("MCP_DNS_REBINDING_PROTECTION", "1") == "1"
    ),
)

# ── Dynamic Agent MCP Tools ──────────────────────────────────────────

# Ensure all agents are registered at boot so we can dynamically discover them
register_all_agents()


# Project names reach the file system (output, checkpoint and approval-marker directories), so they are plain names.
_PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _check_project_name(project_name: str) -> str | None:
    """An error message when a project name is not a plain name, else None."""
    if _PROJECT_NAME.fullmatch(project_name) and ".." not in project_name:
        return None
    return f"Invalid project_name {project_name!r}: use letters, digits, '_', '.' and '-' only (at most 64 characters)."


async def run_agent_by_name(agent_name: str, request: str, project: str = "") -> str:
    """Run a registered agent by its name, resolving dependencies and tracking telemetry.

    FORTIFIED CONTEXT MANAGEMENT:
    Before running the agent, this function retrieves relevant snippets from the
    vector-indexed context store (ADRs, architecture rules, past decisions) and
    prepends them to the request. This ensures agents are fed exactly the relevant
    context without needing to manually hunt through the file system. ``project`` scopes
    the retrieval to one seeded project; empty searches every project.

    The call goes through ``registry.run`` (permit, timeout, retry, usage caps) and gets its own run id, so its
    model calls are grouped as one session in Langfuse.

    Returns the agent's answer: plain text for text agents, otherwise JSON
    (``BaseModel.model_dump_json()`` for structured results, a JSON array for lists).
    """
    if not registry.is_registered(agent_name):
        raise AgentNotFoundError(f"Agent '{agent_name}' not registered")

    run_token = run_id_var.set(run_id_var.get() or f"call_{uuid.uuid4().hex[:12]}")
    start = time.time()
    success = False
    verdict: str | None = None
    try:
        # ── Context Enrichment ────────────────────────────────────────────────
        # Pull relevant snippets from the vector-indexed knowledge base
        enriched_request = request
        with logfire.span("mcp.retrieve_context", agent_name=agent_name, _level="info"):
            context = await retrieve_context(agent_name, request, top_k=3, project=project or None)
            if context:
                enriched_request = f"Relevant Architecture Context from Knowledge Base:\n{context}\n\n---\n\n{request}"
        # ──────────────────────────────────────────────────────────────────────

        span_attrs: dict[str, Any] = {
            "agent_name": agent_name,
            "request_length": len(request),
            "context_retrieved": bool(context),
            "langfuse.session.id": run_id_var.get() or "",
        }
        try:
            with logfire.span("mcp.run_agent", **span_attrs, _level="info"):
                result = await registry.run(agent_name, enriched_request)
                success = True
                verdict = _extract_verdict(result)
                return serialize_output(result)
        except Exception as e:
            logger.warning("Agent '%s' failed: %s", agent_name, e)
            raise AgentExecutionError(f"Agent '{agent_name}' failed: {e}", is_retryable=False) from e
    finally:
        run_id_var.reset(run_token)
        # Additive to the provider's JSONL telemetry entry: one platform_metrics row per MCP call, best-effort.
        with contextlib.suppress(Exception):
            await platform_metrics.record_call(
                agent_name, duration=time.time() - start, success=success, verdict=verdict
            )


def _extract_verdict(result: Any) -> str | None:
    """'passed'/'failed' from a QualityGateResult.passed or ReviewResult.approved output, else None."""
    output = getattr(result, "output", result)
    for field in ("passed", "approved"):
        value = getattr(output, field, None)
        if isinstance(value, bool):
            return "passed" if value else "failed"
    return None


AGENT_DOCSTRINGS = {
    "holiday_planner": (
        "Invokes the Holiday Planner Agent. "
        "Returns a JSON Itinerary (days, items with confirm-where, budget check, or open_questions)."
    ),
    "region_analyzer": "Invokes the Region Analyzer Agent.",
    "asymmetric_reviewer": "Invokes the Asymmetric Reviewer Agent (law & corporate tricks for travel savings).",
    "current_events": "Invokes the Current Events Agent (history buff).",
    "weather": "Invokes the Weather Agent.",
    "accountant": (
        "Invokes the Accountant Agent. "
        "Returns a JSON CostBreakdown (cost lines, totals, exchange rates, or open_questions)."
    ),
    "ba": "Invokes the Business Analyst (BA) agent to write user stories and requirements.",
    "cto": "Invokes the CTO agent for architectural guidance and advisories.",
    "backend": "Invokes the Backend Engineer agent to generate high-concurrency Go services.",
    "frontend": "Invokes the Frontend Engineer agent to generate user interfaces.",
    "database": "Invokes the Database agent to generate schema migrations and seed scripts.",
    "devops": "Invokes the DevOps agent to generate CI/CD pipelines and docker-compose configurations.",
    "docs": "Invokes the Documentation agent to generate API documentation and READMEs.",
    "sre": "Invokes the SRE agent to generate Kubernetes/Helm infrastructure manifests.",
    "test_eng": "Invokes the Test Engineer agent to run tests and analyze coverage.",
    "test_gen": "Invokes the Test Generator agent to write unit and table-driven tests.",
    "go_test": "Invokes the Go Testing agent to run tests and debug test failures.",
    "review": "Invokes the Code Review agent to verify production readiness and standards.",
    "security": "Invokes the Security agent to audit code and scan for vulnerabilities.",
    "ux": "Invokes the UX Design agent to refine page layout and flow.",
    "quality_gate": "Invokes the Quality Gate agent to check for over-engineering and complexity.",
    "service_engineer": "Invokes the Service Engineer agent to modify existing Go codebases.",
    "planner": "Invokes the Planner agent to create execution plans.",
    "innovation_scout": (
        "Invokes the Innovation Scout Agent (ADR-0001 Platform Intelligence Layer) to scan for adoptable "
        "techniques/technologies and draft ADR proposals via the ADR store, pending human/CTO approval. "
        "Returns a JSON InnovationScoutReport (proposals, narrative)."
    ),
    "observer": (
        "Invokes the Observer Agent (ADR-0001 Platform Intelligence Layer) to scan recent platform_metrics for "
        "regressions, flaky agents and latency drift, and persist any findings to platform_findings. "
        "Returns a JSON ObserverReport (findings, narrative)."
    ),
    "product_strategist": (
        "Invokes the Product Strategist Agent (ADR-0001 Platform Intelligence Layer) to mine requirement history "
        "and open platform_findings for customer-facing product/capability proposals. Read-only: never writes "
        "code or tickets. Returns a JSON ProductStrategistReport (proposals, narrative)."
    ),
    "critic": (
        "Invokes the Critic agent to check another agent's claims against the evidence from its run. "
        "Returns a JSON CriticReport (claims with evidence, unsupported_count, passed, feedback)."
    ),
    "api_contract": (
        "Invokes the API Contract agent to find drift between OpenAPI/proto specs, Go handlers and TypeScript types. "
        "Returns a JSON ContractReport (drifts with file:line, passed, feedback)."
    ),
    "observability_engineer": (
        "Invokes the Observability Engineer agent to review Go services for OTel spans, RED metrics, slog logging "
        "and health probes. Returns an Artifact with Prometheus rule files and docs/observability-review.md."
    ),
    "finops": (
        "Invokes the FinOps agent to account for LLM and cloud spend. "
        "Returns a JSON CostReport (cost lines, total_usd, budget_usd, over_budget, recommendations)."
    ),
    "release_manager": (
        "Invokes the Release Manager agent to write the CHANGELOG fragment, version-bump note and PR body "
        "for a finished change. Returns an Artifact with those three files."
    ),
    "crypto_oracle": (
        "Invokes the Crypto Oracle Agent to analyze crypto markets and trends. "
        "Returns a JSON Forecast (scenarios, sources, previous-forecast accountability)."
    ),
    "market_shark": (
        "Invokes the Market Shark Agent to analyze stock markets and investments. "
        "Returns a JSON MarketAnalysis (metrics, bull/bear case, sources)."
    ),
    "price_analyst": (
        "Invokes the Price Analyst Agent to analyze and compare pricing. "
        "Returns a JSON PurchaseAdvice (buy-now / wait scenarios, sources)."
    ),
}


# Appended to every run_<agent> tool description so MCP clients know how to parse the result.
RESULT_FORMAT_NOTE = (
    "\n\nArgs: request (the task); project (optional: the seeded project name whose ADRs, conventions and "
    "architecture notes should be retrieved as context; empty searches every project)."
    "\n\nReturns: plain text for agents that answer in text; otherwise JSON (a structured result such as an "
    "Artifact is a JSON object, a list of results is a JSON array)."
)


def _register_agent_mcp_tools():
    def make_runner(agent_name: str):
        async def agent_runner(request: str, project: str = "") -> str:
            return await run_agent_by_name(agent_name, request, project=project)

        return agent_runner

    for name in registry.list_agents():
        tool_name = f"run_{name}"
        runner = make_runner(name)
        docstring = AGENT_DOCSTRINGS.get(name, f"Invokes the {name} agent to perform its designated task.")
        runner.__doc__ = docstring + RESULT_FORMAT_NOTE
        runner.__name__ = tool_name
        mcp.tool(name=tool_name)(runner)


_register_agent_mcp_tools()

# ── SDLC / Existing Code Tools


@mcp.tool()
def search_adrs(query: str) -> str:
    """Search existing Architecture Decision Records (ADRs) to ensure new advice doesn't contradict past rulings."""
    with logfire.span("mcp.search_adrs", query_length=len(query), _level="info"):
        results = vector_db.search("adrs", query)
        if not results:
            return f"No ADRs found for query: {query}"
        return "Found past architectural context:\n" + "\n---\n".join(results)


@mcp.tool()
def save_adr(decision_id: str, content: str) -> str:
    """Save a new Architecture Decision Record (ADR) to the knowledge base."""
    with logfire.span("mcp.save_adr", decision_id=decision_id, _level="info"):
        vector_db.add_documents("adrs", [content], [{"id": decision_id}], [decision_id])
        return f"Saved ADR {decision_id} successfully."


def _list_capabilities() -> str:
    """Return the live list of capabilities → agent owners, sorted alphabetically.

    Used to keep the delegate_to_orchestration_agent docstring honest at runtime.
    """
    from pydantic_agents.agents.registration import register_all_agents
    from pydantic_agents.core.registry import registry

    register_all_agents()
    lines = []
    for cap, owners in registry.capabilities():
        primary = owners[0]
        rest = f"  (also: {', '.join(owners[1:])})" if len(owners) > 1 else ""
        lines.append(f"  - {cap}: {primary}{rest}")
    return "\n".join(lines)


@mcp.tool()
async def delegate_to_orchestration_agent(
    capability: str,
    context: str,
    task: str,
    owner: str | None = None,
) -> str:
    """
    Delegate a heavy-lifting engineering task to the Orgitration autonomous agents.

    Pass a capability string registered via pydantic_agents.agents.registration; common
    ones include 'backend', 'frontend', 'database', 'security', 'review',
    'quality_gate', 'go_test', 'planning', 'sre', 'docs'. The full live list is
    available via the list_orchestration_capabilities tool — call that first if you
    are not sure which capability fits the task.

    For modifying existing code, prefer the 'modify_existing_code' tool.

    If the capability is ambiguous (claimed by multiple agents), pass
    owner=<agent_name> to disambiguate.
    """
    from pydantic_agents.agents.registration import register_all_agents
    from pydantic_agents.core.exceptions import AgentExecutionError, AgentNotFoundError, AmbiguousCapabilityError
    from pydantic_agents.core.registry import instrumented_query

    register_all_agents()

    full_prompt = (
        f"Context from IDE:\n<ide_context>\n{context}\n</ide_context>\n\nTask:\n<user_task>\n{task}\n</user_task>"
    )

    try:
        return await instrumented_query(
            capability=capability,
            prompt=full_prompt,
            span_name="mcp.delegate_to_agent",
            attributes={"capability": capability, "context_length": len(context), "task_length": len(task)},
            owner=owner,
        )
    except AgentNotFoundError as e:
        return f"ERROR: {e}\n\nAvailable capabilities:\n{_list_capabilities()}"
    except AmbiguousCapabilityError as e:
        owners_str = ", ".join(e.owners)
        return f"ERROR: {e}\n\nRe-call delegate_to_orchestration_agent with owner=<agent_name> from: {owners_str}"
    except AgentExecutionError as e:
        return f"ERROR: {e}"


@mcp.tool()
def list_orchestration_capabilities() -> str:
    """List the Orgitration agent capabilities currently registered, with their owning agent."""
    return "Registered capabilities (capability: primary_agent):\n" + _list_capabilities()


@mcp.tool()
async def run_orchestration_pipeline(
    project_name: str, requirement: str, blueprint: str = "new_feature", framework: str = ""
) -> str:
    """Run the entire multi-agent Orgitration pipeline end-to-end for a given project requirement.

    Because the pipeline can take 10+ minutes, this tool submits the pipeline
    as a background task and returns immediately with a run_id. Poll for
    completion using check_pipeline_status.

    blueprint: one of the catalogue in core/pipeline/config.py (new_feature, quick_backend, langgraph_full,
    crew_sdlc, crew_qa, crew_architecture_review, crew_audit, ...). framework: legacy, langgraph, crewai, autogen
    or composite; empty uses the blueprint's own framework.
    """
    from pydantic_agents.core.pipeline.config import BLUEPRINTS
    from pydantic_agents.core.pipeline_jobs import submit_pipeline_run

    if error := _check_project_name(project_name):
        return f"ERROR: {error}"
    if blueprint not in BLUEPRINTS:
        return f"ERROR: Unknown blueprint {blueprint!r}. Available: {', '.join(BLUEPRINTS)}"
    run_id = await submit_pipeline_run(project_name, requirement, blueprint=blueprint, framework=framework or None)
    return (
        f"Pipeline run '{run_id}' submitted for project '{project_name}' in the background.\n"
        f"Poll for completion with: check_pipeline_status(run_id='{run_id}')"
    )


@mcp.tool()
async def check_pipeline_status(run_id: str) -> str:
    """Check the status of a background pipeline run.

    Returns current status (pending/running/completed/failed/interrupted) and any result. A run cut off by a
    server restart shows as interrupted; continue it with resume_pipeline_run.
    """
    from pydantic_agents.core.pipeline_jobs import get_run

    run = await get_run(run_id)
    if not run:
        return f"No pipeline run found with id '{run_id}'."

    line = f"Run ID: {run_id}\nStatus: {run['status']}\n"
    if run.get("result"):
        line += f"Result: {run['result']}\n"
    if run.get("error"):
        line += f"Error: {run['error']}\n"
    if run.get("completed_at"):
        from datetime import datetime

        dt = datetime.fromisoformat(run["completed_at"])
        line += f"Completed at: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        elapsed = run.get("elapsed_seconds", 0)
        line += f"Elapsed: {elapsed:.1f}s\n"
    return line


@mcp.tool()
async def resume_pipeline_run(run_id: str) -> str:
    """Resume an interrupted, failed or cancelled pipeline run under the same run_id, from its checkpoints."""
    from pydantic_agents.core.pipeline_jobs import resume_pipeline_run as resume

    resumed = await resume(run_id)
    if resumed is None:
        return f"Run '{run_id}' is unknown, still running or already completed; nothing to resume."
    return f"Run '{resumed}' resumed in the background. Poll with check_pipeline_status(run_id='{resumed}')."


@mcp.tool()
async def approve_cto_gate(project_name: str, approved: bool = True, adr_id: str = "", adr_content: str = "") -> str:
    """Approve (or reject) the CTO gate of a project's pipeline run that is waiting for approval.

    The gate polls for a marker file inside the server; this writes it, so an MCP caller can approve a run
    without shell access to the pod.

    Approval requires ``adr_id`` (the same rule ``AgentAdvice.approved_to_proceed`` enforces in code for the
    CrewAI architecture-review gate): there is no approved decision without a recorded one. When ``adr_content``
    is also given, it is saved under ``adr_id`` via ``save_adr`` in the same call, so a human approving here does
    not need a second tool call to persist the decision record.
    """
    from pydantic_agents.core.cto_gate import write_approval

    if error := _check_project_name(project_name):
        return f"ERROR: {error}"
    if approved and not adr_id.strip():
        return "ERROR: approved=true requires adr_id (the decision record this approval is recorded under)."
    if adr_content.strip() and not adr_id.strip():
        return "ERROR: adr_content requires adr_id (nothing to save it under)."
    if adr_content.strip():
        try:
            save_adr(adr_id, adr_content)
        except Exception as e:
            logger.exception("save_adr failed for %r during approve_cto_gate", adr_id)
            return f"ERROR: save_adr failed ({e}); the CTO gate was not approved so it can be retried."
    path = write_approval(project_name, approved)
    suffix = f", ADR {adr_id} saved" if adr_content.strip() else ""
    return f"CTO gate for '{project_name}' set to {'approved' if approved else 'rejected'} ({path.name}){suffix}."


@mcp.tool()
async def cancel_pipeline_run(run_id: str) -> str:
    """Cancel a running background pipeline run by its run_id.

    Sends a cancellation request to the background task.  If the task is
    already done (completed, failed, or already cancelled), reports its
    current status.
    """
    from pydantic_agents.core.pipeline_jobs import PIPELINE_RUNS

    run = PIPELINE_RUNS.get(run_id)
    if not run:
        return f"No pipeline run found with id '{run_id}'."

    task = run.get("_task")
    if task and not task.done():
        task.cancel()
        return f"Cancellation sent for run '{run_id}'."
    return f"Run '{run_id}' is already {run['status']}. No cancellation needed."


def _failed_dependencies(task: Any, results: dict[str, str]) -> list[str]:
    return [d for d in task.dependencies if results.get(d, "").startswith("ERROR")]


def _task_prompt(task: Any, results: dict[str, str]) -> str:
    """The task's prompt, extended with the results of the tasks it depends on."""
    if not task.dependencies:
        return task.prompt
    upstream = "\n\n".join(f"[{d}]\n{results.get(d, '')}" for d in task.dependencies)
    return f"{task.prompt}\n\nResults of the tasks this depends on:\n{upstream}"


async def _run_plan_task(task: Any, results: dict[str, str]) -> None:
    """Run one plan task into ``results``; a failure is recorded as an ``ERROR:`` result, never raised."""
    from pydantic_agents.core.registry import instrumented_query

    failed_deps = _failed_dependencies(task, results)
    if failed_deps:
        results[task.id] = f"ERROR: skipped, dependency failed: {', '.join(failed_deps)}"
        return
    try:
        results[task.id] = await instrumented_query(
            capability=task.capability,
            prompt=_task_prompt(task, results),
            span_name="mcp.plan_task",
            attributes={"task_id": task.id, "capability": task.capability},
        )
    except Exception as e:
        results[task.id] = f"ERROR: {e}"


async def _execute_plan(plan: Any) -> dict[str, str] | str:
    """Run the plan's tasks in dependency order; returns the results, or a message when the plan cannot progress."""
    completed: set[str] = set()
    results: dict[str, str] = {}
    while len(completed) < len(plan.tasks):
        batch = [t for t in plan.tasks if t.id not in completed and all(dep in completed for dep in t.dependencies)]
        if not batch:
            plan_json = plan.model_dump_json(indent=2)
            return f"Plan execution stuck — possible circular dependency.\nPlan: {plan_json}"
        await asyncio.gather(*[_run_plan_task(t, results) for t in batch])
        completed.update(t.id for t in batch)
    return results


def _plan_summary(goal: str, plan: Any, results: dict[str, str]) -> str:
    lines = [f"## Plan Execution: {goal}", ""]
    for t in plan.tasks:
        status = "✗" if results.get(t.id, "").startswith("ERROR") else "✓"
        lines.append(f"  {status} {t.id} ({t.capability}): {t.prompt[:80]}...")
        lines.append(f"     → {results.get(t.id, 'no result')[:200]}")
    return "\n".join(lines)


@mcp.tool()
async def plan_and_execute(goal: str) -> str:
    """Creates a plan to achieve a high-level goal and executes all tasks in dependency order.

    A task's prompt is extended with the results of the tasks it depends on, so dependencies carry data and not
    only order. A task whose dependency failed is skipped and reported as such.
    """
    with logfire.span("mcp.plan_and_execute", goal_length=len(goal), _level="info"):
        run_token = run_id_var.set(run_id_var.get() or f"plan_{uuid.uuid4().hex[:12]}")
        try:
            plan_response = await registry.run("planner", f"Create a plan to achieve this goal: {goal}")
            plan = plan_response.output
            results = await _execute_plan(plan)
        finally:
            run_id_var.reset(run_token)
        if isinstance(results, str):
            return results
        return _plan_summary(goal, plan, results)


@mcp.tool()
async def modify_existing_code(change_description: str, codebase_path: str = "") -> str:
    """
    Modify EXISTING Go backend code through a multi-agent pipeline.
    Use this when you have an existing Go service that needs changes — adding handlers,
    wiring dependencies, refactoring, fixing bugs, adding tracing, or extending features.

    This runs: service_engineer → test_engineer → deterministic verification (go build/vet/test on the changed
    tree) → quality_gate + review in parallel. Each gate reads the verification report.

    The changed files are written under generated-app/<codebase name>/ (they are not applied to the codebase);
    the result lists where.

    Args:
        change_description: What to change and why (e.g., "Add OpenTelemetry tracing
            to the ProcessMessage handler in internal/service/message.go")
        codebase_path: Path to the existing codebase root. Leave empty to auto-detect.
    """
    with logfire.span(
        "mcp.modify_existing_code",
        change_description_length=len(change_description),
        codebase_path=codebase_path or "auto-detect",
        _level="info",
    ):
        from pydantic_agents.agents.registration import register_all_agents
        from pydantic_agents.core.existing_code_handler import format_result_summary, run_existing_code_pipeline

        register_all_agents()

        result = await run_existing_code_pipeline(
            change_description=change_description,
            codebase_path=codebase_path,
            run_tests=True,
            run_quality_gate=True,
            run_review=True,
        )

        return format_result_summary(result)


@mcp.tool()
async def scaffold_and_build_service(service_name: str, bounded_context: str, requirement: str = "") -> str:
    """
    Scaffold and build a new backend service from scratch using the full Orgitration pipeline.
    This runs the BA → Advisory → Implementation → Test → SRE → Review pipeline.
    """
    with logfire.span(
        "mcp.scaffold_service",
        service_name=service_name,
        bounded_context=bounded_context,
        _level="info",
    ):
        from pydantic_agents.core.pipeline.config import PipelineConfig
        from pydantic_agents.core.pipeline.orchestrator import OrgitrationPipeline

        requirement = requirement or f"Build a {service_name} service for {bounded_context} bounded context."
        config = PipelineConfig(
            project=service_name,
            step="all",
            resume=False,
            auto_yes=True,
            requirement=requirement,
        )
        pipeline = OrgitrationPipeline(config)
        await pipeline.execute()
        return f"Service '{service_name}' scaffolding pipeline completed. Artifacts in generated-app/{service_name}/."


# ── Vector DB Tools ────────────────────────────────


def _split_oversized(para: str, max_chars: int) -> tuple[list[str], str]:
    """Hard-split a paragraph on whitespace: the full-size pieces and the remainder (at most ``max_chars``)."""
    pieces: list[str] = []
    while len(para) > max_chars:
        cut = para.rfind(" ", 0, max_chars)
        cut = cut if cut > 0 else max_chars
        pieces.append(para[:cut].strip())
        para = para[cut:].strip()
    return pieces, para


def _text_units(text: str, max_chars: int) -> Iterator[tuple[bool, str]]:
    """The (standalone, text) units of a document. Pieces of an oversized paragraph stand alone; every other
    paragraph can be packed together with its neighbours."""
    for raw in text.split("\n\n"):
        para = raw.strip()
        if not para:
            continue
        pieces, para = _split_oversized(para, max_chars)
        for piece in pieces:
            yield True, piece
        yield False, para


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Split text into chunks of at most ``max_chars`` on paragraph boundaries.

    The embedding model only sees roughly the first 256 word pieces (~1 KB), so
    whole long files would be indexed by their opening section alone.
    """
    chunks: list[str] = []
    current = ""
    for standalone, unit in _text_units(text, max_chars):
        if standalone:  # an oversized paragraph's pieces close the chunk in progress
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit)
        elif current and len(current) + len(unit) + 2 > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


_AGENTS_DIR = ".agents"
_SEED_SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "site-packages",
        "vendor",
        "dist",
        "build",
        "__pycache__",
        ".tox",
        ".terraform",
        ".worktrees",
    }
)


def _seed_patterns(root: str) -> list[str]:
    join = os.path.join
    return [
        join(root, _AGENTS_DIR, "rules", "*.md"),
        join(root, _AGENTS_DIR, "workflows", "*.md"),
        join(root, _AGENTS_DIR, "rules", "**", "*.md"),
        join(root, _AGENTS_DIR, "workflows", "**", "*.md"),
        join(root, "**", "ARCHITECTURE*.md"),
        join(root, "**", "CONVENTIONS*.md"),
        join(root, "**", "CODING_STANDARDS*.md"),
        join(root, "**", "CONTRIBUTING*.md"),
        join(root, "**", "*.adr"),
        join(root, "*.md"),
    ]


def _seed_domain(rel: str, filename: str) -> str:
    """The knowledge domain a context file belongs to, from its lower-cased relative path."""
    if "rule" in rel or "convention" in rel:
        return "conventions"
    if "adr" in rel or "architecture" in rel:
        return "adrs"
    if "workflow" in rel or "pipeline" in rel or "readme" in filename.lower():
        return "project"
    return "architecture"


def _seed_files(root: str) -> list[tuple[str, str]]:
    """The (path, relative path) of every context file under ``root``.

    Vendored directories are skipped, and so is anything that resolves outside ``root`` (a symlink must not pull
    in files from elsewhere on the machine).
    """
    real_root = Path(root).resolve()
    found: list[tuple[str, str]] = []
    for pattern in _seed_patterns(root):
        for fp in glob.glob(pattern, recursive=True):
            rel = os.path.relpath(fp, root)
            if not os.path.isfile(fp) or _SEED_SKIP_DIRS.intersection(rel.split(os.sep)):
                continue
            if Path(fp).resolve().is_relative_to(real_root):
                found.append((fp, rel))
    return found


def _index_seed_file(fp: str, rel: str, project: str) -> bool:
    """Chunk one file into the vector DB; False when it is empty."""
    with open(fp, encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return False
    rel = rel.lower()
    domain = _seed_domain(rel, os.path.basename(fp))
    sanitized = rel.replace("/", "::").replace(chr(92), "::")
    chunks = _chunk_text(content)
    vector_db.add_documents(
        domain,
        chunks,
        [{"source": fp, "domain": domain, "file": rel, "project": project, "chunk": i} for i in range(len(chunks))],
        [f"{project}:file:{sanitized}#{i}" for i in range(len(chunks))],
    )
    return True


def seed_project_files(project_path: str, project: str = "") -> str:
    """Index a project's context files into the vector DB. No path restriction: for trusted callers such as
    ``scripts/seed_projects.py``. The MCP tool ``seed_project_context`` checks the path first."""
    project = project or os.path.basename(os.path.normpath(project_path))
    files_indexed = 0
    errors: list[str] = []
    for fp, rel in _seed_files(project_path):
        try:
            files_indexed += _index_seed_file(fp, rel, project)
        except Exception as e:
            errors.append(f"{fp}: {e}")

    summary = f"Indexed {files_indexed} files into vector DB"
    if files_indexed > 0:
        summary += " across domains: conventions, adrs, project, architecture"
    if errors:
        summary += f"\nErrors: {len(errors)}"
        for err in errors[:5]:
            summary += f"\n  - {err}"
    return summary


def _allowed_seed_roots() -> list[Path]:
    """Directories the MCP seeding tool may read: ``SEED_ROOTS`` (os.pathsep-separated) or the output directory."""
    configured = [r for r in settings.seed_roots.split(os.pathsep) if r.strip()]
    return [Path(r).expanduser().resolve() for r in configured or [str(settings.output_dir)]]


def _check_seed_root(project_path: str) -> Path:
    """The resolved project path, if it sits inside an allowed root; raises PathTraversalError otherwise."""
    candidate = Path(project_path).expanduser().resolve()
    if not any(candidate.is_relative_to(root) for root in _allowed_seed_roots()):
        raise PathTraversalError("project_path is outside the directories this server may read (set SEED_ROOTS)")
    return candidate


@mcp.tool()
def seed_project_context(project_path: str, project: str = "") -> str:
    """Seed the vector DB with project files (rules, workflows, READMEs, ADRs).

    Scans the given project path for relevant context files (.md, .adr, rules/) and
    indexes them into the vector DB so agent context retrieval can find them. The path must be
    inside the server's allowed directories (``SEED_ROOTS``, default the output directory).

    Args:
        project_path: Absolute path to the project root (e.g., /Users/jocham/programming/projects/mediusfy/platform)
        project: Project name used to namespace document ids and tag metadata so several
                 repos can share the store without overwriting each other's README.md etc.
                 Defaults to the basename of project_path.
    """
    try:
        root = _check_seed_root(project_path)
    except PathTraversalError as e:
        return f"Error: {e}"
    return seed_project_files(str(root), project or os.path.basename(os.path.normpath(project_path)))


@mcp.tool()
def query_context(query: str, domain: str = "", top_k: int = 5, project: str = "") -> str:
    """Query the vector DB directly for relevant context about any topic.

    Searches all knowledge domains (or a specific one) and returns
    the most relevant snippets.

    Args:
        query: What to search for (e.g., "favorites system", "settings panel")
        domain: Optional domain filter (adrs, architecture, conventions, project).
                Empty string searches all domains.
        top_k: Max results per domain (default 5)
        project: Optional project filter (e.g. "openclaw", "modulex", "infrastructure"), matching the
                 name the documents were seeded under. Empty string searches all projects.
    """
    domains = [domain] if domain else ["adrs", "architecture", "conventions", "project"]
    all_results = []

    for d in domains:
        with logfire.span("mcp.query_context", domain=d, query_length=len(query), _level="info"):
            try:
                results = vector_db.search(d, query, n_results=top_k, project=project or None)
                if results:
                    all_results.append(f"\n=== Domain: {d} ===")
                    for i, r in enumerate(results, 1):
                        snippet = r[:300] + "..." if len(r) > 300 else r
                        all_results.append(f"[{i}] {snippet}")
            except Exception as e:
                all_results.append(f"[Domain {d}] Error: {e}")

    if not all_results:
        return f"No context found for: {query}"

    return "\n".join(all_results)


# ── Cross-Runtime Memory Bridge Tools ─────────────────────────────────────


@mcp.tool()
def sync_memory(all_runtimes: bool = True, max_entries: int = 50) -> str:
    """Sync memory across all agent runtimes (pgvector ↔ Claude Code ↔ Antigravity).

    First pushes recent pgvector activity logs into Claude Code's agent-memory
    files and Antigravity's implicit memory. Then pulls Claude Code's memory
    back into pgvector for discovery by Orgitration agents.

    Args:
        all_runtimes: If True, sync to both Claude Code and Antigravity.
                      If False, only sync pgvector → Claude Code.
        max_entries: Max recent pgvector entries to process.
    """
    with logfire.span("mcp.sync_memory", all_runtimes=all_runtimes, _level="info"):
        synced_count = sync_memory_to_claude(max_entries=max_entries)
        pulled_count = sync_claude_to_pgvector()

    return (
        f"Memory sync complete:\n"
        f"  pgvector → Claude Code: {synced_count} entries\n"
        f"  Claude Code → pgvector: {pulled_count} entries"
    )


@mcp.tool()
def memory_bridge_status() -> str:
    """Get the current status of all memory systems (pgvector, Claude Code, Antigravity).

    Reports how many memory files each Claude Code agent has, how many
    Antigravity implicit entries exist, and pgvector table health.
    """
    status = get_bridge_status()
    return (
        f"Memory Bridge Status:\n"
        f"  Claude Code agents: {len(status.get('claude_agent_memory', {}))}\n"
        f"  Claude memory files: {status.get('claude_agent_memory', {})}\n"
        f"  Antigravity implicit entries: {status.get('antigravity_implicit_count', 0)}\n"
        f"  pgvector tables active: {list(status.get('pgvector_tables', {}).keys())}\n"
        f"  Runtime: Orgitration ↔ Claude Code ↔ Antigravity"
    )


if __name__ == "__main__":
    mcp.run()
