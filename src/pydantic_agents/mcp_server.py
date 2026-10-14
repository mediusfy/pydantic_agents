import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import contextlib
import time
from typing import Any

import logfire
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic_ai import UsageLimits

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.context_engine import retrieve_context
from pydantic_agents.core.event_broker import Event, EventKind, event_broker
from pydantic_agents.core.exceptions import AgentExecutionError, AgentNotFoundError
from pydantic_agents.core.memory_bridge import get_bridge_status, sync_claude_to_pgvector, sync_memory_to_claude
from pydantic_agents.core.observability import configure_observability
from pydantic_agents.core.registry import registry
from pydantic_agents.core.telemetry import telemetry
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


async def run_agent_by_name(agent_name: str, request: str) -> str:
    """Run a registered agent by its name, resolving dependencies and tracking telemetry.

    FORTIFIED CONTEXT MANAGEMENT:
    Before running the agent, this function retrieves relevant snippets from the
    vector-indexed context store (ADRs, architecture rules, past decisions) and
    prepends them to the request. This ensures agents are fed exactly the relevant
    context without needing to manually hunt through the file system.
    """
    try:
        agent = registry.get(agent_name)
    except ValueError as e:
        raise AgentNotFoundError(f"Agent '{agent_name}' not registered") from e

    agent_tuple = registry._agents.get(agent_name)
    if not agent_tuple:
        raise AgentNotFoundError(f"Agent '{agent_name}' not registered")

    _, _, deps_factory, _ = agent_tuple

    # ── Context Enrichment ────────────────────────────────────────────────
    # Pull relevant snippets from the vector-indexed knowledge base
    enriched_request = request
    with logfire.span("mcp.retrieve_context", agent_name=agent_name, _level="info"):
        context = retrieve_context(agent_name, request, top_k=3)
        if context:
            enriched_request = f"Relevant Architecture Context from Knowledge Base:\n{context}\n\n---\n\n{request}"
    # ──────────────────────────────────────────────────────────────────────

    span_attrs: dict[str, Any] = {
        "agent_name": agent_name,
        "request_length": len(request),
        "context_retrieved": bool(context),
    }

    start = time.time()
    success = False
    try:
        with logfire.span("mcp.run_agent", **span_attrs, _level="info"):
            resolved_deps = deps_factory() if callable(deps_factory) else {}
            result = await agent.run(enriched_request, deps=resolved_deps, usage_limits=UsageLimits(request_limit=500))
            success = True
            if hasattr(result, "data"):
                return str(result.data)
            return str(result.output) if hasattr(result, "output") else str(result)
    except Exception as e:
        logger.warning("Agent '%s' failed: %s", agent_name, e)
        raise AgentExecutionError(f"Agent '{agent_name}' failed: {e}", is_retryable=False) from e
    finally:
        duration = time.time() - start
        with contextlib.suppress(Exception):
            await telemetry.log_call(agent_name, duration, success, metadata=span_attrs)


AGENT_DOCSTRINGS = {
    "holiday_planner": "Invokes the Holiday Planner Agent.",
    "region_analyzer": "Invokes the Region Analyzer Agent.",
    "asymmetric_reviewer": "Invokes the Asymmetric Reviewer Agent (law & corporate tricks for travel savings).",
    "current_events": "Invokes the Current Events Agent (history buff).",
    "weather": "Invokes the Weather Agent.",
    "accountant": "Invokes the Accountant Agent.",
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
    "crypto_oracle": "Invokes the Crypto Oracle Agent to analyze crypto markets and trends.",
    "market_shark": "Invokes the Market Shark Agent to analyze stock markets and investments.",
    "price_analyst": "Invokes the Price Analyst Agent to analyze and compare pricing.",
}


def _register_agent_mcp_tools():
    def make_runner(agent_name: str):
        async def agent_runner(request: str) -> str:
            return await run_agent_by_name(agent_name, request)

        return agent_runner

    for name in registry.list_agents():
        tool_name = f"run_{name}"
        runner = make_runner(name)
        docstring = AGENT_DOCSTRINGS.get(name, f"Invokes the {name} agent to perform its designated task.")
        runner.__doc__ = docstring
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
async def run_orchestration_pipeline(project_name: str, requirement: str) -> str:
    """Run the entire multi-agent Orgitration pipeline end-to-end for a given project requirement.

    Because the pipeline can take 10+ minutes, this tool submits the pipeline
    as a background task and returns immediately with a run_id. Poll for
    completion using check_pipeline_status.
    """
    from pydantic_agents.core.pipeline_jobs import submit_pipeline_run

    run_id = await submit_pipeline_run(project_name, requirement)
    return (
        f"Pipeline run '{run_id}' submitted for project '{project_name}' in the background.\n"
        f"Poll for completion with: check_pipeline_status(run_id='{run_id}')"
    )


@mcp.tool()
def check_pipeline_status(run_id: str) -> str:
    """Check the status of a background pipeline run.

    Returns current status (pending/running/completed/failed) and any result.
    """
    from pydantic_agents.core.pipeline_jobs import PIPELINE_RUNS

    run = PIPELINE_RUNS.get(run_id)
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


@mcp.tool()
async def plan_and_execute(goal: str) -> str:
    """Creates a plan to achieve a high-level goal and executes all tasks in dependency order."""
    with logfire.span("mcp.plan_and_execute", goal_length=len(goal), _level="info"):
        from pydantic_agents.agents.planner_agent import planner_agent
        from pydantic_agents.core.registry import instrumented_query

        plan_response = await planner_agent.run(
            f"Create a plan to achieve this goal: {goal}",
            usage_limits=UsageLimits(request_limit=500),
        )
        plan = plan_response.output

        completed: set[str] = set()
        results: dict[str, str] = {}

        while len(completed) < len(plan.tasks):
            batch = [t for t in plan.tasks if t.id not in completed and all(dep in completed for dep in t.dependencies)]
            if not batch:
                return f"Plan execution stuck — possible circular dependency.\nPlan: {plan.model_dump_json(indent=2)}"

            async def _run_task(t):
                try:
                    result = await instrumented_query(
                        capability=t.capability,
                        prompt=t.prompt,
                        span_name="mcp.plan_task",
                        attributes={"task_id": t.id, "capability": t.capability},
                    )
                    results[t.id] = result
                except Exception as e:
                    results[t.id] = f"ERROR: {e}"

            await asyncio.gather(*[_run_task(t) for t in batch])

            for t in batch:
                completed.add(t.id)

        summary_lines = [f"## Plan Execution: {goal}", ""]
        for t in plan.tasks:
            status = "✓" if "ERROR" not in results.get(t.id, "") else "✗"
            summary_lines.append(f"  {status} {t.id} ({t.capability}): {t.prompt[:80]}...")
            summary_lines.append(f"     → {results.get(t.id, 'no result')[:200]}")

        return "\n".join(summary_lines)


@mcp.tool()
async def modify_existing_code(change_description: str, codebase_path: str = "") -> str:
    """
    Modify EXISTING Go backend code through a multi-agent pipeline.
    Use this when you have an existing Go service that needs changes — adding handlers,
    wiring dependencies, refactoring, fixing bugs, adding tracing, or extending features.

    This runs: service_engineer → test_engineer → quality_gate → review
    Each agent validates the previous one's output.

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


# ── Event-Driven Async Handoff Tools ──────────────────────────────────────


@mcp.tool()
async def submit_async_task(
    task_type: str,
    payload: str,
    callback_tool: str | None = None,
) -> str:
    """Submit a long-running task to the background queue.

    task_type: One of "build", "test_suite", "deploy", "pipeline"
    payload: JSON payload with task parameters
    callback_tool: Optional tool name to call on completion

    Returns:
        task_id — use with check_task_status to poll for results.
    """
    with logfire.span(
        "mcp.submit_async_task",
        task_type=task_type,
        payload_length=len(payload),
        _level="info",
    ):
        task_id = await event_broker.submit(
            Event(
                kind=EventKind.TASK_SUBMITTED,
                data={
                    "type": task_type,
                    "payload": payload,
                    "callback_tool": callback_tool,
                },
            )
        )
        return f"Task '{task_id}' submitted. Use check_task_status('{task_id}') to poll for results."


@mcp.tool()
async def check_task_status(task_id: str) -> str:
    """Check the status of an async task submitted via submit_async_task."""
    with logfire.span("mcp.check_task_status", task_id=task_id, _level="info"):
        status = await event_broker.get_status(task_id)
        if not status:
            return f"Task '{task_id}' not found."
        return f"Task: {task_id}\nStatus: {status.get('status', 'unknown')}\nResult: {status.get('result', 'pending')}"


@mcp.tool()
async def wait_for_task(task_id: str, timeout_seconds: int = 300) -> str:
    """Wait synchronously for an async task to complete (up to timeout_seconds)."""
    with logfire.span("mcp.wait_for_task", task_id=task_id, timeout=timeout_seconds, _level="info"):
        import asyncio

        result = await asyncio.wait_for(event_broker.wait_for(task_id, timeout_seconds), timeout_seconds + 10)
        if result is None:
            return f"Task '{task_id}' timed out after {timeout_seconds}s."
        return f"Task: {task_id}\nStatus: {result.get('status', 'completed')}\nResult: {result.get('result', '')}"


# ── Vector DB Tools for Async Task Context ────────────────────────────────


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Split text into chunks of at most ``max_chars`` on paragraph boundaries.

    The embedding model only sees roughly the first 256 word pieces (~1 KB), so
    whole long files would be indexed by their opening section alone.
    """
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        while len(para) > max_chars:  # oversized paragraph: hard-split on whitespace
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


@mcp.tool()
def seed_project_context(project_path: str, project: str = "") -> str:
    """Seed the vector DB with project files (rules, workflows, READMEs, ADRs).

    Scans the given project path for relevant context files (.md, .adr, rules/) and
    indexes them into the vector DB so agent context retrieval can find them.

    Args:
        project_path: Absolute path to the project root (e.g., /Users/jocham/programming/projects/mediusfy/platform)
        project: Project name used to namespace document ids and tag metadata so several
                 repos can share the store without overwriting each other's README.md etc.
                 Defaults to the basename of project_path.
    """
    import glob
    import os

    project = project or os.path.basename(os.path.normpath(project_path))
    skip_dirs = {
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

    files_indexed = 0
    errors = []

    # Patterns to scan
    patterns = [
        os.path.join(project_path, ".agents", "rules", "*.md"),
        os.path.join(project_path, ".agents", "workflows", "*.md"),
        os.path.join(project_path, ".agents", "rules", "**", "*.md"),
        os.path.join(project_path, ".agents", "workflows", "**", "*.md"),
        os.path.join(project_path, "**", "ARCHITECTURE*.md"),
        os.path.join(project_path, "**", "CONVENTIONS*.md"),
        os.path.join(project_path, "**", "CODING_STANDARDS*.md"),
        os.path.join(project_path, "**", "CONTRIBUTING*.md"),
        os.path.join(project_path, "**", "*.adr"),
        os.path.join(project_path, "*.md"),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for fp in matches:
            if not os.path.isfile(fp):
                continue
            if skip_dirs.intersection(os.path.relpath(fp, project_path).split(os.sep)):
                continue
            try:
                with open(fp, encoding="utf-8") as f:
                    content = f.read()
                if not content.strip():
                    continue

                # Determine domain from path
                rel = os.path.relpath(fp, project_path).lower()
                if "rule" in rel or "convention" in rel:
                    domain = "conventions"
                elif "adr" in rel or "architecture" in rel:
                    domain = "adrs"
                elif "workflow" in rel or "pipeline" in rel or "readme" in os.path.basename(fp).lower():
                    domain = "project"
                else:
                    domain = "architecture"

                sanitized = rel.replace("/", "::").replace(chr(92), "::")
                chunks = _chunk_text(content)
                vector_db.add_documents(
                    domain,
                    chunks,
                    [
                        {"source": fp, "domain": domain, "file": rel, "project": project, "chunk": i}
                        for i in range(len(chunks))
                    ],
                    [f"{project}:file:{sanitized}#{i}" for i in range(len(chunks))],
                )
                files_indexed += 1
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


@mcp.tool()
def task_context(task_id: str) -> str:
    """Get full context from the vector DB about a specific completed async task.

    Searches the project domain for context related to a given task_id.

    Args:
        task_id: The task ID to look up context for
    """
    results = vector_db.search("project", task_id, n_results=3)
    if not results:
        return f"No context found for task: {task_id}"
    return "\n---\n".join(results[:3])


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
