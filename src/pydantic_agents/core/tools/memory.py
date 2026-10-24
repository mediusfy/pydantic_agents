import datetime
import json
import logging
import os
from typing import Any

from pydantic_ai import RunContext

from pydantic_agents.core.exceptions import OrgitrationDependencyError, OrgitrationError
from pydantic_agents.core.models.core import ActivityLog
from pydantic_agents.core.session_recorder import current_run_context
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import safe_path_join
from pydantic_agents.core.vector_db import vector_db

logger = logging.getLogger(__name__)


def _store_errors() -> tuple[type[BaseException], ...]:
    """Errors that mean the memory store is unavailable (as opposed to a bug in the caller, which must stay loud).

    OSError covers connection and timeout failures, OrgitrationError/OrgitrationDependencyError the typed errors
    raised by this package (a missing vector extra), and asyncpg's own hierarchy (which is not OSError) is added
    when the vector extra is installed.
    """
    errors: tuple[type[BaseException], ...] = (OSError, OrgitrationError, OrgitrationDependencyError)
    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        return errors
    return (*errors, asyncpg.PostgresError, asyncpg.InterfaceError)


def _degrade_message(tool: str, exc: BaseException) -> str:
    """Memory is best-effort: an unreachable store must not abort the agent run that called the tool.

    The full error is logged; the model only gets the exception type, which avoids echoing connection details.
    """
    logger.warning("tool.%s: memory store unavailable: %s", tool, exc)
    return f"Memory store unavailable ({type(exc).__name__}); continuing without it."


def _scope(ctx: RunContext[Any]) -> tuple[str, str | None]:
    """The (agent_name, run_id) that activity is attributed to."""
    run_ctx = current_run_context.get() or {}
    agent_name = getattr(ctx.deps, "agent_name", None) or run_ctx.get("agent_name") or "system"
    return agent_name, run_ctx.get("run_id") or None


async def log_activity(ctx: RunContext[Any], action: str, summary: str, artifact_ids: list[str] | None = None) -> str:
    """Logs a significant agent activity to the project memory for future spot-checking.

    Best-effort: if the memory store is unreachable this returns a message saying the activity was not logged
    instead of raising, so the calling agent's run continues. Programming errors still raise.
    """
    pass_num = getattr(ctx.deps, "pass_num", 0) if ctx.deps else 0
    agent_name, run_id = _scope(ctx)

    log_entry = ActivityLog(
        pass_num=pass_num,
        agent_name=agent_name,
        action=action,
        summary=summary,
        timestamp=datetime.datetime.now().isoformat(),
        artifact_ids=artifact_ids or [],
    )

    doc_id = f"log_{pass_num}_{agent_name}_{datetime.datetime.now().timestamp()}"
    narrative_doc = f"During pass {pass_num}, the {agent_name} agent performed an action: {action}. Summary: {summary}"
    metadata = {
        "json_data": log_entry.model_dump_json(),
        "type": "activity",
        "agent_name": agent_name,
        "pass_num": str(pass_num),
    }
    if run_id:
        metadata["run_id"] = run_id

    try:
        await vector_db.async_add_documents("activity_logs", [narrative_doc], [metadata], [doc_id])
    except _store_errors() as e:
        return f"Activity not logged. {_degrade_message('log_activity', e)}"
    return f"Activity logged successfully: {action}"


async def query_history(
    ctx: RunContext[Any],
    query: str,
    pass_num: int | None = None,
    all_agents: bool = False,
    include_legacy: bool = False,
) -> str:
    """Queries this agent's history for the current run to find past decisions or actions.

    Best-effort: if the memory store is unreachable this returns a message saying no history is available instead
    of raising. Programming errors still raise.

    Args:
        query: What to look for.
        pass_num: Only return entries logged in this pass.
        all_agents: Also include entries logged by other agents in the same run.
        include_legacy: Also include older entries that were logged before entries were tagged with agent and
            run. These cannot be attributed, so they come from any agent and any run.
    """
    agent_name, run_id = _scope(ctx)
    filters: dict[str, str | None] = {}
    if not all_agents:
        filters["agent_name"] = agent_name
    if run_id:
        filters["run_id"] = run_id
    if pass_num is not None:
        filters["pass_num"] = str(pass_num)

    try:
        results = await vector_db.async_search("activity_logs", query, n_results=5, filters=filters)
        if include_legacy:
            legacy = await vector_db.async_search(
                "activity_logs", query, n_results=5, filters={"type": "activity", "agent_name": None}
            )
            results = results + [r for r in legacy if r not in results]
    except _store_errors() as e:
        return f"No history available. {_degrade_message('query_history', e)}"
    if not results:
        return "No relevant history found."

    formatted_logs = []
    for res in results:
        try:
            log = json.loads(res)
            formatted_logs.append(f"Pass {log['pass_num']} | {log['agent_name']} | {log['action']}: {log['summary']}")
        except Exception:
            formatted_logs.append(res)

    return "Relevant Project History:\n" + "\n".join(formatted_logs)


async def read_and_memorize_url(ctx: RunContext[Any], url: str) -> str:
    """Reads a large webpage, splits it into semantic chunks, and saves it directly to memory.

    Args:
        url: The HTTP or HTTPS URL to fetch. Do NOT pass local file paths or file:// URLs. Use read_file for local files.
    """
    if url.startswith("file://") or not (url.startswith("http://") or url.startswith("https://")):
        return f"Error: Invalid URL '{url}'. This tool requires an http:// or https:// URL. For local files, use the read_file tool instead."

    try:
        return await vector_db.async_ingest_url("knowledge_base", url)
    except Exception as e:
        return f"Failed to ingest URL: {e}"


async def read_and_memorize_pdf(ctx: RunContext[Any], file_path: str) -> str:
    """Reads a PDF document, splits it into semantic chunks, and saves it directly to memory."""
    try:
        safe_path_join(os.getcwd(), file_path)
        return await vector_db.async_ingest_pdf("knowledge_base", file_path)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Failed to ingest PDF: {e}"


register_tool("log_activity", log_activity, "Logs activity to project memory", ["memory"])
register_tool("query_history", query_history, "Queries project history", ["memory"])
register_tool(
    "read_and_memorize_url",
    read_and_memorize_url,
    "Scrapes and memorizes large webpages",
    ["memory", "web", "research", "news"],
)
register_tool(
    "read_and_memorize_pdf",
    read_and_memorize_pdf,
    "Reads and memorizes PDF documents",
    ["memory", "research", "documentation"],
)
