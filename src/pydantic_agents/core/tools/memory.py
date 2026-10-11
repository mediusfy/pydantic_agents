import datetime
import json
import os
from typing import Any

from pydantic_ai import RunContext

from pydantic_agents.core.models.core import ActivityLog
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import safe_path_join
from pydantic_agents.core.vector_db import vector_db


async def log_activity(ctx: RunContext[Any], action: str, summary: str, artifact_ids: list[str] | None = None) -> str:
    """Logs a significant agent activity to the project memory for future spot-checking."""
    pass_num = getattr(ctx.deps, "pass_num", 0) if ctx.deps else 0
    agent_name = getattr(ctx.deps, "agent_name", "system") if ctx.deps else "system"

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

    await vector_db.async_add_documents(
        "activity_logs", [narrative_doc], [{"json_data": log_entry.model_dump_json(), "type": "activity"}], [doc_id]
    )
    return f"Activity logged successfully: {action}"


async def query_history(ctx: RunContext[Any], query: str, pass_num: int | None = None) -> str:
    """Queries the project history to find past decisions or actions."""
    results = await vector_db.async_search("activity_logs", query, n_results=5)
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
