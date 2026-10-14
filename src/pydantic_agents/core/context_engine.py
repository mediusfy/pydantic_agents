"""Context Engine — vector-indexed context retrieval for agent execution.

FORTIFIED CONTEXT MANAGEMENT:
Instead of relying on agents knowing which files to read, the context engine
retrieves relevant snippets from an indexed knowledge base before each agent
execution. This ensures agents are fed exactly the relevant architecture rules,
past decisions, and dependencies without manual file hunting.

Knowledge domains:
  - adrs: Architecture Decision Records (past decisions)
  - architecture: Architecture rules and patterns
  - dependencies: External dependency configurations
  - conventions: Coding conventions and standards
  - project: Project-specific context

The engine uses the existing vector_db module for storage and retrieval.
"""

import logging
import uuid
from typing import Any

import logfire

from pydantic_agents.core.vector_db import vector_db

logger = logging.getLogger(__name__)

# Mapping: agent_name → knowledge domains to query for that agent
AGENT_CONTEXT_DOMAINS: dict[str, list[str]] = {
    "backend": ["adrs", "architecture", "dependencies", "conventions"],
    "frontend": ["adrs", "architecture", "dependencies"],
    "database": ["adrs", "architecture"],
    "service_engineer": ["adrs", "architecture", "dependencies", "project"],
    "security": ["adrs", "architecture"],
    "review": ["adrs", "architecture", "conventions"],
    "quality_gate": ["adrs", "architecture", "conventions"],
    "sre": ["adrs", "architecture", "dependencies"],
    "devops": ["adrs", "dependencies"],
    "test_eng": ["conventions", "architecture"],
    "test_gen": ["conventions"],
    "docs": ["conventions"],
    "ux": ["architecture"],
    "cto": ["adrs", "architecture"],
    "ba": ["architecture", "project"],
    "planner": ["adrs", "architecture", "project"],
    # Travel agents get architecture-only (generic)
    "holiday_planner": ["project"],
    "region_analyzer": ["project"],
    "asymmetric_reviewer": ["project"],
}


def retrieve_context(agent_name: str, request: str, top_k: int = 3) -> str:
    """Retrieve relevant context snippets from all knowledge domains for a given agent.

    Args:
        agent_name: The agent requesting context.
        request: The prompt/request text to match against.
        top_k: Max results per domain.

    Returns:
        Formatted context string, or empty string if nothing relevant found.
    """
    domains = AGENT_CONTEXT_DOMAINS.get(agent_name, ["architecture", "project"])
    all_snippets: list[str] = []

    for domain in domains:
        with logfire.span(
            "context_engine.retrieve",
            domain=domain,
            agent_name=agent_name,
            _level="debug",
        ):
            try:
                results = vector_db.search(domain, request, n_results=top_k)
                if results:
                    for snippet in results:
                        if isinstance(snippet, str) and snippet.strip():
                            all_snippets.append(snippet)
            except Exception as e:
                logger.debug("Context retrieval failed for domain '%s': %s", domain, e)

    if not all_snippets:
        return ""

    # Deduplicate and format
    seen: set[str] = set()
    unique_snippets: list[str] = []
    for s in all_snippets:
        key = s[:100]
        if key not in seen:
            seen.add(key)
            unique_snippets.append(s)

    formatted = "\n\n---\n\n".join(unique_snippets[:top_k])
    return f"\n{formatted}\n"


def store_artifact_context(
    domain: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    doc_id: str | None = None,
) -> None:
    """Store an artifact (ADR, decision, convention) into the vector knowledge base.

    This should be called after each pipeline stage so downstream agents
    benefit from the latest context.

    Args:
        domain: Knowledge domain (adrs, architecture, conventions, etc.)
        content: The artifact content to index.
        metadata: Optional metadata (agent_name, timestamp, etc.)
        doc_id: Optional document ID for deduplication.
    """
    with logfire.span(
        "context_engine.store",
        domain=domain,
        doc_id=doc_id or "auto",
        content_length=len(content),
        _level="info",
    ):
        try:
            vector_db.add_documents(
                domain,
                [content],
                metadatas=[metadata or {}],
                ids=[doc_id] if doc_id else [f"auto_{uuid.uuid4().hex[:12]}"],
            )
            logger.info("Stored %d chars in domain '%s'", len(content), domain)
        except Exception as e:
            logger.warning("Failed to store context in '%s': %s", domain, e)
