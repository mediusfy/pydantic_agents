"""Context Engine — vector-indexed context retrieval for agent execution.

FORTIFIED CONTEXT MANAGEMENT:
Instead of relying on agents knowing which files to read, the context engine
retrieves relevant snippets from an indexed knowledge base before each agent
execution. This ensures agents are fed exactly the relevant architecture rules,
past decisions, and dependencies without manual file hunting.

Knowledge domains:
  - adrs: Architecture Decision Records (past decisions)
  - architecture: Architecture rules and patterns
  - conventions: Coding conventions and standards
  - project: Project-specific context

Retrieval is asynchronous (the embedding and the database query run off the event loop) and scoped to one
project when the caller names one; without a project every seeded project is searched.
"""

import logging
import uuid
from typing import Any

import logfire

from pydantic_agents.core.vector_db import vector_db

logger = logging.getLogger(__name__)

# Mapping: agent_name → knowledge domains to query for that agent
AGENT_CONTEXT_DOMAINS: dict[str, list[str]] = {
    "backend": ["adrs", "architecture", "conventions"],
    "frontend": ["adrs", "architecture"],
    "database": ["adrs", "architecture"],
    "service_engineer": ["adrs", "architecture", "project"],
    "security": ["adrs", "architecture"],
    "review": ["adrs", "architecture", "conventions"],
    "quality_gate": ["adrs", "architecture", "conventions"],
    "sre": ["adrs", "architecture"],
    "devops": ["adrs", "architecture"],
    "test_eng": ["conventions", "architecture"],
    "test_gen": ["conventions"],
    "docs": ["conventions"],
    "ux": ["architecture"],
    "cto": ["adrs", "architecture"],
    "ba": ["architecture", "project"],
    "planner": ["adrs", "architecture", "project"],
    "critic": ["conventions"],
    "api_contract": ["architecture", "conventions"],
    "observability_engineer": ["adrs", "architecture", "conventions"],
    "finops": ["project"],
    "release_manager": ["conventions", "project"],
    # Travel agents get architecture-only (generic)
    "holiday_planner": ["project"],
    "region_analyzer": ["project"],
    "asymmetric_reviewer": ["project"],
}

# Longest query the embedding model can use: MiniLM reads about 256 word pieces, so a longer query is judged
# by its opening alone. Callers should pass the requirement or story titles, never an artifact dump.
MAX_QUERY_CHARS = 1_000

_UNAVAILABLE_MARKER = "Vector DB not available"


def _usable(snippet: Any) -> bool:
    """A snippet worth putting in a prompt: non-empty text that is not the store's own unavailability notice."""
    return isinstance(snippet, str) and bool(snippet.strip()) and _UNAVAILABLE_MARKER not in snippet


def _interleave(per_domain: list[list[str]]) -> list[str]:
    """Round-robin over the domains' ranked snippets: the best of each domain first. Concatenating them let the
    first domain fill the whole ``top_k`` and silently discarded the other domains' results."""
    out: list[str] = []
    for rank in range(max((len(snippets) for snippets in per_domain), default=0)):
        out.extend(snippets[rank] for snippets in per_domain if rank < len(snippets))
    return out


def _unique_snippets(snippets: list[str]) -> list[str]:
    """Drop snippets whose first 100 characters were already seen, keeping the first of each."""
    seen: set[str] = set()
    unique: list[str] = []
    for snippet in snippets:
        key = snippet[:100]
        if key not in seen:
            seen.add(key)
            unique.append(snippet)
    return unique


async def retrieve_context(agent_name: str, request: str, top_k: int = 3, project: str | None = None) -> str:
    """Retrieve relevant context snippets from all knowledge domains for a given agent.

    Args:
        agent_name: The agent requesting context.
        request: The prompt/request text to match against (only its first MAX_QUERY_CHARS are used).
        top_k: Max results per domain, and in total.
        project: Restrict to documents seeded under this project name; None searches every project.

    Returns:
        Formatted context string, or empty string if nothing relevant found.
    """
    domains = AGENT_CONTEXT_DOMAINS.get(agent_name, ["architecture", "project"])
    with logfire.span("context_engine.retrieve", domains=domains, agent_name=agent_name, _level="debug"):
        # One embedding of the request, the domains searched concurrently.
        outcomes = await vector_db.async_search_domains(domains, request[:MAX_QUERY_CHARS], top_k, project)
    per_domain: list[list[str]] = []
    for domain, outcome in zip(domains, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            # Warning, not debug: a dimension mismatch or a down database otherwise just means "no context".
            logger.warning("Context retrieval failed for domain '%s': %s", domain, outcome)
            continue
        per_domain.append([snippet for snippet in outcome or [] if _usable(snippet)])
    all_snippets = _interleave(per_domain)

    if not all_snippets:
        return ""

    formatted = "\n\n---\n\n".join(_unique_snippets(all_snippets)[:top_k])
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
