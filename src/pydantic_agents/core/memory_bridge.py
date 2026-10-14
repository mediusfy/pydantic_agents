"""Cross-runtime memory bridge: Orgitration pgvector ↔ Claude Code ↔ Antigravity.

Writes context entries from pgvector activity_logs into Claude Code agent-memory
directories and Antigravity implicit memory, then pulls Claude's memory back
into pgvector for a bidirectional sync.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

from pydantic_agents.core.vector_db import vector_db

logger = logging.getLogger(__name__)

CLAUDE_AGENTS_DIR = Path.home() / ".claude" / "agents"
ANTIGRAVITY_IMPLICIT_DIR = Path.home() / ".gemini" / "antigravity" / "implicit"
ANTIGRAVITY_ANNOTATIONS_DIR = Path.home() / ".gemini" / "antigravity" / "annotations"


def _get_claude_agents() -> list[str]:
    """List Claude Code agents that have a config directory."""
    if not CLAUDE_AGENTS_DIR.exists():
        return []
    return [d.name for d in CLAUDE_AGENTS_DIR.iterdir() if d.is_dir()]


def _claude_agent_memory_dir(agent_name: str) -> Path:
    """Path to a Claude Code agent's memory directory."""
    return CLAUDE_AGENTS_DIR / agent_name / "agent-memory"


def claude_memory_count() -> dict[str, int]:
    """Count memory files per Claude Code agent."""
    agents = _get_claude_agents()
    counts: dict[str, int] = {}
    for agent in sorted(agents):
        mem_dir = _claude_agent_memory_dir(agent)
        if mem_dir.exists():
            counts[agent] = len(list(mem_dir.iterdir()))
        else:
            counts[agent] = 0
    return counts


def sync_memory_to_claude(max_entries: int = 50) -> int:
    """Push recent pgvector activity entries to Claude Code agent-memory dirs.

    Returns number of entries pushed.
    """
    agents = _get_claude_agents()
    if not agents:
        logger.warning("No Claude Code agents found at %s", CLAUDE_AGENTS_DIR)
        return 0

    try:
        results = vector_db.search("activity_logs", "", n_results=max_entries)
    except Exception:
        logger.warning("Vector DB activity_logs search failed — nothing to sync")
        return 0

    if not results:
        return 0

    entries = [doc for doc in results if doc.strip()]

    pushed = 0
    for agent in agents:
        mem_dir = _claude_agent_memory_dir(agent)
        mem_dir.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(entries):
            sig = hashlib.sha256(entry.encode()).hexdigest()[:8]
            filepath = mem_dir / f"orchestration_{i:04d}_{sig}.md"
            if filepath.exists():
                continue
            filepath.write_text(f"# Orgitration Activity\n\n{entry}\n")
            pushed += 1

    if pushed:
        logger.info("Pushed %d memory entries to %d Claude Code agent(s)", pushed, len(agents))
    else:
        logger.info("No new memory entries to push to Claude Code agents")

    # Also write to Antigravity implicit memory
    if ANTIGRAVITY_IMPLICIT_DIR.exists():
        ANTIGRAVITY_IMPLICIT_DIR.mkdir(parents=True, exist_ok=True)
    ag_pushed = 0
    for _, entry in enumerate(entries):
        sig = hashlib.sha256(entry.encode()).hexdigest()[:8]
        filepath = ANTIGRAVITY_IMPLICIT_DIR / f"orchestration_activity_{sig}.md"
        if filepath.exists():
            continue
        filepath.write_text(f"# Orgitration Activity\n\n{entry}\n")
        ag_pushed += 1
    if ag_pushed:
        logger.info("Pushed %d entries to Antigravity implicit memory", ag_pushed)

    return pushed


def sync_claude_to_pgvector(_max_entries: int = 50) -> int:
    """Pull Claude Code agent-memory files back into pgvector.

    Returns number of documents added.
    """
    agents = _get_claude_agents()
    if not agents:
        return 0

    added = 0
    for agent in agents:
        mem_dir = _claude_agent_memory_dir(agent)
        if not mem_dir.exists():
            continue
        for fpath in sorted(mem_dir.iterdir()):
            if not fpath.is_file() or fpath.suffix not in (".md", ".txt"):
                continue
            content = fpath.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
            try:
                vector_db.add_documents(
                    "activity_logs",
                    [content],
                    ids=[f"claude_{agent}_{fpath.stem}"],
                    metadatas=[{"source": "claude_code", "agent": agent, "file": fpath.name}],
                )
                added += 1
            except Exception as e:
                logger.warning("Failed to add %s to pgvector: %s", fpath, e)
    if added:
        logger.info("Pulled %d entries from Claude Code agents into pgvector", added)
    return added


def get_bridge_status() -> dict[str, Any]:
    """Get the current status of all memory systems."""
    activity_results = vector_db.search("activity_logs", "", n_results=100)
    activity_count = len(activity_results) if activity_results else 0
    claude_agent_memory = claude_memory_count()
    antigravity_implicit = len(list(ANTIGRAVITY_IMPLICIT_DIR.glob("*.md"))) if ANTIGRAVITY_IMPLICIT_DIR.exists() else 0
    antigravity_annotations = (
        len(list(ANTIGRAVITY_ANNOTATIONS_DIR.glob("*"))) if ANTIGRAVITY_ANNOTATIONS_DIR.exists() else 0
    )
    return {
        "pgvector_tables": {
            "activity_logs": {"count": activity_count},
            "knowledge_base": {"count": "available"},
            "project": {"available": True},
        },
        "claude_agent_memory": claude_agent_memory,
        "antigravity_implicit_count": antigravity_implicit,
        "antigravity_annotations_count": antigravity_annotations,
    }
