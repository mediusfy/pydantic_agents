"""Deterministic heuristics over ``platform_metrics`` rows, used by the ``observer`` agent's ``scan_for_findings``
tool (``agents/observer.py``) to decide what to persist as a ``platform_findings`` row.

Pure functions, no I/O: given the rows ``platform_metrics.recent_calls()`` already returned, decide what is worth
flagging. Kept separate from the agent/tool wiring so the detection logic is unit-testable without a database or an
LLM call.
"""

from dataclasses import dataclass
from itertools import groupby
from typing import Any

# A mixed window needs at least this many calls before "flaky" is worth raising (avoids one lucky retry = alarm).
_MIN_FLAKY_WINDOW = 3
# A latency/cost drift window needs at least this many calls, split in half, to compare "recent" vs "earlier".
_MIN_DRIFT_WINDOW = 4
# Newest-half average at least this many times the older-half average counts as drift.
_DRIFT_RATIO = 1.5


@dataclass
class DetectedFinding:
    kind: str
    severity: str
    agent: str | None
    summary: str
    evidence: dict[str, Any]


def _group_by_agent(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_agent: dict[str, list[dict[str, Any]]] = {}
    # recent_calls() returns newest first; keep that order within each agent's group.
    for agent, group in groupby(sorted(rows, key=lambda r: r["agent"]), key=lambda r: r["agent"]):
        by_agent.setdefault(agent, []).extend(group)
    return by_agent


def _failure_finding(agent: str, rows: list[dict[str, Any]]) -> DetectedFinding | None:
    successes = [r["success"] for r in rows]
    if len(successes) >= _MIN_FLAKY_WINDOW and any(successes) and not all(successes):
        failed = successes.count(False)
        return DetectedFinding(
            kind="flaky",
            severity="medium",
            agent=agent,
            summary=f"{agent}: {failed}/{len(successes)} of the last {len(successes)} calls failed (flaky)",
            evidence={"total": len(successes), "failed": failed},
        )
    if len(successes) >= 2 and all(not s for s in successes):
        return DetectedFinding(
            kind="regression",
            severity="high",
            agent=agent,
            summary=f"{agent}: all {len(successes)} of the last {len(successes)} calls failed",
            evidence={"total": len(successes), "failed": len(successes)},
        )
    return None


def _drift_finding(agent: str, rows: list[dict[str, Any]]) -> DetectedFinding | None:
    durations = [r["duration"] for r in rows if r.get("duration") is not None]
    if len(durations) < _MIN_DRIFT_WINDOW:
        return None
    mid = len(durations) // 2
    newer, older = durations[:mid], durations[mid:]
    older_avg = sum(older) / len(older)
    newer_avg = sum(newer) / len(newer)
    if older_avg > 0 and newer_avg >= older_avg * _DRIFT_RATIO:
        return DetectedFinding(
            kind="health",
            severity="medium",
            agent=agent,
            summary=f"{agent}: latency drift, {older_avg:.2f}s -> {newer_avg:.2f}s over {len(durations)} calls",
            evidence={"older_avg_seconds": older_avg, "newer_avg_seconds": newer_avg, "sample_size": len(durations)},
        )
    return None


def detect_findings(rows: list[dict[str, Any]]) -> list[DetectedFinding]:
    """Regressions, flaky agents, and latency drift, grouped by agent. Empty input yields no findings."""
    findings: list[DetectedFinding] = []
    for agent, group in _group_by_agent(rows).items():
        for finder in (_failure_finding, _drift_finding):
            found = finder(agent, group)
            if found is not None:
                findings.append(found)
    return findings
