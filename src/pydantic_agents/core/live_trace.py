"""Live pipeline trace — tail the telemetry file for real-time status.

Usage:
    while true; do clear; python3 -m pydantic_agents.core.live_trace; sleep 5; done
"""

import contextlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.utils import safe_path_join


def resolve_trace_path(arg: str) -> Path:
    """The telemetry file to read: a ``.jsonl`` file inside the current directory.

    Raises PathTraversalError if ``arg`` points outside it and ValueError if it is not a ``.jsonl`` file.
    """
    path = safe_path_join(Path.cwd(), arg)
    if path.suffix != ".jsonl":
        raise ValueError(f"{arg!r} is not a .jsonl file")
    return path


def load_entries(filepath: str = "telemetry.jsonl") -> list[dict]:
    entries: list[dict] = []
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    with contextlib.suppress(json.JSONDecodeError):
                        entries.append(json.loads(line))
    except FileNotFoundError:
        pass
    return entries


def _clock(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def _age(seconds: float) -> str:
    return f"{seconds:.0f}s ago" if seconds < 120 else f"{seconds / 60:.0f}m ago"


def _print_header(entries: list[dict]) -> None:
    now = entries[-1]["timestamp"]
    age = _age(now - entries[0]["timestamp"])
    total_success = sum(1 for e in entries if e["success"])
    total_pct = total_success / len(entries) * 100
    print(f"╔═══ PIPELINE TRACE ── {_clock(now)}  ({age} run) ═══╗")  # noqa: T201 — CLI display tool
    print(f"║ Total agent runs: {len(entries):>8}         ")  # noqa: T201
    print(f"║ Overall success: {total_success}/{len(entries)} ({total_pct:.0f}%)  ")  # noqa: T201


def _print_agent_table(recent: list[dict]) -> None:
    """Runs, successes, failures and average duration for the busiest agents in the window."""
    agent_counts = Counter(e["agent"] for e in recent)
    fail_agents = Counter(e["agent"] for e in recent if not e["success"])
    agent_durs: dict[str, list[float]] = defaultdict(list)
    for e in recent:
        agent_durs[e["agent"]].append(e["duration_sec"])

    print(f"║ {'Agent':<22s} {'Runs':>5s} {'OK':>4s} {'Fail':>4s} {'Avg dur':>7s}")  # noqa: T201
    print(f"║ {'─' * 22} {'─' * 5} {'─' * 4} {'─' * 4} {'─' * 7}")  # noqa: T201
    for agent, count in agent_counts.most_common(12):
        fail_count = fail_agents.get(agent, 0)
        avg = sum(agent_durs[agent]) / len(agent_durs[agent])
        print(f"║ {agent:<22s} {count:>5d} {count - fail_count:>4d} {fail_count:>4d} {avg:>5.0f}s")  # noqa: T201


def _print_recent_failures(entries: list[dict]) -> None:
    recent_failures = [e for e in entries if not e["success"]][-5:]
    if not recent_failures:
        return
    print("╠═══ Recent failures ═══╣")  # noqa: T201
    for f in recent_failures:
        print(f"║ ✗ {f['agent']:<22s} at {_clock(f['timestamp'])} ({f['duration_sec']:.0f}s)")  # noqa: T201


def show_status(entries: list[dict]):  # noqa: T201 — CLI display tool
    if not entries:
        print("No telemetry data yet.")
        return

    now = entries[-1]["timestamp"]
    _print_header(entries)

    # Windows: last 10 min (and, for the count only, last 2 min), relative to the newest entry
    r10 = [e for e in entries if e["timestamp"] >= now - 600]
    r2 = [e for e in entries if e["timestamp"] >= now - 120]
    print(f"╠═══ Last 10 min: {len(r10)} runs  |  Last 2 min: {len(r2)} runs ═══╣")  # noqa: T201
    if r10:
        _print_agent_table(r10)

    _print_recent_failures(entries)
    print(f"╚═══ {'═' * 50} ═══╝")  # noqa: T201


def main():
    try:
        path = resolve_trace_path(sys.argv[1] if len(sys.argv) > 1 else "telemetry.jsonl")
    except (PathTraversalError, ValueError) as e:
        print(f"live_trace: {e}", file=sys.stderr)  # noqa: T201 — CLI display tool
        sys.exit(2)
    entries = load_entries(str(path))
    show_status(entries)


if __name__ == "__main__":
    main()
