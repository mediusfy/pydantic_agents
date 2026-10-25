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


def show_status(entries: list[dict]):  # noqa: T201 — CLI display tool
    if not entries:
        print("No telemetry data yet.")
        return

    now = entries[-1]["timestamp"]
    start = entries[0]["timestamp"]

    # Windows: last 2 min and last 10 min
    two_m = now - 120
    ten_m = now - 600

    def since(t: float) -> str:
        d = now - t
        if d < 120:
            return f"{d:.0f}s ago"
        return f"{d / 60:.0f}m ago"

    # ── Header ──
    # ── Header ──
    print(f"╔═══ PIPELINE TRACE ── {datetime.fromtimestamp(now).strftime('%H:%M:%S')}  ({since(start)} run) ═══╗")  # noqa: T201
    print(f"║ Total agent runs: {len(entries):>8}         ")  # noqa: T201
    total_success = sum(1 for e in entries if e["success"])
    total_pct = total_success / len(entries) * 100 if entries else 0
    print(f"║ Overall success: {total_success}/{len(entries)} ({total_pct:.0f}%)  ")  # noqa: T201

    # ── Last 10 min ──
    r10 = [e for e in entries if e["timestamp"] >= ten_m]
    r2 = [e for e in entries if e["timestamp"] >= two_m]

    print(f"╠═══ Last 10 min: {len(r10)} runs  |  Last 2 min: {len(r2)} runs ═══╣")  # noqa: T201

    if r10:
        # Agent breakdown
        agent_counts = Counter(e["agent"] for e in r10)
        failures_10 = [e for e in r10 if not e["success"]]
        fail_agents = Counter(e["agent"] for e in failures_10)

        # Duration stats
        agent_durs: dict[str, list[float]] = defaultdict(list)
        for e in r10:
            agent_durs[e["agent"]].append(e["duration_sec"])

        print(f"║ {'Agent':<22s} {'Runs':>5s} {'OK':>4s} {'Fail':>4s} {'Avg dur':>7s}")  # noqa: T201
        print(f"║ {'─' * 22} {'─' * 5} {'─' * 4} {'─' * 4} {'─' * 7}")  # noqa: T201
        for agent, count in agent_counts.most_common(12):
            succ = count - fail_agents.get(agent, 0)
            avg = sum(agent_durs[agent]) / len(agent_durs[agent])
            fail_count = fail_agents.get(agent, 0)
            print(f"║ {agent:<22s} {count:>5d} {succ:>4d} {fail_count:>4d} {avg:>5.0f}s")  # noqa: T201

    # ── Recent failures ──
    recent_failures = [e for e in entries if not e["success"]][-5:]
    if recent_failures:
        print("╠═══ Recent failures ═══╣")  # noqa: T201
        for f in recent_failures:
            dt = datetime.fromtimestamp(f["timestamp"]).strftime("%H:%M:%S")
            print(f"║ ✗ {f['agent']:<22s} at {dt} ({f['duration_sec']:.0f}s)")  # noqa: T201

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
