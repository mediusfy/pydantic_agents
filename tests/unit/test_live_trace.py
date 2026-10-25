import json
import sys

import pytest

from pydantic_agents.core import live_trace
from pydantic_agents.core.exceptions import PathTraversalError


def test_a_jsonl_file_inside_the_working_directory_is_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    assert live_trace.resolve_trace_path("telemetry.jsonl") == tmp_path.resolve() / "telemetry.jsonl"
    assert live_trace.resolve_trace_path("logs/run.jsonl") == tmp_path.resolve() / "logs" / "run.jsonl"


@pytest.mark.parametrize("arg", ["../outside.jsonl", "/etc/passwd.jsonl", "logs/../../outside.jsonl", "/"])
def test_a_path_outside_the_working_directory_is_refused(tmp_path, monkeypatch, arg):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(PathTraversalError):
        live_trace.resolve_trace_path(arg)


@pytest.mark.parametrize("arg", ["notes.txt", "telemetry", "id_rsa", ".env"])
def test_only_jsonl_files_are_read(tmp_path, monkeypatch, arg):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="not a .jsonl file"):
        live_trace.resolve_trace_path(arg)


def test_main_refuses_a_bad_path_with_exit_code_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["live_trace", "../secret.jsonl"])

    with pytest.raises(SystemExit) as exit_info:
        live_trace.main()

    assert exit_info.value.code == 2
    assert "live_trace:" in capsys.readouterr().err


def test_main_shows_a_trace_inside_the_working_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    entry = {"agent": "ba", "success": True, "duration_sec": 1.5, "timestamp": 1_700_000_000.0}
    (tmp_path / "telemetry.jsonl").write_text(json.dumps(entry) + "\nnot json\n")
    monkeypatch.setattr(sys, "argv", ["live_trace"])

    live_trace.main()

    assert "ba" in capsys.readouterr().out


# --- show_status ------------------------------------------------------------------------------------------------

NOW = 1_700_000_000.0


def _entry(agent: str, ago: float, ok: bool = True, duration: float = 2.0) -> dict:
    return {"agent": agent, "success": ok, "duration_sec": duration, "timestamp": NOW - ago}


def _clock(ts: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _status(capsys, entries: list[dict]) -> list[str]:
    live_trace.show_status(entries)
    return capsys.readouterr().out.splitlines()


def test_no_entries_says_so(capsys):
    assert _status(capsys, []) == ["No telemetry data yet."]


def test_a_full_report_has_header_windows_table_failures_and_footer(capsys):
    entries = [
        _entry("old_agent", 3000),  # outside the 10-minute window, but in the totals
        _entry("ba", 500, duration=4.0),
        _entry("ba", 400, ok=False, duration=6.0),
        _entry("cto", 100, duration=1.0),
        _entry("ba", 60),
    ]

    assert _status(capsys, entries) == [
        f"╔═══ PIPELINE TRACE ── {_clock(NOW - 60)}  (49m ago run) ═══╗",
        "║ Total agent runs:        5         ",
        "║ Overall success: 4/5 (80%)  ",
        "╠═══ Last 10 min: 4 runs  |  Last 2 min: 2 runs ═══╣",
        "║ Agent                   Runs   OK Fail Avg dur",
        "║ ────────────────────── ───── ──── ──── ───────",
        "║ ba                         3    2    1     4s",
        "║ cto                        1    1    0     1s",
        "╠═══ Recent failures ═══╣",
        f"║ ✗ ba                     at {_clock(NOW - 400)} (6s)",
        f"╚═══ {'═' * 50} ═══╝",
    ]


def test_the_run_age_is_in_seconds_under_two_minutes_and_minutes_after(capsys):
    short = _status(capsys, [_entry("a", 90), _entry("a", 0)])[0]
    long = _status(capsys, [_entry("a", 600), _entry("a", 0)])[0]

    assert "(90s ago run)" in short
    assert "(10m ago run)" in long


def test_a_single_entry_reports_a_zero_second_run(capsys):
    lines = _status(capsys, [_entry("solo", 0, ok=False)])

    assert lines[0].endswith("(0s ago run) ═══╗")
    assert lines[2] == "║ Overall success: 0/1 (0%)  "
    assert "╠═══ Last 10 min: 1 runs  |  Last 2 min: 1 runs ═══╣" in lines


def test_the_table_lists_the_twelve_busiest_agents_most_active_first(capsys):
    entries = [_entry(f"agent{i:02d}", 10) for i in range(14)]
    entries += [_entry("busy", 10)] * 3 + [_entry("busy", 0)]

    lines = _status(capsys, entries)

    start = lines.index("║ ────────────────────── ───── ──── ──── ───────") + 1
    rows = [line for line in lines[start:] if not line.startswith(("╠", "╚"))]
    assert len(rows) == 12
    assert rows[0].split()[1] == "busy"
    assert rows[0].split()[2] == "4"


def test_only_the_five_most_recent_failures_are_listed_oldest_first(capsys):
    entries = [_entry(f"f{i}", 1000 - i * 10, ok=False) for i in range(7)] + [_entry("fine", 5)]

    lines = _status(capsys, entries)

    failures = [line for line in lines if line.startswith("║ ✗")]
    assert [line.split()[2] for line in failures] == ["f2", "f3", "f4", "f5", "f6"]


def test_no_failures_means_no_failure_section(capsys):
    assert "╠═══ Recent failures ═══╣" not in _status(capsys, [_entry("a", 5)])
