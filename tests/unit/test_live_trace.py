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
