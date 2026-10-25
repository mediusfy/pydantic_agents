"""The agentapi server loop: single-command mode and the interactive stdin loop."""

import io
import json
import sys
from unittest.mock import AsyncMock

import pytest

from pydantic_agents.core.agy_bridge import server


@pytest.fixture(autouse=True)
def quiet_startup(monkeypatch):
    monkeypatch.setattr("pydantic_agents.agents.registration.register_all_agents", lambda: None)


def _stdout_lines(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


async def test_single_command_mode_answers_once_and_echoes_the_id(monkeypatch, capsys):
    monkeypatch.setattr(server, "_read_command_async", AsyncMock(return_value='{"command": "ping", "id": "7"}'))
    handle = AsyncMock(return_value={"response": "pong"})
    monkeypatch.setattr(server, "_handle_command", handle)

    await server.run_server_loop()

    handle.assert_awaited_once_with({"command": "ping", "id": "7"})
    assert _stdout_lines(capsys) == [{"response": "pong", "id": "7"}]


async def test_single_command_without_an_id_gets_no_id(monkeypatch, capsys):
    monkeypatch.setattr(server, "_read_command_async", AsyncMock(return_value='{"command": "ping"}'))
    monkeypatch.setattr(server, "_handle_command", AsyncMock(return_value={"response": "pong"}))

    await server.run_server_loop()

    assert _stdout_lines(capsys) == [{"response": "pong"}]


async def test_single_command_with_invalid_json_reports_an_error(monkeypatch, capsys):
    monkeypatch.setattr(server, "_read_command_async", AsyncMock(return_value="{nope"))
    handle = AsyncMock()
    monkeypatch.setattr(server, "_handle_command", handle)

    await server.run_server_loop()

    [line] = _stdout_lines(capsys)
    assert line["error"].startswith("invalid JSON:")
    handle.assert_not_awaited()


async def test_single_command_mode_lets_handler_errors_propagate(monkeypatch):
    monkeypatch.setattr(server, "_read_command_async", AsyncMock(return_value='{"command": "x"}'))
    monkeypatch.setattr(server, "_handle_command", AsyncMock(side_effect=RuntimeError("bad")))

    with pytest.raises(RuntimeError, match="bad"):
        await server.run_server_loop()


async def test_interactive_loop_handles_each_line_until_eof(monkeypatch, capsys):
    lines = '{"command": "a", "id": 1}\n\n   \nnot json\n{"command": "boom"}\n{"command": "b"}\n'
    monkeypatch.setattr(sys, "stdin", io.StringIO(lines))
    monkeypatch.setattr(server, "_read_command_async", AsyncMock(return_value=None))

    async def handle(cmd):
        if cmd["command"] == "boom":
            raise RuntimeError("bad")
        return {"ok": cmd["command"]}

    monkeypatch.setattr(server, "_handle_command", handle)

    await server.run_server_loop()

    out = _stdout_lines(capsys)
    assert out[0] == {"ok": "a", "id": 1}
    assert out[1]["error"].startswith("invalid JSON:")
    assert out[2] == {"error": "internal error: bad"}
    assert out[3] == {"ok": "b"}
    assert len(out) == 4  # blank lines produce no output
