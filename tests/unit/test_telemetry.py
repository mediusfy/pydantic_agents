import json

import pytest

from pydantic_agents.core.telemetry import Telemetry


async def test_log_call_writes_jsonl(tmp_path):
    log_file = tmp_path / "test.jsonl"
    t = Telemetry(str(log_file))
    await t.log_call("test-agent", 1.5, True, {"key": "value"})
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["agent"] == "test-agent"
    assert entry["duration_sec"] == 1.5
    assert entry["success"] is True
    assert entry["metadata"]["key"] == "value"


async def test_log_call_rounds_duration(tmp_path):
    log_file = tmp_path / "round.jsonl"
    t = Telemetry(str(log_file))
    await t.log_call("fast", 0.12345, True)
    entry = json.loads(log_file.read_text().strip())
    assert entry["duration_sec"] == 0.12345


async def test_multiple_logs(tmp_path):
    log_file = tmp_path / "multi.jsonl"
    t = Telemetry(str(log_file))
    await t.log_call("a", 0.1, True)
    await t.log_call("b", 0.2, False)
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2


async def test_track_success(tmp_path):
    log_file = tmp_path / "track.jsonl"
    t = Telemetry(str(log_file))

    async def dummy():
        return 42

    result = await t.track("test", dummy)
    assert result == 42
    entry = json.loads(log_file.read_text().strip())
    assert entry["agent"] == "test"
    assert entry["success"] is True


async def test_track_failure_re_raises(tmp_path):
    log_file = tmp_path / "fail.jsonl"
    t = Telemetry(str(log_file))

    async def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await t.track("test", failing)

    entry = json.loads(log_file.read_text().strip())
    assert entry["success"] is False
