"""The pgvector <-> Claude Code / Antigravity memory sync, against temporary directories and a fake vector DB."""

import hashlib
from unittest.mock import MagicMock

import pytest

from pydantic_agents.core import memory_bridge


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Point the module's directories at tmp_path; the Claude agents dir and the Antigravity dir both exist."""
    claude = tmp_path / "claude" / "agents"
    implicit = tmp_path / "gemini" / "implicit"
    claude.mkdir(parents=True)
    implicit.mkdir(parents=True)
    monkeypatch.setattr(memory_bridge, "CLAUDE_AGENTS_DIR", claude)
    monkeypatch.setattr(memory_bridge, "ANTIGRAVITY_IMPLICIT_DIR", implicit)
    return claude, implicit


@pytest.fixture
def db(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(memory_bridge, "vector_db", fake)
    return fake


def _sig(entry: str) -> str:
    return hashlib.sha256(entry.encode()).hexdigest()[:8]


def _agent(claude, name):
    (claude / name).mkdir()
    return claude / name / "agent-memory"


# --- sync_memory_to_claude --------------------------------------------------------------------------------------


def test_nothing_is_pushed_without_claude_agents(dirs, db):
    assert memory_bridge.sync_memory_to_claude() == 0
    db.search.assert_not_called()


def test_a_failing_search_pushes_nothing(dirs, db):
    _agent(dirs[0], "ba")
    db.search.side_effect = RuntimeError("db down")

    assert memory_bridge.sync_memory_to_claude() == 0


def test_no_results_pushes_nothing(dirs, db):
    _agent(dirs[0], "ba")
    db.search.return_value = []

    assert memory_bridge.sync_memory_to_claude() == 0


def test_entries_are_written_for_every_agent_and_to_antigravity(dirs, db):
    claude, implicit = dirs
    ba, cto = _agent(claude, "ba"), _agent(claude, "cto")
    db.search.return_value = ["first entry", "  ", "second entry"]

    pushed = memory_bridge.sync_memory_to_claude(max_entries=7)

    db.search.assert_called_once_with("activity_logs", "", n_results=7)
    assert pushed == 4  # two non-blank entries x two agents
    for mem_dir in (ba, cto):
        files = sorted(p.name for p in mem_dir.iterdir())
        assert files == [
            f"orchestration_0000_{_sig('first entry')}.md",
            f"orchestration_0001_{_sig('second entry')}.md",
        ]
        assert (mem_dir / files[0]).read_text() == "# Orgitration Activity\n\nfirst entry\n"
    assert sorted(p.name for p in implicit.iterdir()) == sorted(
        [f"orchestration_activity_{_sig('first entry')}.md", f"orchestration_activity_{_sig('second entry')}.md"]
    )


def test_a_second_sync_writes_nothing_new(dirs, db):
    _agent(dirs[0], "ba")
    db.search.return_value = ["only entry"]

    assert memory_bridge.sync_memory_to_claude() == 1
    assert memory_bridge.sync_memory_to_claude() == 0


def test_identical_entries_share_one_antigravity_file(dirs, db):
    _agent(dirs[0], "ba")
    db.search.return_value = ["same", "same"]

    assert memory_bridge.sync_memory_to_claude() == 2  # the Claude files are numbered, so both are written
    assert len(list(dirs[1].iterdir())) == 1


def test_a_missing_antigravity_dir_is_skipped_not_fatal(dirs, db):
    claude, implicit = dirs
    ba = _agent(claude, "ba")
    implicit.rmdir()
    db.search.return_value = ["entry"]

    assert memory_bridge.sync_memory_to_claude() == 1
    assert len(list(ba.iterdir())) == 1
    assert not implicit.exists()


# --- sync_claude_to_pgvector ------------------------------------------------------------------------------------


def test_nothing_is_pulled_without_claude_agents(dirs, db):
    assert memory_bridge.sync_claude_to_pgvector() == 0
    db.add_documents.assert_not_called()


def test_memory_files_are_added_with_stable_ids_and_metadata(dirs, db):
    mem = _agent(dirs[0], "ba")
    mem.mkdir()
    (mem / "note.md").write_text("  remember this  ")
    (mem / "todo.txt").write_text("a todo")

    assert memory_bridge.sync_claude_to_pgvector() == 2

    calls = {c.kwargs["ids"][0]: c for c in db.add_documents.call_args_list}
    assert set(calls) == {"claude_ba_note", "claude_ba_todo"}
    note = calls["claude_ba_note"]
    assert note.args == ("activity_logs", ["remember this"])
    assert note.kwargs["metadatas"] == [{"source": "claude_code", "agent": "ba", "file": "note.md"}]


def test_other_files_empty_files_directories_and_missing_memory_dirs_are_skipped(dirs, db):
    claude = dirs[0]
    mem = _agent(claude, "ba")
    mem.mkdir()
    (mem / "image.png").write_text("not text")
    (mem / "empty.md").write_text("   \n")
    (mem / "nested.md").mkdir()
    _agent(claude, "cto")  # no agent-memory directory at all

    assert memory_bridge.sync_claude_to_pgvector() == 0
    db.add_documents.assert_not_called()


def test_a_failing_add_is_skipped_and_the_rest_continue(dirs, db):
    mem = _agent(dirs[0], "ba")
    mem.mkdir()
    (mem / "a.md").write_text("first")
    (mem / "b.md").write_text("second")
    db.add_documents.side_effect = [RuntimeError("db down"), None]

    assert memory_bridge.sync_claude_to_pgvector() == 1
    assert db.add_documents.call_count == 2
