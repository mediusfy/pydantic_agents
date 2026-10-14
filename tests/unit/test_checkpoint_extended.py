import time
from pathlib import Path

import pytest

from pydantic_agents.core.adapters.state_repository import LocalDiskStateRepository
from pydantic_agents.core.checkpoint import CheckpointManager
from pydantic_agents.core.models.core import PipelineState


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
    repo = LocalDiskStateRepository(project_name="test_proj")
    return CheckpointManager(repository=repo, run_id="test_run")


@pytest.fixture
def sample_state():
    return PipelineState(run_id="test_run", stage="init", attempt=1)


class TestCheckpointSaveLoad:
    def test_round_trip(self, manager, sample_state):
        manager.save("stage1", sample_state)
        loaded = manager.load("stage1")
        assert loaded is not None
        assert loaded.run_id == "test_run"
        assert loaded.stage == "init"

    def test_load_nonexistent(self, manager):
        assert manager.load("does_not_exist") is None

    def test_load_corrupt_json(self, manager):
        filepath = Path(manager.repository._get_filepath(manager.run_id, "corrupt"))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("{invalid_json: true}")
        assert manager.load("corrupt") is None

    def test_multiple_stages(self, manager, sample_state):
        manager.save("stage_a", sample_state)
        manager.save("stage_b", sample_state)
        assert manager.stage_exists("stage_a")
        assert manager.stage_exists("stage_b")


class TestCheckpointClear:
    def test_clear_removes_run_files(self, manager, sample_state):
        manager.save("stage1", sample_state)
        manager.save("stage2", sample_state)
        manager.clear()
        assert not manager.stage_exists("stage1")
        assert not manager.stage_exists("stage2")

    def test_clear_other_run_unaffected(self, manager, sample_state, tmp_path, monkeypatch):
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        other_repo = LocalDiskStateRepository(project_name="test_proj")
        other = CheckpointManager(repository=other_repo, run_id="other_run")
        manager.save("stage1", sample_state)
        other.save("stage_other", sample_state)
        manager.clear()
        assert other.stage_exists("stage_other")


class TestFindLatestRun:
    def test_no_checkpoints_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        repo = LocalDiskStateRepository(project_name="empty")
        m = CheckpointManager(repository=repo)
        assert m.find_latest_run_id() is None

    def test_finds_latest_by_mtime(self, manager, tmp_path, monkeypatch):
        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        run_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        run_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        repo = LocalDiskStateRepository(project_name="test_proj")
        older = CheckpointManager(repository=repo, run_id=run_a)
        older.save("s", PipelineState(run_id=run_a, stage="s"))
        time.sleep(0.01)
        newer = CheckpointManager(repository=repo, run_id=run_b)
        newer.save("s", PipelineState(run_id=run_b, stage="s"))

        latest = CheckpointManager(repository=repo).find_latest_run_id()
        assert latest == run_b


class TestAtomicSave:
    """Post-fix: save() writes via a .tmp + os.replace, so the destination
    is never observed in a partial state. Previously a SIGTERM mid-write
    left a zero-length file that find_latest_run_id would later pick up."""

    def test_save_leaves_no_tmp_file_behind(self, manager, sample_state):
        manager.save("stage1", sample_state)
        filepath = Path(manager.repository._get_filepath(manager.run_id, "stage1"))
        assert filepath.exists(), "final checkpoint not created"
        assert not filepath.with_suffix(filepath.suffix + ".tmp").exists(), "tmp file was left on disk after save()"

    def test_save_overwrite_is_atomic(self, manager, sample_state):
        # First write.
        manager.save("stage1", sample_state)
        # Mutate state and save again — final content must reflect the new state,
        # never a half-overwritten file.
        new_state = PipelineState(run_id="test_run", stage="updated", attempt=42)
        manager.save("stage1", new_state)
        loaded = manager.load("stage1")
        assert loaded is not None
        assert loaded.stage == "updated"
        assert loaded.attempt == 42
