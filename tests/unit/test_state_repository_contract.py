import uuid

import pytest

from pydantic_agents.core.adapters.state_repository import LocalDiskStateRepository, build_state_repository
from pydantic_agents.core.checkpoint import CheckpointManager
from pydantic_agents.core.models.core import PipelineState

turso = pytest.importorskip("turso")

from pydantic_agents.core.adapters.turso_state_repository import TursoStateRepository  # noqa: E402

BACKENDS = ["disk", "turso"]


@pytest.fixture
def make_repo(tmp_path, monkeypatch):
    monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
    monkeypatch.setattr("pydantic_agents.core.adapters.turso_state_repository.settings.checkpoint_dir", str(tmp_path))

    def _make(backend: str, project: str = "proj"):
        if backend == "turso":
            return TursoStateRepository(project)
        return LocalDiskStateRepository(project)

    return _make


def _state(run_id: str, stage: str = "init", attempt: int = 1) -> PipelineState:
    return PipelineState(run_id=run_id, stage=stage, attempt=attempt)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "stage, attempt",
    [("stage1", 1), ("stage with space", 2), ("a/b", 3)],
)
def test_round_trip(make_repo, backend, stage, attempt):
    repo = make_repo(backend)
    repo.save("run1", stage, _state("run1", "s", attempt))
    loaded = repo.load("run1", stage)
    assert loaded is not None
    assert (loaded.run_id, loaded.stage, loaded.attempt) == ("run1", "s", attempt)


@pytest.mark.parametrize("backend", BACKENDS)
def test_load_missing_returns_none(make_repo, backend):
    assert make_repo(backend).load("run1", "nope") is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_overwrite_keeps_latest(make_repo, backend):
    repo = make_repo(backend)
    repo.save("run1", "s", _state("run1", "first", 1))
    repo.save("run1", "s", _state("run1", "second", 42))
    loaded = repo.load("run1", "s")
    assert loaded is not None
    assert (loaded.stage, loaded.attempt) == ("second", 42)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "saved, queried, expected",
    [
        ([("run1", "a")], ("run1", "a"), True),
        ([("run1", "a")], ("run1", "b"), False),
        ([("run1", "a")], ("run2", "a"), False),
    ],
)
def test_exists(make_repo, backend, saved, queried, expected):
    repo = make_repo(backend)
    for run_id, stage in saved:
        repo.save(run_id, stage, _state(run_id))
    assert repo.exists(*queried) is expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_clear_only_affects_target_run(make_repo, backend):
    repo = make_repo(backend)
    repo.save("run1", "a", _state("run1"))
    repo.save("run1", "b", _state("run1"))
    repo.save("run2", "a", _state("run2"))
    repo.clear("run1")
    assert not repo.exists("run1", "a")
    assert not repo.exists("run1", "b")
    assert repo.exists("run2", "a")


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_latest_run_id(make_repo, backend):
    repo = make_repo(backend)
    assert repo.find_latest_run_id() is None
    run_a, run_b = str(uuid.uuid4()), str(uuid.uuid4())
    repo.save(run_a, "s", _state(run_a))
    repo.save(run_b, "s", _state(run_b))
    assert repo.find_latest_run_id() == run_b


@pytest.mark.parametrize("backend", BACKENDS)
def test_projects_are_isolated(make_repo, backend):
    one, two = make_repo(backend, "one"), make_repo(backend, "two")
    one.save("run1", "s", _state("run1"))
    assert not two.exists("run1", "s")
    assert two.find_latest_run_id() is None


def test_turso_corrupt_row_returns_none(make_repo):
    repo = make_repo("turso")
    repo._conn.execute(
        "INSERT INTO pipeline_checkpoints (run_id, stage, state_json, updated_at) VALUES (?, ?, ?, ?)",
        ("run1", "bad", "{not json", 1.0),
    )
    repo._conn.commit()
    assert repo.load("run1", "bad") is None


def test_turso_state_survives_reopen(make_repo):
    first = make_repo("turso")
    first.save("run1", "s", _state("run1", "persisted", 7))
    first.close()
    loaded = make_repo("turso").load("run1", "s")
    assert loaded is not None
    assert (loaded.stage, loaded.attempt) == ("persisted", 7)


@pytest.mark.parametrize(
    "backend, expected_type",
    [("disk", LocalDiskStateRepository), ("turso", TursoStateRepository)],
)
def test_build_state_repository_selects_backend(make_repo, monkeypatch, backend, expected_type):
    make_repo("disk")  # applies checkpoint_dir patches
    monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.state_backend", backend)
    assert isinstance(build_state_repository("proj"), expected_type)


def test_checkpoint_manager_works_with_turso(make_repo):
    mgr = CheckpointManager(repository=make_repo("turso"), run_id="run1")
    mgr.save("s", _state("run1"))
    assert mgr.stage_exists("s")
    assert mgr.load("s") is not None
    mgr.clear()
    assert not mgr.stage_exists("s")
