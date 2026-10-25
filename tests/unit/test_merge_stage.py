"""MergeStage: routing generated artifacts into a project tree, with backups and a path-escape guard."""

import logging
from unittest.mock import patch

import pytest

from pydantic_agents.core.models.core import Artifact, PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.stages.merge_stage import MergeStage, _clean_artifact_path


def _state(*artifacts: Artifact) -> PipelineState:
    state = PipelineState(run_id="r", stage="quality_gate", requirement="x")
    state.artifacts = list(artifacts)
    state.approved = False
    return state


def _config(target, **kwargs) -> PipelineConfig:
    kwargs.setdefault("dry_run", False)
    config = PipelineConfig(**kwargs)
    config.target_root = str(target)  # the constructor derives this from the project name
    config.project = "proj"
    return config


def _art(agent: str, **files: str) -> Artifact:
    return Artifact(agent_name=agent, files={k.replace("__", "/").replace("_dot_", "."): v for k, v in files.items()})


@pytest.fixture(autouse=True)
def no_context_store():
    with patch("pydantic_agents.core.context_engine.store_artifact_context") as store:
        yield store


# --- early exits ------------------------------------------------------------------------------------------------


async def test_a_dry_run_writes_nothing_and_approves(tmp_path):
    state = _state(_art("backend", main_dot_go="x"))

    out = await MergeStage().run(state, _config(tmp_path, dry_run=True))

    assert (out.stage, out.approved) == ("merge", True)
    assert list(tmp_path.iterdir()) == []


async def test_a_missing_target_root_approves_without_writing(tmp_path, caplog):
    state = _state(_art("backend", main_dot_go="x"))

    with caplog.at_level(logging.WARNING, logger="pydantic_agents.core.stages.merge_stage"):
        out = await MergeStage().run(state, _config(tmp_path / "nope"))

    assert (out.stage, out.approved) == ("merge", True)
    assert any("Target project root does not exist" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "nope").exists()


async def test_no_artifacts_approves_without_writing(tmp_path):
    out = await MergeStage().run(_state(), _config(tmp_path))

    assert (out.stage, out.approved) == ("merge", True)
    assert list(tmp_path.iterdir()) == []


# --- routing ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("agent", "subdir"),
    [
        ("frontend", "src"),
        ("frontend_component", "src"),
        ("backend", "backend"),
        ("database", "database"),
        ("infrastructure", "deploy"),
        ("devops", "deploy"),
        ("sre", "deploy"),
        ("docs", "docs"),
        ("test_eng", "tests"),
        ("test_gen", "tests"),
        ("review", "."),
        ("security", "."),
    ],
)
async def test_each_agent_s_files_go_to_its_subdirectory(tmp_path, agent, subdir):
    stage = MergeStage()

    await stage.run(_state(_art(agent, thing_dot_txt="content")), _config(tmp_path))

    assert (tmp_path / subdir / "thing.txt").read_text() == "content"
    assert stage.result.files_written == 1


async def test_artifact_path_prefixes_are_stripped_before_routing(tmp_path):
    await MergeStage().run(
        _state(_art("backend", **{"generated-app__default__backend__pkg__a_dot_go": "x"})), _config(tmp_path)
    )

    assert (tmp_path / "backend" / "pkg" / "a.go").read_text() == "x"


async def test_nested_directories_are_created(tmp_path):
    await MergeStage().run(_state(_art("docs", **{"guides__deep__page_dot_md": "# hi"})), _config(tmp_path))

    assert (tmp_path / "docs" / "guides" / "deep" / "page.md").read_text() == "# hi"


# --- existing files ---------------------------------------------------------------------------------------------


async def test_identical_content_is_skipped(tmp_path):
    target = tmp_path / "backend" / "main.go"
    target.parent.mkdir()
    target.write_text("package main\n")
    stage = MergeStage()

    await stage.run(_state(_art("backend", main_dot_go="  package main  ")), _config(tmp_path))

    assert (stage.result.files_skipped, stage.result.files_written, stage.result.backups_created) == (1, 0, 0)
    assert not (tmp_path / ".merge-backups").exists()


async def test_changed_content_is_backed_up_then_overwritten(tmp_path):
    target = tmp_path / "backend" / "main.go"
    target.parent.mkdir()
    target.write_text("old")
    stage = MergeStage()

    await stage.run(_state(_art("backend", main_dot_go="new")), _config(tmp_path))

    assert target.read_text() == "new"
    assert (tmp_path / ".merge-backups" / "backend" / "main.go").read_text() == "old"
    assert (stage.result.files_written, stage.result.backups_created, stage.result.files_skipped) == (1, 1, 0)


async def test_counts_cover_a_mix_of_new_changed_and_identical_files(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "same.md").write_text("same")
    (tmp_path / "docs" / "edit.md").write_text("old")
    stage = MergeStage()
    artifact = _art("docs", same_dot_md="same", edit_dot_md="new", fresh_dot_md="fresh")

    await stage.run(_state(artifact), _config(tmp_path))

    assert (stage.result.files_written, stage.result.files_skipped, stage.result.backups_created) == (2, 1, 1)


# --- the escape guard -------------------------------------------------------------------------------------------


async def test_a_path_that_climbs_out_of_the_project_is_blocked(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    stage = MergeStage()

    await stage.run(_state(_art("review", **{"..__..__evil_dot_txt": "x"})), _config(root))

    assert not (tmp_path / "evil.txt").exists()
    assert stage.result.files_written == 0
    [error] = stage.result.errors
    assert error.startswith("Path escape blocked: ")


async def test_a_sibling_directory_sharing_the_project_name_prefix_is_blocked(tmp_path):
    root, sibling = tmp_path / "proj", tmp_path / "proj-evil"
    root.mkdir()
    sibling.mkdir()
    stage = MergeStage()

    await stage.run(_state(_art("review", **{"..__proj-evil__x_dot_txt": "x"})), _config(root))

    assert not (sibling / "x.txt").exists()
    assert len(stage.result.errors) == 1


async def test_a_blocked_file_does_not_stop_the_others(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    stage = MergeStage()

    await stage.run(_state(_art("review", **{"..__bad_dot_txt": "x", "ok_dot_txt": "y"})), _config(root))

    assert (root / "ok.txt").read_text() == "y"
    assert stage.result.files_written == 1
    assert len(stage.result.errors) == 1


# --- context store and summary ----------------------------------------------------------------------------------


async def test_each_written_file_is_stored_as_project_context(tmp_path, no_context_store):
    await MergeStage().run(_state(_art("backend", main_dot_go="package main")), _config(tmp_path))

    no_context_store.assert_called_once_with(
        domain="project",
        content="File: main.go\nAgent: backend\nContent:\npackage main",
        metadata={"agent": "backend", "file": "main.go", "project": "proj"},
    )


async def test_skipped_files_are_not_stored(tmp_path, no_context_store):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("same")

    await MergeStage().run(_state(_art("docs", a_dot_md="same")), _config(tmp_path))

    no_context_store.assert_not_called()


async def test_a_failing_context_store_does_not_fail_the_merge(tmp_path, no_context_store):
    no_context_store.side_effect = RuntimeError("db down")
    stage = MergeStage()

    out = await stage.run(_state(_art("docs", a_dot_md="x")), _config(tmp_path))

    assert (tmp_path / "docs" / "a.md").read_text() == "x"
    assert (out.stage, out.approved, stage.result.files_written) == ("merge", True, 1)


async def test_the_summary_and_each_error_are_logged(tmp_path, caplog):
    root = tmp_path / "proj"
    root.mkdir()

    with caplog.at_level(logging.INFO, logger="pydantic_agents.core.stages.merge_stage"):
        await MergeStage().run(_state(_art("review", **{"..__bad_dot_txt": "x", "ok_dot_txt": "y"})), _config(root))

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Merge complete: 1 written, 0 skipped (identical), 0 backups in ") for m in messages)
    assert any(m.startswith("  Merge error: Path escape blocked:") for m in messages)
    assert any(m == "  Merged: ok.txt" for m in messages)


async def test_an_artifact_with_no_files_is_harmless(tmp_path):
    artifact = Artifact(agent_name="docs", files={})

    out = await MergeStage().run(_state(artifact), _config(tmp_path))

    assert (out.stage, out.approved) == ("merge", True)


@pytest.mark.parametrize(
    ("path", "cleaned"),
    [
        ("src/components/Foo.vue", "src/components/Foo.vue"),
        ("cmd/server/main.go", "cmd/server/main.go"),
        ("frontend/src/App.vue", "src/App.vue"),
        ("generated-app/default/frontend/src/App.vue", "src/App.vue"),
        ("default/frontend/src/App.vue", "src/App.vue"),
        ("pkg/frontend/x.go", "pkg/frontend/x.go"),
        ("frontend", "frontend"),
    ],
)
def test_clean_artifact_path(path, cleaned):
    assert _clean_artifact_path(path) == cleaned
