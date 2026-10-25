"""WorkspaceContext: repository and markdown context built from a directory on disk."""

import json
import logging

import pytest

from pydantic_agents.core.adapters.workspace import WorkspaceContext

NO_CONTEXT = "REPOSITORY CONTEXT: No existing repository context found."


def _ctx(tmp_path) -> WorkspaceContext:
    return WorkspaceContext(str(tmp_path))


# --- get_repo_context -------------------------------------------------------------------------------------------


def test_an_empty_directory_has_no_repo_context(tmp_path):
    assert _ctx(tmp_path).get_repo_context() == NO_CONTEXT


def test_package_json_dependencies_come_before_dev_dependencies(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {"vue": "3", "pinia": "2"}, "devDependencies": {"vitest": "1"}})
    )

    assert _ctx(tmp_path).get_repo_context() == "REPOSITORY CONTEXT:\nJS/TS Dependencies: vue, pinia, vitest"


@pytest.mark.parametrize(
    "content",
    ['{"name": "x"}', '{"dependencies": {}, "devDependencies": {}}', "not json", "[1, 2]", ""],
    ids=["no-deps", "empty-deps", "invalid-json", "not-an-object", "empty-file"],
)
def test_a_package_json_without_usable_dependencies_adds_nothing(tmp_path, content):
    (tmp_path / "package.json").write_text(content)

    assert _ctx(tmp_path).get_repo_context() == NO_CONTEXT


def test_go_mod_requirements_are_listed_by_module(tmp_path):
    (tmp_path / "go.mod").write_text(
        "module example.com/app\n\ngo 1.24\n\nrequire (\n\tgithub.com/a/b v1.0.0\n\tgithub.com/c/d v2.0.0\n)\n"
    )

    context = _ctx(tmp_path).get_repo_context()

    assert context == "REPOSITORY CONTEXT:\nGo Dependencies: require, github.com/a/b, github.com/c/d..."


def test_only_the_first_ten_go_requirements_are_listed(tmp_path):
    lines = "\n".join(f"\tgithub.com/m/p{i} v1.0.0" for i in range(15))
    (tmp_path / "go.mod").write_text(f"module x\n\nrequire (\n{lines}\n)\n")

    listed = _ctx(tmp_path).get_repo_context().split("Go Dependencies: ")[1].rstrip(".").split(", ")

    assert listed == ["require", *(f"github.com/m/p{i}" for i in range(9))]


def test_a_go_mod_with_a_blank_indented_line_is_ignored(tmp_path):
    # An indented blank line has no module name; the whole go.mod section is dropped rather than half-reported.
    (tmp_path / "go.mod").write_text("module x\n\nrequire (\n\tgithub.com/a/b v1.0.0\n\t\n)\n")

    assert _ctx(tmp_path).get_repo_context() == NO_CONTEXT


def test_the_go_packages_tree_shows_two_levels_and_ten_files_per_directory(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "sub" / "deep").mkdir(parents=True)
    for i in range(12):
        (pkg / f"f{i}.go").write_text("")
    (pkg / "sub" / "inner.go").write_text("")
    (pkg / "sub" / "deep" / "hidden.go").write_text("")

    context = _ctx(tmp_path).get_repo_context()

    header, *tree = context.split("Existing Go Packages (pkg/):\n")[1].split("\n")
    assert header == "pkg/"
    root_files = [line for line in tree if line.startswith("  f")]
    assert len(root_files) == 10  # twelve files, ten shown
    assert "  sub/" in tree
    assert "    inner.go" in tree
    assert not any("deep" in line or "hidden" in line for line in tree)  # the third level is not walked


def test_frontend_components_get_their_own_section(tmp_path):
    components = tmp_path / "src" / "components"
    components.mkdir(parents=True)
    (components / "Button.vue").write_text("")

    assert _ctx(tmp_path).get_repo_context() == (
        "REPOSITORY CONTEXT:\nExisting Frontend Components (src/components/):\ncomponents/\n  Button.vue"
    )


def test_sections_are_separated_by_a_blank_line_in_a_fixed_order(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"vue": "3"}}))
    (tmp_path / "go.mod").write_text("require (\n\tgithub.com/a/b v1\n)\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "src" / "components").mkdir(parents=True)

    context = _ctx(tmp_path).get_repo_context()

    sections = context.removeprefix("REPOSITORY CONTEXT:\n").split("\n\n")
    assert [s.split(":")[0] for s in sections] == [
        "JS/TS Dependencies",
        "Go Dependencies",
        "Existing Go Packages (pkg/)",
        "Existing Frontend Components (src/components/)",
    ]


# --- get_markdown_context ---------------------------------------------------------------------------------------


def test_no_markdown_files_means_no_guidelines(tmp_path):
    assert _ctx(tmp_path).get_markdown_context() == ""


def test_standard_files_and_adrs_are_combined_in_order(tmp_path):
    (tmp_path / "AGENTS.md").write_text("  agent rules  \n")
    (tmp_path / "CODING_STANDARDS.md").write_text("standards")
    adrs = tmp_path / "adrs"
    adrs.mkdir()
    (adrs / "0002-b.md").write_text("second")
    (adrs / "0001-a.md").write_text("first")
    (adrs / "notes.txt").write_text("ignored")

    assert _ctx(tmp_path).get_markdown_context() == (
        "LOCAL REPOSITORY GUIDELINES:\n"
        "### AGENTS.md\nagent rules\n\n"
        "### CODING_STANDARDS.md\nstandards\n\n"
        "### Architectural Decision Records (ADRs)\n"
        "==== 0001-a.md ====\nfirst\n\n"
        "==== 0002-b.md ====\nsecond"
    )


def test_empty_markdown_files_are_skipped(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   \n")
    (tmp_path / "adrs").mkdir()
    (tmp_path / "adrs" / "0001.md").write_text("")

    assert _ctx(tmp_path).get_markdown_context() == ""


def test_an_adrs_file_instead_of_a_directory_is_ignored(tmp_path):
    (tmp_path / "adrs").write_text("not a directory")

    assert _ctx(tmp_path).get_markdown_context() == ""


def test_unreadable_files_are_logged_and_skipped(tmp_path, caplog):
    (tmp_path / "AGENTS.md").mkdir()  # exists but cannot be read as a file
    adrs = tmp_path / "adrs"
    adrs.mkdir()
    (adrs / "0001-bad.md").mkdir()
    (adrs / "0002-good.md").write_text("fine")

    with caplog.at_level(logging.WARNING, logger="pydantic_agents.core.adapters.workspace"):
        context = _ctx(tmp_path).get_markdown_context()

    assert (
        context
        == "LOCAL REPOSITORY GUIDELINES:\n### Architectural Decision Records (ADRs)\n==== 0002-good.md ====\nfine"
    )
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Failed to read AGENTS.md:") for m in messages)
    assert any(m.startswith("Failed to read ADR 0001-bad.md:") for m in messages)
