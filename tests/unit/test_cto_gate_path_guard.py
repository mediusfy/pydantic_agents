"""AP-110: the CTO gate's read path must carry the guard its write path already has.

``write_approval`` joins the caller-chosen ``project`` with ``safe_path_join`` and documents why. Three lines
above it, ``_approval_file_path`` built the same path with a plain ``Path`` join and then called
``mkdir(parents=True)``, so a ``project`` of ``../../tmp/x`` both escaped ``output_dir`` and created directories
outside it. The root cause is that the same path expression was written twice; both now derive it from one
helper so a third copy cannot drift out of guard again. See ADR-0002.
"""

import pytest

from pydantic_agents.core.cto_gate import _APPROVAL_FILE, _approval_file_path, write_approval
from pydantic_agents.core.exceptions import PathTraversalError

# Escapes one level out of output_dir into a sibling that does not exist, so "was it created?" is answerable.
# Shared by the two tests that only assert on the raise; the test that asserts the directory was *not* created
# uses a private name, so another test's side effect cannot satisfy its precondition for it.
ESCAPE_DIR = "pa-ap110-escape"
TRAVERSAL = f"../{ESCAPE_DIR}/nested"


@pytest.fixture(autouse=True)
def _output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("pydantic_agents.core.settings.settings.output_dir", tmp_path)
    return tmp_path


def test_approval_file_path_rejects_traversal():
    """The read path must refuse a project name that escapes the output dir."""
    with pytest.raises(PathTraversalError):
        _approval_file_path(TRAVERSAL)


def test_a_rejected_traversal_creates_no_directory(_output_dir):
    """The guard has to run before the mkdir, or the escape still happens as a side effect."""
    own_escape = "pa-ap110-mkdir-probe"
    escaped = _output_dir.parent / own_escape
    assert not escaped.exists(), "precondition: the escape target must not exist yet"

    with pytest.raises(PathTraversalError):
        _approval_file_path(f"../{own_escape}/nested")

    assert not escaped.exists(), f"a rejected traversal still created {escaped}"


def test_write_approval_still_rejects_traversal():
    """Unchanged behaviour on the write path; guards the refactor."""
    with pytest.raises(PathTraversalError):
        write_approval(TRAVERSAL)


def test_read_and_write_paths_agree(_output_dir):
    """One shared helper, so the gate cannot read one location and write another."""
    written = write_approval("proj", approved=True)
    read = _approval_file_path("proj")

    assert read == written
    assert read.is_relative_to(_output_dir)


def test_normal_project_path_is_unchanged(_output_dir):
    """The fix must not move the marker for an ordinary project name."""
    assert _approval_file_path("myproject") == _output_dir / "myproject" / _APPROVAL_FILE


def test_approval_file_path_still_creates_the_parent_directory(_output_dir):
    """Callers rely on the directory existing so the marker can be written into it."""
    path = _approval_file_path("myproject")

    assert path.parent.is_dir()
    assert path.parent == _output_dir / "myproject"
