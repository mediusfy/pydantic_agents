import pytest

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.writer import ArtifactWriter


def test_writes_files(tmp_path):
    writer = ArtifactWriter(str(tmp_path))
    art = Artifact(
        agent_name="backend",
        files={"main.go": "package main", "go.mod": "module test"},
    )
    writer.write(art)
    assert (tmp_path / "backend" / "main.go").read_text() == "package main"
    assert (tmp_path / "backend" / "go.mod").read_text() == "module test"


def test_creates_directories(tmp_path):
    writer = ArtifactWriter(str(tmp_path))
    art = Artifact(
        agent_name="frontend",
        files={"src/components/App.vue": "<template><div/></template>"},
    )
    writer.write(art)
    assert (tmp_path / "frontend" / "src" / "components" / "App.vue").exists()


def test_path_traversal_prevention(tmp_path):
    writer = ArtifactWriter(str(tmp_path))
    art = Artifact(
        agent_name="hacker",
        files={"../../etc/passwd": "root:x:0:0:root:/root:/bin/bash"},
    )
    with pytest.raises(PathTraversalError, match="Path traversal"):
        writer.write(art)


def test_path_traversal_absolute(tmp_path):
    writer = ArtifactWriter(str(tmp_path))
    art = Artifact(
        agent_name="hacker",
        files={"/etc/passwd": "root:x:0:0:"},
    )
    with pytest.raises(PathTraversalError, match="Path traversal"):
        writer.write(art)


@pytest.mark.parametrize(
    "bad_agent_name",
    [
        "../escaped",
        "../../etc",
        "/tmp/escaped",
        "backend/../../escaped",
        "backend/nested",
        ".",
        "",
    ],
)
def test_path_traversal_in_agent_name_is_rejected(tmp_path, bad_agent_name):
    writer = ArtifactWriter(str(tmp_path))
    art = Artifact(
        agent_name=bad_agent_name,
        files={"main.go": "package main"},
    )
    with pytest.raises(PathTraversalError, match="Path traversal"):
        writer.write(art)
