"""Tests for utility functions."""

import pytest

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.utils import format_artifacts, safe_path_join, safe_write


class TestSafePathJoin:
    def test_normal_join(self, tmp_path):
        result = safe_path_join(str(tmp_path), "subdir", "file.txt")
        assert result == tmp_path / "subdir" / "file.txt"

    def test_traversal_prevention(self, tmp_path):
        with pytest.raises(PathTraversalError, match="Path traversal"):
            safe_path_join(str(tmp_path), "..", "..", "etc", "passwd")

    def test_absolute_prevention(self, tmp_path):
        with pytest.raises(PathTraversalError, match="Path traversal"):
            safe_path_join(str(tmp_path), "/etc/passwd")

    def test_single_component(self, tmp_path):
        result = safe_path_join(str(tmp_path), "file.txt")
        assert result == tmp_path / "file.txt"


class TestSafeWrite:
    def test_writes_content(self, tmp_path):
        result = safe_write(str(tmp_path), "hello.txt", "world")
        assert result == tmp_path / "hello.txt"
        assert result.read_text() == "world"

    def test_creates_directories(self, tmp_path):
        result = safe_write(str(tmp_path), "a/b/c/deep.txt", "nested")
        assert result.exists()
        assert result.read_text() == "nested"


class TestFormatArtifacts:
    def test_basic_formatting(self):
        arts = [
            Artifact(agent_name="backend", files={"main.go": "package main"}, metadata={"user_story_id": "US-001"}),
        ]
        result = format_artifacts(arts, max_chars=10_000)
        assert "backend" in result
        assert "main.go" in result
        assert "package main" in result

    def test_truncation(self):
        arts = [
            Artifact(agent_name="backend", files={"main.go": "x" * 10_000}, metadata={"user_story_id": "US-001"}),
        ]
        result = format_artifacts(arts, max_chars=500)
        # Result includes header + truncated content (may slightly exceed max_chars)
        assert len(result) < len(arts[0].files["main.go"])  # signficantly shorter than full
        assert "... (truncated)" in result

    def test_error_artifact(self):
        arts = [Exception("something broke")]
        result = format_artifacts(arts, max_chars=10_000)
        assert "something broke" in result
        assert "Error" in result

    def test_multiple_artifacts(self):
        arts = [
            Artifact(agent_name="a1", files={"f1.go": "p1"}, metadata={"user_story_id": "US-001"}),
            Artifact(agent_name="a2", files={"f2.go": "p2"}, metadata={"user_story_id": "US-002"}),
        ]
        result = format_artifacts(arts, max_chars=10_000)
        assert "a1" in result
        assert "a2" in result

    def test_no_agent_header(self):
        arts = [
            Artifact(agent_name="backend", files={"main.go": "package main"}, metadata={"user_story_id": "US-001"}),
        ]
        result = format_artifacts(arts, max_chars=10_000, include_agent_header=False)
        assert "=== Agent:" not in result
        assert "package main" in result

    def test_empty_artifacts(self):
        result = format_artifacts([])
        assert result == ""
