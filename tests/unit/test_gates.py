from unittest.mock import patch

from pydantic_agents.core.gates import HumanInTheLoopGate
from pydantic_agents.core.models.core import Artifact


def sample_artifacts():
    return [
        Artifact(
            agent_name="backend",
            files={"main.go": "package main"},
            metadata={"user_story_id": "US-001"},
        )
    ]


def test_auto_yes_returns_true():
    assert HumanInTheLoopGate.prompt_proceed([], auto_yes=True) is True


@patch("builtins.input", return_value="y")
def test_confirm_yes(mock_input):
    assert HumanInTheLoopGate.prompt_proceed(sample_artifacts()) is True


@patch("builtins.input", return_value="n")
def test_confirm_no(mock_input):
    assert HumanInTheLoopGate.prompt_proceed(sample_artifacts()) is False


@patch("builtins.input", return_value="")
def test_confirm_empty_means_yes(mock_input):
    assert HumanInTheLoopGate.prompt_proceed(sample_artifacts()) is True
