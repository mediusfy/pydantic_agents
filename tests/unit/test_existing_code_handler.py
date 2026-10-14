"""Tests for the existing code pipeline handler."""

from pydantic_agents.core.existing_code_handler import ExistingCodeResult, format_result_summary
from pydantic_agents.core.models.core import Artifact


def test_result_defaults():
    result = ExistingCodeResult(passed=False)
    assert result.passed is False
    assert result.changed_files == []
    assert result.artifacts == []
    assert result.test_files == []
    assert result.quality_gate_score == 0
    assert result.quality_gate_feedback == []
    assert result.review_approved is False
    assert result.review_risk == ""
    assert result.errors == []


def test_result_with_data():
    art = Artifact(agent_name="backend", files={"main.go": "package main"}, metadata={"user_story_id": "US-001"})
    result = ExistingCodeResult(
        passed=True,
        changed_files=["main.go"],
        artifacts=[art],
        test_files=["main_test.go"],
        quality_gate_score=30,
        quality_gate_feedback=["Looks good"],
        review_approved=True,
        review_risk="LOW",
    )
    assert result.passed is True
    assert result.changed_files == ["main.go"]
    assert result.quality_gate_score == 30
    assert result.review_approved is True


class TestFormatResultSummary:
    def test_passed_result(self):
        result = ExistingCodeResult(
            passed=True,
            changed_files=["internal/service/message.go"],
            review_approved=True,
            review_risk="LOW",
        )
        summary = format_result_summary(result)
        assert "PASSED" in summary
        assert "internal/service/message.go" in summary
        assert "LOW" in summary

    def test_failed_result(self):
        result = ExistingCodeResult(
            passed=False,
            errors=["Service Engineer failed: timeout"],
        )
        summary = format_result_summary(result)
        assert "FAILED" in summary
        assert "Service Engineer failed" in summary

    def test_with_test_files(self):
        result = ExistingCodeResult(
            passed=True,
            changed_files=["handler.go"],
            test_files=["handler_test.go"],
            review_approved=True,
            review_risk="LOW",
        )
        summary = format_result_summary(result)
        assert "handler_test.go" in summary

    def test_with_quality_gate_feedback(self):
        result = ExistingCodeResult(
            passed=True,
            changed_files=["main.go"],
            quality_gate_score=45,
            quality_gate_feedback=["Consider simplifying the abstraction"],
            review_approved=True,
            review_risk="MEDIUM",
        )
        summary = format_result_summary(result)
        assert "45" in summary
        assert "Consider simplifying" in summary

    def test_rejected_result(self):
        result = ExistingCodeResult(
            passed=False,
            changed_files=["main.go"],
            review_approved=False,
            review_risk="HIGH",
            errors=["Missing error handling"],
        )
        summary = format_result_summary(result)
        assert "rejected" in summary
        assert "HIGH" in summary
