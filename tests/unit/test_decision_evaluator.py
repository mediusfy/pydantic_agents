"""Tests for the decision evaluator."""

from tests.evaluation.decision_evaluator import (
    _get_nested,
    evaluate_decisions,
    validate_artifact_structure,
    validate_story_structure,
)


class TestDecisionEvaluator:
    def test_exact_match_scores_1(self):
        output = {"approved": True, "security_score": 85}
        expected = {"approved": True, "security_score": 85}
        report = evaluate_decisions(output, expected)
        assert report.score == 1.0
        assert report.passed
        assert report.mismatches == 0

    def test_mismatch_reduces_score(self):
        output = {"approved": False, "security_score": 40}
        expected = {"approved": True, "security_score": 85}
        report = evaluate_decisions(output, expected)
        assert report.score < 1.0
        assert report.mismatches >= 1

    def test_partial_match_scored_proportionally(self):
        output = {"approved": True, "security_score": 85, "risk_level": "LOW"}
        expected = {"approved": True, "security_score": 100}
        report = evaluate_decisions(output, expected)
        assert report.score == 0.5

    def test_nested_keys_resolved(self):
        output = {"advisory_results": {"cto": "Use hex arch", "ux": "Use Vue 3"}}
        expected = {"advisory_results.cto": "Use hex arch"}
        report = evaluate_decisions(output, expected)
        assert report.matches == 1

    def test_case_insensitive_string_matching(self):
        output = {"status": "soft_deleted"}
        expected = {"status": "Soft_Deleted"}
        report = evaluate_decisions(output, expected)
        assert report.matches == 1

    def test_float_tolerance(self):
        output = {"score": 0.92}
        expected = {"score": 0.90}
        report = evaluate_decisions(output, expected, tolerance=0.05)
        assert report.matches == 1

    def test_list_overlap_uses_tolerance(self):
        output = {"feedback": ["use hex arch", "add otel", "fix lint"]}
        expected = {"feedback": ["use hex arch", "add otel"]}
        report = evaluate_decisions(output, expected, tolerance=0.3)
        assert report.matches == 1


class TestStructureValidators:
    def test_valid_user_story_passes(self):
        from tests.evaluation.fixtures import make_user_story

        result = validate_story_structure([make_user_story()])
        assert result["valid"]
        assert len(result["errors"]) == 0

    def test_missing_gherkin_detected(self):
        from tests.evaluation.fixtures import make_user_story

        story = make_user_story()
        story.gherkin = ""
        result = validate_story_structure([story])
        assert not result["valid"]

    def test_missing_when_then_detected(self):
        from tests.evaluation.fixtures import make_user_story

        story = make_user_story()
        story.gherkin = "As a user I want payment processing"
        result = validate_story_structure([story])
        assert not result["valid"]

    def test_empty_artifact_detected(self):
        from tests.evaluation.fixtures import make_artifact

        art = make_artifact()
        result = validate_artifact_structure([art])
        assert result["valid"]

    def test_empty_file_content_detected(self):
        from tests.evaluation.fixtures import make_artifact

        art = make_artifact(files={"empty.go": ""})
        result = validate_artifact_structure([art])
        assert not result["valid"]


class TestNestedAccessor:
    def test_dict_access(self):
        assert _get_nested({"a": {"b": "c"}}, "a.b") == "c"

    def test_object_access(self):
        from tests.evaluation.fixtures import make_user_story

        story = make_user_story(title="Test Story")
        assert _get_nested(story, "title") == "Test Story"

    def test_missing_returns_none(self):
        assert _get_nested({"a": 1}, "b") is None
        assert _get_nested({"a": 1}, "a.b") is None
