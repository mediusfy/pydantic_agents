"""Decision evaluator for agent outputs.

Evaluates whether agent decisions match expected outcomes.
Supports scoring against golden references and rule-based checks.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionReport:
    score: float = 1.0
    matches: int = 0
    mismatches: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "passed": self.passed,
            "details": self.details,
        }


def evaluate_decisions(
    output: Any,
    expected: dict[str, Any],
    tolerance: float = 0.1,
) -> DecisionReport:
    report = DecisionReport()

    for key, expected_val in expected.items():
        actual_val = _get_nested(output, key)
        match = _values_match(actual_val, expected_val, tolerance)
        if match:
            report.matches += 1
            report.details.append({"key": key, "status": "match", "expected": expected_val, "actual": actual_val})
        else:
            report.mismatches += 1
            report.details.append({"key": key, "status": "mismatch", "expected": expected_val, "actual": actual_val})

    total = report.matches + report.mismatches
    if total > 0:
        report.score = report.matches / total
        report.passed = report.score >= 0.8

    return report


def _get_nested(obj: Any, key: str) -> Any:
    parts = key.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "model_dump"):
            d = current.model_dump()
            current = d.get(part)
        else:
            return None
    return current


def _values_match(actual: Any, expected: Any, tolerance: float) -> bool:
    if actual is None:
        return expected is None
    if isinstance(expected, (bool, int, float, str)):
        if isinstance(actual, (bool, int, float, str)):
            if isinstance(expected, float):
                return abs(float(actual) - expected) <= tolerance
            if isinstance(expected, bool):
                return bool(actual) == expected
            return str(actual).lower() == str(expected).lower()
        return False
    if isinstance(expected, list) and isinstance(expected[0], str) and isinstance(actual, list):
        actual_lower = {str(a).lower() for a in actual}
        expected_lower = {str(e).lower() for e in expected}
        overlap = len(actual_lower & expected_lower)
        return overlap >= len(expected_lower) * (1.0 - tolerance)
    if isinstance(expected, dict) and isinstance(actual, dict):
        matches = sum(1 for k, v in expected.items() if _values_match(actual.get(k), v, tolerance))
        return matches >= len(expected) * (1.0 - tolerance)
    return False


def validate_story_structure(stories: list[Any], min_stories: int = 1) -> dict[str, Any]:
    errors = []
    if len(stories) < min_stories:
        errors.append(f"Expected at least {min_stories} stories, got {len(stories)}")
    for i, story in enumerate(stories):
        sid = getattr(story, "id", f"[{i}]")
        if not getattr(story, "title", None):
            errors.append(f"Story {sid}: missing title")
        if not getattr(story, "gherkin", None):
            errors.append(f"Story {sid}: missing gherkin")
        gherkin = str(getattr(story, "gherkin", ""))
        if gherkin and "When" not in gherkin and "Then" not in gherkin:
            errors.append(f"Story {sid}: gherkin missing Given/When/Then")
    return {"valid": len(errors) == 0, "errors": errors}


def validate_artifact_structure(artifacts: list[Any]) -> dict[str, Any]:
    errors = []
    for art in artifacts:
        aname = getattr(art, "agent_name", "?")
        files = getattr(art, "files", {})
        if not files:
            errors.append(f"Artifact from {aname}: no files")
        for fname, fcontent in files.items():
            if not fcontent or not fcontent.strip():
                errors.append(f"Artifact from {aname}: empty file '{fname}'")
    return {"valid": len(errors) == 0, "errors": errors}
