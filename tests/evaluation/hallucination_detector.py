"""Hallucination detection for agent outputs.

Provides utilities to detect when agents generate content unsupported by
their input context. Three primary checks:

1. Factual consistency  — output statements match the input facts
2. Source grounding     — claims are traceable to context or tools
3. Self-contradiction   — output does not contradict itself
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HallucinationReport:
    score: float = 1.0
    issues: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradictory_pairs: list[tuple[str, str]] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "issues": self.issues,
            "unsupported_claims": len(self.unsupported_claims),
            "contradictions": len(self.contradictory_pairs),
            "passed": self.passed,
        }


def detect_hallucination(
    output_text: str, source_context: str, check_self_contradiction: bool = True
) -> HallucinationReport:
    report = HallucinationReport()

    _check_factual_consistency(report, output_text, source_context)

    if check_self_contradiction:
        _check_self_contradiction(report, output_text)

    if report.issues:
        penalty = min(1.0, len(report.issues) * 0.15)
        report.score = max(0.0, 1.0 - penalty)
        report.passed = report.score >= 0.8

    return report


def _check_factual_consistency(report: HallucinationReport, output: str, context: str) -> None:
    claims = _extract_claims(output)
    context_lower = context.lower()

    for claim in claims:
        if claim.lower() not in context_lower and not _is_generic_claim(claim):
            report.unsupported_claims.append(claim)
            report.issues.append(f"Unsupported claim not found in context: '{claim[:80]}'")


def _check_self_contradiction(report: HallucinationReport, text: str) -> None:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if _are_contradictory(sentences[i], sentences[j]):
                report.contradictory_pairs.append((sentences[i][:80], sentences[j][:80]))
                report.issues.append("Self-contradiction detected")


def _extract_claims(text: str) -> list[str]:
    claims = []
    for sentence in text.split("."):
        s = sentence.strip()
        if _looks_like_claim(s):
            claims.append(s)
    return claims


def _looks_like_claim(sentence: str) -> bool:
    if len(sentence) < 20:
        return False
    if sentence.startswith(("Given", "When", "Then", "As a", "I want", "So that")):
        return True
    return bool(any(kw in sentence.lower() for kw in ["should", "must", "will", "requires", "implements", "returns"]))


def _is_generic_claim(claim: str) -> bool:
    generic_patterns = [
        "use the",
        "follow the",
        "ensure that",
        "make sure",
        "consider",
        "note that",
        "remember",
        "keep in mind",
        "it is important",
        "we recommend",
        "best practice",
    ]
    claim_lower = claim.lower()
    return any(claim_lower.startswith(p) for p in generic_patterns)


_CONTRADICTIONS = [
    ("activ", "inactiv"),
    ("enabl", "disabl"),
    ("synchron", "asynchron"),
    ("stateless", "stateful"),
    ("strongly consistent", "eventually consistent"),
    ("optimistic", "pessimistic"),
    ("allow", "deny"),
    ("mantain", "delete"),
    ("success", "fail"),
]


def _are_contradictory(a: str, b: str) -> bool:
    a_lower = a.lower()
    b_lower = b.lower()
    for pos, neg in _CONTRADICTIONS:
        if pos in a_lower and neg in b_lower:
            return True
        if neg in a_lower and pos in b_lower:
            return True
    return False
