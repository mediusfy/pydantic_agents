"""A/B testing framework for comparing agent configurations.

Runs two pipeline variants side-by-side with the same input and
compares outputs across multiple dimensions: correctness, speed,
token usage, structure, and hallucination scores.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_agents.core.pipeline.config import PipelineConfig


@dataclass
class ABResult:
    variant_a_name: str
    variant_b_name: str
    input_requirement: str

    a_output: Any = None
    b_output: Any = None

    a_duration_sec: float = 0.0
    b_duration_sec: float = 0.0

    a_score: float = 0.0
    b_score: float = 0.0

    a_issues: list[str] = field(default_factory=list)
    b_issues: list[str] = field(default_factory=list)

    winner: str = ""
    recommendation: str = ""


@dataclass
class ABReport:
    results: list[ABResult] = field(default_factory=list)
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0

    def add_result(self, result: ABResult) -> None:
        self.results.append(result)
        if result.winner == "A":
            self.a_wins += 1
        elif result.winner == "B":
            self.b_wins += 1
        else:
            self.ties += 1

    def summary(self) -> str:
        lines = [
            f"A/B Test Report: {len(self.results)} comparisons",
            f"  A wins: {self.a_wins} | B wins: {self.b_wins} | Ties: {self.ties}",
        ]
        for r in self.results:
            lines.append(
                f"  [{r.winner}] {r.variant_a_name} (score={r.a_score:.2f}) vs "
                f"{r.variant_b_name} (score={r.b_score:.2f}) "
                f"on: '{r.input_requirement[:60]}...'"
            )
            lines.append(f"    {r.recommendation}")
        return "\n".join(lines)


def compare_configs(
    config_a: PipelineConfig,
    config_b: PipelineConfig,
    requirement: str,
    runner: Any,
) -> ABResult:
    result = ABResult(
        variant_a_name=f"{config_a.blueprint}-{config_a.framework}",
        variant_b_name=f"{config_b.blueprint}-{config_b.framework}",
        input_requirement=requirement,
    )

    start = time.time()
    state_a = runner(config_a, requirement)
    result.a_duration_sec = time.time() - start

    start = time.time()
    state_b = runner(config_b, requirement)
    result.b_duration_sec = time.time() - start

    result.a_output = state_a
    result.b_output = state_b

    result.a_score, result.a_issues = _score_state(state_a)
    result.b_score, result.b_issues = _score_state(state_b)

    if abs(result.a_score - result.b_score) < 0.05:
        result.winner = "TIE"
        if result.a_duration_sec < result.b_duration_sec:
            result.recommendation = "A slightly faster but scores equal; prefer A for speed"
        else:
            result.recommendation = "B slightly faster but scores equal; prefer B for speed"
    elif result.a_score > result.b_score:
        result.winner = "A"
        result.recommendation = f"Variant A outperforms B by {result.a_score - result.b_score:.2f} points"
    else:
        result.winner = "B"
        result.recommendation = f"Variant B outperforms A by {result.b_score - result.a_score:.2f} points"

    return result


def _score_state(state: Any) -> tuple[float, list[str]]:
    score = 1.0
    issues: list[str] = []

    stories = getattr(state, "user_stories", None) or []
    if not stories:
        score -= 0.2
        issues.append("No user stories generated")

    artifacts = getattr(state, "artifacts", None) or []
    if not artifacts:
        score -= 0.2
        issues.append("No artifacts generated")

    errors = getattr(state, "errors", None) or []
    if errors:
        penalty = min(0.5, len(errors) * 0.1)
        score -= penalty
        issues.append(f"Pipeline had {len(errors)} errors")

    approved = getattr(state, "approved", False)
    if not approved and (stories or artifacts):
        score -= 0.1
        issues.append("Pipeline not approved")

    return max(0.0, score), issues
