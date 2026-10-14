"""Agent evaluation framework.

Provides:
- Hallucination detection (factual consistency, source grounding, self-contradiction)
- Decision correctness evaluation (golden reference comparison)
- A/B testing (side-by-side agent configuration comparison)
- Shared fixtures (mock LLM responses, golden test data)
"""

from tests.evaluation.ab_testing import ABReport, ABResult, compare_configs
from tests.evaluation.decision_evaluator import (
    DecisionReport,
    evaluate_decisions,
    validate_artifact_structure,
    validate_story_structure,
)
from tests.evaluation.fixtures import (
    MOCK_ADVISORY_OUTPUT,
    MOCK_BA_OUTPUT,
    MOCK_BACKEND_OUTPUT,
    MOCK_QUALITY_GATE_OUTPUT,
    MockLLMResponse,
    make_agent_advice,
    make_artifact,
    make_pipeline_state,
    make_quality_gate_result,
    make_review_result,
    make_security_report,
    make_user_story,
)
from tests.evaluation.hallucination_detector import (
    HallucinationReport,
    detect_hallucination,
)
