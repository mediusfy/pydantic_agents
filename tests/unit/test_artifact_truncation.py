"""AP-105: a quality gate must never report a pass on code it did not receive.

``format_artifacts`` enforces a character budget and stops once it is spent. It did leave a prose note behind,
but the note claimed "review continues on remaining items" while the code had in fact broken out of the artifact
loop, it never said how much was missing, and — the part that matters — nothing was machine-readable. So
``_run_gate`` could not tell a complete batch from a partial one and still returned "no findings", which the
stage records as an approval. That is a silent false approval inside an autonomous remediation loop. See ADR-0002.

Still open on AP-105 after this: splitting one oversized artifact across several gate calls, and an unforgeable
provenance envelope for model-authored content.
"""

import asyncio

from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.utils import format_artifacts


def _artifact(name: str, files: dict[str, str]) -> Artifact:
    return Artifact(agent_name=name, files=files, metadata={"user_story_id": f"US-{name}"})


def _many_artifacts(count: int, body_chars: int = 400) -> list[Artifact]:
    return [_artifact(f"agent{i}", {f"file{i}.go": "x" * body_chars}) for i in range(count)]


def test_format_artifacts_report_counts_omitted_artifacts():
    """The budget drops whole artifacts; the caller must be able to learn how many."""
    from pydantic_agents.core.utils import format_artifacts_report

    artifacts = _many_artifacts(4, body_chars=400)
    # Room for roughly one artifact.
    report = format_artifacts_report(artifacts, max_chars=500)

    assert report.truncated is True
    assert report.omitted_artifacts >= 2, f"only {report.omitted_artifacts} of 4 artifacts reported as omitted"
    assert report.omitted_artifacts <= 3


def test_format_artifacts_report_counts_omitted_files():
    """Files dropped inside the artifact that crossed the budget are counted too."""
    from pydantic_agents.core.utils import format_artifacts_report

    art = _artifact("backend", {f"f{i}.go": "y" * 300 for i in range(10)})
    report = format_artifacts_report([art], max_chars=600)

    assert report.truncated is True
    assert report.omitted_files > 0


def test_truncation_notice_states_the_view_is_incomplete():
    """The note must be accurate and quantified, and must stop claiming the review carried on."""
    from pydantic_agents.core.utils import format_artifacts_report

    report = format_artifacts_report(_many_artifacts(4, body_chars=400), max_chars=500)

    assert "review continues on remaining items" not in report.text
    assert "TRUNCATED" in report.text
    assert "INCOMPLETE" in report.text
    assert str(report.omitted_artifacts) in report.text


def test_untruncated_output_reports_no_omissions():
    """The normal path must not raise a false alarm, and must match the string-returning wrapper."""
    from pydantic_agents.core.utils import format_artifacts_report

    artifacts = _many_artifacts(3, body_chars=50)
    report = format_artifacts_report(artifacts, max_chars=80_000)

    assert report.truncated is False
    assert report.omitted_artifacts == 0
    assert report.omitted_files == 0
    assert "TRUNCATED" not in report.text
    assert report.text == format_artifacts(artifacts, max_chars=80_000)


def test_a_batch_landing_exactly_on_the_budget_is_not_truncated():
    """Reaching the budget is not the same as losing content.

    If hitting ``max_chars`` exactly counted as truncation, a complete batch would be failed closed as
    inconclusive for no reason. Truncation is reported only when something was actually dropped or cut.
    """
    from pydantic_agents.core.utils import _artifact_header, format_artifacts_report

    art = _artifact("solo", {"f.go": "z" * 100})
    # What the formatter accumulates internally: the header plus the file snippet, without join separators.
    exact = len(_artifact_header(art)) + len(f"--- f.go ---\n{'z' * 100}")

    report = format_artifacts_report([art], max_chars=exact)

    assert report.truncated is False
    assert report.omitted_artifacts == 0
    assert report.omitted_files == 0
    assert "TRUNCATED" not in report.text


def test_a_file_cut_mid_content_counts_as_truncation():
    """Nothing is omitted whole, but the gate still did not see all of the file."""
    from pydantic_agents.core.utils import _artifact_header, format_artifacts_report

    art = _artifact("solo", {"f.go": "z" * 100})
    exact = len(_artifact_header(art)) + len(f"--- f.go ---\n{'z' * 100}")

    report = format_artifacts_report([art], max_chars=exact - 1)

    assert report.truncated is True
    assert report.omitted_artifacts == 0
    assert report.omitted_files == 0
    assert report.omitted_chars == 1


def test_one_oversized_artifact_reports_the_characters_it_lost():
    """The headline AP-105 case omits no whole artifact and no whole file.

    Counts of 0 and 0 would read as a spurious alarm over ~20 KB of code the gate never saw, so the loss has to
    be quantified in characters or an operator will dismiss the finding.
    """
    from pydantic_agents.core.utils import format_artifacts_report

    report = format_artifacts_report([_artifact("backend", {"big.go": "x" * 100_000})], max_chars=80_000)

    assert report.truncated is True
    assert report.omitted_artifacts == 0
    assert report.omitted_files == 0
    assert report.omitted_chars > 19_000, "the characters cut from the oversized file must be reported"
    assert str(report.omitted_chars) in report.text


def test_the_inconclusive_finding_quantifies_what_the_gate_missed():
    """The operator-facing message must carry the character count, not just two zeroes."""

    async def _always_passes(_state, _config, _text, _verification):
        return None

    findings = _run_gate_over([_artifact("backend", {"big.go": "x" * 200_000})], _always_passes)

    assert findings is not None
    assert "characters of code not shown" in findings[0]
    assert "0 characters" not in findings[0]


def _run_gate_over(artifacts, gate):
    from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

    return asyncio.run(QualityGateStage._run_gate(gate, None, None, [list(artifacts)], "verification: ok"))


def test_a_gate_cannot_pass_on_a_truncated_batch():
    """The finding that matters: a clean verdict on a partial view is not a pass."""
    from pydantic_agents.core.stages.quality_gate_stage import GATE_BATCH_CHARS

    seen: list[str] = []

    async def _always_passes(_state, _config, text, _verification):
        seen.append(text)
        return None  # "no findings" — the gate is happy with what it saw

    # A batch far larger than one gate prompt can hold.
    oversized = _many_artifacts(6, body_chars=GATE_BATCH_CHARS // 2)
    findings = _run_gate_over(oversized, _always_passes)

    assert seen, "the gate should still have been called"
    assert findings is not None, "the gate returned a pass on a batch it only partly received (AP-105)"
    assert any("INCONCLUSIVE" in f for f in findings)


def test_a_gate_still_passes_an_untruncated_batch():
    """No false alarm: a batch that fits is judged normally."""

    async def _always_passes(_state, _config, _text, _verification):
        return None

    assert _run_gate_over(_many_artifacts(2, body_chars=100), _always_passes) is None


def test_a_real_finding_still_wins_over_the_truncation_notice():
    """A gate that found something reports that, not the truncation note."""
    from pydantic_agents.core.stages.quality_gate_stage import GATE_BATCH_CHARS

    async def _finds_something(_state, _config, _text, _verification):
        return ["hardcoded credential in main.go"]

    findings = _run_gate_over(_many_artifacts(6, body_chars=GATE_BATCH_CHARS // 2), _finds_something)

    assert findings == ["hardcoded credential in main.go"]
