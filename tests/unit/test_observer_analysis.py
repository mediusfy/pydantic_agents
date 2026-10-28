"""observer_analysis.detect_findings: deterministic heuristics over platform_metrics rows (no I/O, no LLM)."""

from pydantic_agents.core.observer_analysis import detect_findings


def _row(agent, success, duration=None):
    return {"agent": agent, "success": success, "duration": duration, "verdict": None}


def test_no_rows_means_no_findings():
    assert detect_findings([]) == []


def test_all_passing_is_not_a_finding():
    rows = [_row("review", True) for _ in range(5)]
    assert detect_findings(rows) == []


def test_all_failing_is_a_regression():
    rows = [_row("backend", False) for _ in range(3)]
    findings = detect_findings(rows)
    assert len(findings) == 1
    assert findings[0].kind == "regression"
    assert findings[0].severity == "high"
    assert findings[0].agent == "backend"


def test_mixed_success_is_flaky():
    rows = [_row("security", True), _row("security", False), _row("security", True), _row("security", False)]
    findings = detect_findings(rows)
    assert len(findings) == 1
    assert findings[0].kind == "flaky"
    assert findings[0].evidence == {"total": 4, "failed": 2}


def test_a_short_mixed_window_is_not_flagged():
    rows = [_row("security", True), _row("security", False)]
    assert detect_findings(rows) == []


def test_latency_drift_is_a_health_finding():
    rows = [
        _row("database", True, 3.0),
        _row("database", True, 3.2),
        _row("database", True, 1.0),
        _row("database", True, 1.1),
    ]
    findings = detect_findings(rows)
    assert len(findings) == 1
    assert findings[0].kind == "health"
    assert findings[0].agent == "database"
    assert findings[0].evidence["newer_avg_seconds"] > findings[0].evidence["older_avg_seconds"]


def test_stable_latency_is_not_a_finding():
    rows = [
        _row("database", True, 1.0),
        _row("database", True, 1.05),
        _row("database", True, 0.95),
        _row("database", True, 1.02),
    ]
    assert detect_findings(rows) == []


def test_findings_are_grouped_per_agent():
    rows = [_row("backend", False), _row("backend", False), _row("frontend", True), _row("frontend", True)]
    findings = detect_findings(rows)
    assert {f.agent for f in findings} == {"backend"}
