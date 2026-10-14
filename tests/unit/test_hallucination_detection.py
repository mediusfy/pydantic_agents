"""Tests for the hallucination detector."""

from tests.evaluation.hallucination_detector import (
    _are_contradictory,
    _extract_claims,
    _is_generic_claim,
    detect_hallucination,
)


class TestHallucinationDetection:
    def test_no_hallucination_when_output_matches_context(self):
        report = detect_hallucination(
            "The system uses PostgreSQL for persistent storage.",
            "The system uses PostgreSQL for storage. It runs on Kubernetes.",
        )
        assert report.passed
        assert report.score == 1.0
        assert len(report.issues) == 0

    def test_detects_unsupported_claims(self):
        report = detect_hallucination(
            "The customer should use MongoDB for storage. Redis should be used for caching.",
            "The system uses PostgreSQL for storage.",
        )
        assert report.score < 1.0
        assert len(report.unsupported_claims) > 0

    def test_generic_claims_ignored(self):
        report = detect_hallucination(
            "Ensure that the database connection is pooled. Use the connection string from config.",
            "The system uses PostgreSQL.",
        )
        assert report.passed

    def test_contradiction_detection(self):
        assert _are_contradictory("The system is active", "The system is inactive")
        assert _are_contradictory("We use synchronous calls", "We use asynchronous calls")
        assert not _are_contradictory("The system uses PostgreSQL", "The system uses Redis")

    def test_self_contradiction_lowers_score(self):
        report = detect_hallucination(
            "The system uses optimistic locking. Updates use pessimistic concurrency control.",
            "The system handles concurrent updates.",
        )
        assert report.score < 1.0

    def test_empty_inputs_cause_no_crash(self):
        report = detect_hallucination("", "")
        assert report.passed
        assert report.score == 1.0

    def test_claim_extraction_filters_short_sentences(self):
        claims = _extract_claims("Yes. No. The system must handle 10k requests per second.")
        assert len(claims) == 1

    def test_generic_claim_patterns_recognized(self):
        assert _is_generic_claim("Use the configuration file for settings")
        assert _is_generic_claim("It is important to validate input")
        assert not _is_generic_claim("The system requires PostgreSQL 15 or later")


class TestContextDrivenHallucination:
    def test_code_generation_stays_grounded(self):
        context = "Use Go with chi router and slog for logging."
        output = 'package main\n\nimport "github.com/go-chi/chi/v5"\n\nfunc main() { r := chi.NewRouter() }'
        report = detect_hallucination(output, context)
        assert report.passed

    def test_false_positive_tolerated(self):
        context = "The service runs on Kubernetes."
        output = "The service deploys to Kubernetes and scales automatically."
        report = detect_hallucination(output, context)
        assert report.score >= 0.85
