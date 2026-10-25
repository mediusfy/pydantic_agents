from pydantic_agents.core.existing_code_handler import MAX_RETRIES, ExistingCodeResult, format_result_summary


def test_a_passing_run_with_nothing_else_to_report():
    result = ExistingCodeResult(passed=True)

    assert format_result_summary(result) == (
        "## Existing Code Pipeline Results\n"
        "\n"
        "**Status**: PASSED ✅\n"
        f"**Attempts**: 1 / {MAX_RETRIES + 1}\n"
        "\n**Review**: approved ✅ (risk: )"
    )


def test_a_failing_run_lists_every_section_and_only_the_last_five_errors():
    result = ExistingCodeResult(
        passed=False,
        retries_used=1,
        changed_files=["a.go", "b.go"],
        test_files=["a_test.go"],
        quality_gate_score=62,
        quality_gate_feedback=["fix x"],
        review_risk="HIGH",
        errors=[f"e{i}" for i in range(1, 8)],
    )

    assert format_result_summary(result).split("\n") == [
        "## Existing Code Pipeline Results",
        "",
        "**Status**: FAILED 🛑",
        f"**Attempts**: 2 / {MAX_RETRIES + 1}",
        "",
        "**Changed files**:",
        "  - a.go",
        "  - b.go",
        "",
        "**Test files**:",
        "  - a_test.go",
        "",
        "**Quality Gate**: score=62",
        "  - fix x",
        "",
        "**Review**: rejected 🛑 (risk: HIGH)",
        "",
        "**Errors blocking completion**:",
        "  - e3",
        "  - e4",
        "  - e5",
        "  - e6",
        "  - e7",
        "  - (... and 2 more)",
    ]


def test_five_errors_or_fewer_are_all_shown_without_a_remainder_line():
    result = ExistingCodeResult(errors=["e1", "e2", "e3", "e4", "e5"])

    summary = format_result_summary(result)

    assert "  - e1" in summary
    assert "more)" not in summary


def test_the_quality_gate_section_needs_feedback():
    assert "Quality Gate" not in format_result_summary(ExistingCodeResult(quality_gate_score=90))
