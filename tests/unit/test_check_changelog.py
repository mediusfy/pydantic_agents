import pytest

from scripts.check_changelog import has_new_entry, unreleased_bullets

BASE = """# Changelog

## [Unreleased]

### Added
- existing entry

## [0.2.0]

### Added
- released entry
"""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "base", "head", "expected"),
    [
        ("no change", BASE, BASE, False),
        ("new bullet under unreleased", BASE, BASE.replace("- existing entry", "- existing entry\n- new entry"), True),
        (
            "bullet added to a released version only",
            BASE,
            BASE.replace("- released entry", "- released entry\n- late"),
            False,
        ),
        ("bullet removed", BASE, BASE.replace("- existing entry\n", ""), False),
        ("file created by this PR", "", BASE, True),
        ("file missing in head", BASE, "", False),
    ],
)
def test_has_new_entry(name: str, base: str, head: str, expected: bool) -> None:
    assert has_new_entry(base, head) is expected, name


@pytest.mark.unit
def test_unreleased_bullets_stops_at_next_version() -> None:
    assert unreleased_bullets(BASE) == {"- existing entry"}
