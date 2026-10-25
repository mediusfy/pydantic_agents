import pytest

from scripts.check_changelog import _base_changelog, has_new_entry, main, unreleased_bullets

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


@pytest.mark.unit
@pytest.mark.parametrize("ref", ["-h", "--output=/tmp/x", "-p", "a b", "origin/main\nHEAD", "", "$(id)", "a;b"])
def test_base_changelog_refuses_refs_that_could_be_read_as_options(ref: str) -> None:
    with pytest.raises(ValueError, match="not a usable git ref"):
        _base_changelog(ref)


@pytest.mark.unit
@pytest.mark.parametrize("ref", ["origin/main", "HEAD~1", "refs/heads/feat/x", "v1.2.3", "a" * 40, "origin/main~2^"])
def test_base_changelog_accepts_ordinary_refs(ref: str) -> None:
    # A ref that does not exist gives "" rather than an error: the check then treats the base as empty.
    assert isinstance(_base_changelog(ref), str)


@pytest.mark.unit
def test_main_exits_2_for_an_unsafe_ref(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check_changelog.py", "--output=x"]) == 2
    assert "not a usable git ref" in capsys.readouterr().err
