"""Fail a pull request that does not add a bullet under ``## [Unreleased]``.

Usage: ``check_changelog.py BASE_REF`` (e.g. ``origin/main``), run from the repo
root with enough history to resolve ``BASE_REF``.
"""

import subprocess
import sys
from pathlib import Path

CHANGELOG = "CHANGELOG.md"
HEADING = "## [Unreleased]"


def unreleased_bullets(text: str) -> set[str]:
    """Return the stripped ``- `` bullet lines inside the Unreleased section."""
    bullets: set[str] = set()
    in_section = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_section = line.strip() == HEADING
        elif in_section and line.lstrip().startswith("- "):
            bullets.add(line.strip())
    return bullets


def has_new_entry(base_text: str, head_text: str) -> bool:
    return bool(unreleased_bullets(head_text) - unreleased_bullets(base_text))


def _base_changelog(base_ref: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{CHANGELOG}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} BASE_REF", file=sys.stderr)
        return 2
    path = Path(CHANGELOG)
    head_text = path.read_text(encoding="utf-8") if path.exists() else ""
    if has_new_entry(_base_changelog(argv[1]), head_text):
        print("changelog: new entry found under [Unreleased]")
        return 0
    print(
        f"::error file={CHANGELOG}::Add a bullet under '{HEADING}' describing this change "
        "(see AGENTS.md). A maintainer can apply the 'skip-changelog' label if nothing needs recording."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
