"""parse_code_blocks_to_files: file names from the lines above a fence or an inline path comment."""

import pytest

from pydantic_agents.core.utils import parse_code_blocks_to_files as parse


def _block(body: str, lang: str = "go") -> str:
    return f"```{lang}\n{body}\n```"


@pytest.mark.parametrize(
    "header",
    [
        "backend/main.go",
        "### backend/main.go",
        "**backend/main.go**",
        "`backend/main.go`",
        "backend/main.go:",
        "  backend/main.go  ",
    ],
)
def test_a_file_name_directly_above_the_fence(header):
    assert parse(f"{header}\n{_block('package main')}\n") == {"backend/main.go": "package main"}


@pytest.mark.parametrize(
    ("above", "found"),
    [
        ("main.go\n", True),  # directly above
        ("main.go\n\n", True),  # one blank line between
        ("main.go\n\n\n", True),  # two blank lines between (three lines up)
        ("main.go\n\n\n\n", False),  # four lines up is too far
    ],
)
def test_the_name_may_be_up_to_three_lines_above(above, found):
    text = f"{above}{_block('x')}\n"

    assert parse(text) == ({"main.go": "x"} if found else {"output.txt": text})


def test_the_nearest_candidate_wins():
    text = f"far.go\nnear.go\n{_block('x')}\n"

    assert parse(text) == {"near.go": "x"}


@pytest.mark.parametrize(
    "prose",
    ["See main.go for details", "x" * 100 + ".go", "no dots or slashes here", ""],
    ids=["has-spaces", "too-long", "no-dot-or-slash", "blank"],
)
def test_lines_that_do_not_look_like_a_file_name_are_ignored(prose):
    text = f"{prose}\n{_block('x')}\n"

    assert parse(text) == {"output.txt": text}


def test_markup_after_a_heading_marker_stays_in_the_name():
    # Only the outer run of * ` # : is stripped, so a "## " heading containing bold text keeps its asterisks.
    assert parse(f"## **`backend/main.go`**:\n{_block('x')}\n") == {"**`backend/main.go": "x"}


def test_a_dotted_word_counts_as_a_file_name():
    assert parse(f"v1.2\n{_block('x')}\n") == {"v1.2": "x"}


@pytest.mark.parametrize(
    ("first_line", "name"),
    [
        ("// cmd/server/main.go", "cmd/server/main.go"),
        ("# scripts/run.py", "scripts/run.py"),
        ("/* styles/app.css */", "styles/app.css"),
        ("<!-- web/index.html -->", "web/index.html"),
        ("//   spaced.go", "spaced.go"),
    ],
)
def test_an_inline_path_comment_names_the_file_and_is_not_part_of_it(first_line, name):
    block = _block(first_line + "\nbody line", lang="text")

    assert parse(block) == {name: "body line"}


def test_an_inline_comment_is_kept_when_a_header_already_named_the_file():
    body = "// other.go\nbody"

    assert parse(f"a.go\n{_block(body)}\n") == {"a.go": body}


@pytest.mark.parametrize(
    "first_line",
    [
        "// path: svc/a.go",
        "// just a comment with words",
        "package main",
        "",
        "  // not at start of the block? yes it is",
    ],
)
def test_other_first_lines_are_not_file_names(first_line):
    text = _block(first_line + "\nmore")

    assert parse(text) == {"output.txt": text}


def test_only_the_first_line_of_a_block_can_name_the_file():
    text = _block("package main\n// late/name.go")

    assert parse(text) == {"output.txt": text}


def test_a_blank_first_line_ends_the_chance_of_an_inline_name():
    text = _block("\n// late/name.go\nbody")

    assert parse(text) == {"output.txt": text}


def test_several_files_in_order_with_prose_between():
    text = (
        "Here is the backend:\n"
        f"### backend/main.go\n{_block('package main')}\n"
        "And the frontend:\n"
        f"**src/App.vue**\n{_block('<template/>', lang='vue')}\n"
    )

    parsed = parse(text)

    assert parsed == {"backend/main.go": "package main", "src/App.vue": "<template/>"}
    assert list(parsed) == ["backend/main.go", "src/App.vue"]


def test_a_repeated_name_keeps_the_last_content():
    text = f"a.go\n{_block('first')}\na.go\n{_block('second')}\n"

    assert parse(text) == {"a.go": "second"}


def test_unnamed_blocks_are_dropped_when_another_block_is_named():
    text = f"{_block('unnamed')}\nkept.go\n{_block('kept')}\n"

    assert parse(text) == {"kept.go": "kept"}


def test_content_keeps_its_indentation_and_internal_blank_lines():
    body = "func main() {\n\n\tx := 1\n    y := 2\n}"

    assert parse(f"main.go\n{_block(body)}\n") == {"main.go": body}


def test_an_empty_named_block_gives_an_empty_file():
    assert parse("empty.txt\n```\n```\n") == {"empty.txt": ""}


def test_an_unclosed_block_is_dropped():
    text = f"done.go\n{_block('ok')}\nlate.go\n```go\nnever closed\n"

    assert parse(text) == {"done.go": "ok"}


@pytest.mark.parametrize("text", ["", "no code at all", "just\nprose\nlines"])
def test_text_without_blocks_is_returned_whole_as_output_txt(text):
    assert parse(text) == {"output.txt": text}


def test_any_fence_line_closes_a_block_even_with_a_language_tag():
    text = "a.go\n```go\nfirst\n```python\nsecond\n```\n"

    assert parse(text) == {"a.go": "first"}  # the ```python line closed the block; the rest is a new, unnamed block


def test_fences_may_be_indented():
    assert parse("a.go\n  ```go\n  body\n  ```\n") == {"a.go": "  body"}
