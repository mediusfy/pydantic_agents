"""CodeValidator: what each check reports and whether it fails the run."""

import pytest

from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.validators import CodeValidator


def _art(**files: str) -> Artifact:
    return Artifact(agent_name="test", files={name.replace("__", "."): body for name, body in files.items()})


GOOD_GO = "package main\n\nfunc main() {}\n"
GOOD_VUE = "<template><div/></template>\n<script setup>\nexport default {}\n</script>\n"


# --- Go -----------------------------------------------------------------------------------------------------------


async def test_valid_go_passes_and_says_so(capsys):
    assert await CodeValidator.validate_go_async([_art(main__go=GOOD_GO)]) is True
    assert capsys.readouterr().out == "[Validator] Go structural validation passed\n"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("   \n", "empty file"),
        ("func main() {}\n", "missing 'package' declaration"),
        ("package main\nfunc main() {\n", "unbalanced braces (1 open vs 0 close)"),
        ("package main\nfunc main() {}}\n", "unbalanced braces (1 open vs 2 close)"),
    ],
    ids=["empty", "no-package", "missing-close", "extra-close"],
)
async def test_invalid_go_fails_with_its_reason(capsys, content, message):
    assert await CodeValidator.validate_go_async([_art(bad__go=content)]) is False
    assert capsys.readouterr().out == f"[Validator] bad.go: {message}\n"


async def test_every_bad_go_file_is_reported_and_other_files_are_ignored(capsys):
    artifacts = [_art(a__go="", notes__txt="not go"), _art(b__go="func x() {}", ok__go=GOOD_GO)]

    assert await CodeValidator.validate_go_async(artifacts) is False
    out = capsys.readouterr().out
    assert "[Validator] a.go: empty file\n" in out
    assert "[Validator] b.go: missing 'package' declaration\n" in out
    assert "ok.go" not in out
    assert "notes.txt" not in out
    assert "passed" not in out


async def test_a_package_declaration_may_be_indented(capsys):
    assert await CodeValidator.validate_go_async([_art(x__go="  package x\n")]) is True


# --- Vue / TSX ----------------------------------------------------------------------------------------------------


async def test_valid_vue_passes_and_says_so(capsys):
    assert await CodeValidator.validate_vue_async([_art(App__vue=GOOD_VUE)]) is True
    assert capsys.readouterr().out == "[Validator] Vue/TSX structural validation passed\n"


@pytest.mark.parametrize(
    ("name", "content", "messages", "ok"),
    [
        ("A.vue", "<template><div/>\n<script setup>export default {}</script>", ["missing closing </template>"], False),
        ("A.vue", "<template><div/></template>\n<script>export default {}", ["missing closing </script>"], False),
        (
            "A.vue",
            "<template><div>\n<script>x",
            ["missing closing </template>", "missing closing </script>", "SFC may be missing component definition"],
            False,
        ),
        ("A.vue", "<template><div/></template>", ["SFC may be missing component definition"], True),
        ("A.tsx", "const A = () => null;", ["missing component export"], False),
        ("A.tsx", "export const A = () => null;", [], True),
        ("A.tsx", "export default function A() { return null }", [], True),
        ("A.vue", "<script>const c = defineComponent({})</script>", [], True),
        ("A.vue", "<script setup>const emit = defineEmits(['x'])</script>", [], True),
    ],
    ids=[
        "template-unclosed",
        "script-unclosed",
        "all-three-problems",
        "sfc-without-definition-only-warns",
        "tsx-without-export",
        "tsx-named-export",
        "tsx-default-export",
        "define-component",
        "define-emits",
    ],
)
async def test_vue_and_tsx_checks(capsys, name, content, messages, ok):
    artifact = Artifact(agent_name="test", files={name: content})

    assert await CodeValidator.validate_vue_async([artifact]) is ok

    printed = [line for line in capsys.readouterr().out.splitlines() if "structural validation passed" not in line]
    assert printed == [f"[Validator] {name}: {m}" for m in messages]


async def test_a_tsx_export_const_does_not_satisfy_a_vue_file(capsys):
    artifact = Artifact(agent_name="test", files={"A.vue": "<template><i/></template>\nexport const x = 1"})

    assert await CodeValidator.validate_vue_async([artifact]) is True  # only a warning
    assert "SFC may be missing component definition" in capsys.readouterr().out


async def test_non_vue_files_are_not_checked(capsys):
    assert await CodeValidator.validate_vue_async([_art(main__go="x", style__css="y")]) is True
    assert capsys.readouterr().out == ""
