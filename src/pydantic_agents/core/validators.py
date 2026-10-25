import asyncio

from pydantic_agents.core.models.core import Artifact


def _files_ending_with(artifacts: list[Artifact], suffixes: str | tuple[str, ...]) -> list[tuple[str, str]]:
    """The (path, content) of every artifact file whose name ends with one of ``suffixes``."""
    return [(path, content) for art in artifacts for path, content in art.files.items() if path.endswith(suffixes)]


def _go_problem(content: str) -> str | None:
    """Why a Go file fails the structural check, or None when it passes."""
    if not content.strip():
        return "empty file"
    if not any(line.strip().startswith("package ") for line in content.splitlines()):
        return "missing 'package' declaration"
    open_braces = content.count("{")
    close_braces = content.count("}")
    if open_braces != close_braces:
        return f"unbalanced braces ({open_braces} open vs {close_braces} close)"
    return None


def _has_component_export(path: str, content: str) -> bool:
    if path.endswith(".vue"):  # Vue SFCs: a component export in the script block or <script setup>
        return "export default" in content or "defineComponent" in content or "defineEmits" in content
    return "export default" in content or "export const" in content


def _vue_problems(path: str, content: str) -> list[tuple[str, bool]]:
    """The (message, fails the file) problems found in a Vue SFC or TSX file."""
    problems = []
    if "<template>" in content and "</template>" not in content:
        problems.append(("missing closing </template>", True))
    if "<script" in content and "</script>" not in content:
        problems.append(("missing closing </script>", True))
    if not _has_component_export(path, content):
        if path.endswith(".vue"):
            problems.append(("SFC may be missing component definition", False))  # a warning for SFCs only
        else:
            problems.append(("missing component export", True))
    return problems


class CodeValidator:
    """Validates generated code for basic structural correctness."""

    @staticmethod
    async def validate_go_async(artifacts: list[Artifact]) -> bool:
        """Validate Go files: check for basic syntactic correctness.

        Uses simple structural checks (balanced braces, presence of 'package')
        instead of running `go vet`, which requires a complete module graph
        that generated code fragments rarely have.
        """
        go_files = _files_ending_with(artifacts, ".go")
        if not go_files:
            return True

        all_ok = True
        for path, content in go_files:
            problem = _go_problem(content)
            if problem:
                print(f"[Validator] {path}: {problem}")
                all_ok = False

        if all_ok:
            print("[Validator] Go structural validation passed")
        return all_ok

    @staticmethod
    async def validate_vue_async(artifacts: list[Artifact]) -> bool:
        """Validate Vue/TSX files for basic structural correctness."""
        vue_files = _files_ending_with(artifacts, (".vue", ".tsx"))
        if not vue_files:
            return True

        all_ok = True
        for path, content in vue_files:
            for message, fails in _vue_problems(path, content):
                print(f"[Validator] {path}: {message}")
                all_ok = all_ok and not fails

        if all_ok:
            print("[Validator] Vue/TSX structural validation passed")
        return all_ok

    @staticmethod
    async def run_all_async(artifacts: list[Artifact]) -> bool:
        go_result, vue_result = await asyncio.gather(
            CodeValidator.validate_go_async(artifacts),
            CodeValidator.validate_vue_async(artifacts),
        )
        return go_result and vue_result
