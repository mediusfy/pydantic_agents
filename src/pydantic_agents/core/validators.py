import asyncio

from pydantic_agents.core.models.core import Artifact


class CodeValidator:
    """Validates generated code for basic structural correctness."""

    @staticmethod
    async def validate_go_async(artifacts: list[Artifact]) -> bool:
        """Validate Go files: check for basic syntactic correctness.

        Uses simple structural checks (balanced braces, presence of 'package')
        instead of running `go vet`, which requires a complete module graph
        that generated code fragments rarely have.
        """
        go_files = []
        for art in artifacts:
            for path, content in art.files.items():
                if path.endswith(".go"):
                    go_files.append((path, content))

        if not go_files:
            return True

        all_ok = True
        for path, content in go_files:
            if not content.strip():
                print(f"[Validator] {path}: empty file")
                all_ok = False
                continue

            lines = content.splitlines()
            has_package = any(line.strip().startswith("package ") for line in lines)
            if not has_package:
                print(f"[Validator] {path}: missing 'package' declaration")
                all_ok = False
                continue

            open_braces = content.count("{")
            close_braces = content.count("}")
            if open_braces != close_braces:
                print(f"[Validator] {path}: unbalanced braces ({open_braces} open vs {close_braces} close)")
                all_ok = False

        if all_ok:
            print("[Validator] Go structural validation passed")
        return all_ok

    @staticmethod
    async def validate_vue_async(artifacts: list[Artifact]) -> bool:
        """Validate Vue/TSX files for basic structural correctness."""
        vue_files = []
        for art in artifacts:
            for path, content in art.files.items():
                if path.endswith((".vue", ".tsx")):
                    vue_files.append((path, content))

        if not vue_files:
            return True

        all_ok = True
        for path, content in vue_files:
            if "<template>" in content and "</template>" not in content:
                print(f"[Validator] {path}: missing closing </template>")
                all_ok = False
            if "<script" in content and "</script>" not in content:
                print(f"[Validator] {path}: missing closing </script>")
                all_ok = False
            # Vue SFCs: check for component export in script block or <script setup>
            if path.endswith(".vue"):
                has_export = "export default" in content or "defineComponent" in content or "defineEmits" in content
            else:
                has_export = "export default" in content or "export const" in content
            if not has_export:
                if path.endswith(".vue"):
                    print(f"[Validator] {path}: SFC may be missing component definition")
                else:
                    print(f"[Validator] {path}: missing component export")
                    all_ok = False

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
