"""Deterministic verification of generated code: build, vet and test with the real toolchain.

This is the ``qa_executor`` of the suite: no model is involved. The LLM gates (security, quality gate, review)
receive its report, so their judgement starts from facts instead of from a guess about whether the code compiles.

Two entry points:

- ``verify_artifacts(artifacts)`` writes every file of the given artifacts into one temporary Go module and runs
  ``go build``, ``go vet`` and ``go test`` there. For generated services this is the whole tree.
- ``verify_tree(root, changed_files)`` copies an existing codebase (minus VCS and dependency directories) into a
  temporary directory, overlays the changed files, and runs the same three commands with the module as it is.
  Nothing is written into the real tree.

Both never raise: a missing toolchain, a timeout or an unwritable file name is a failed check with a message.
"""

import asyncio
import contextlib
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_agents.core.tools.go_common import init_and_tidy
from pydantic_agents.core.utils import WRITE_NAME_ERRORS, safe_write, sanitized_subprocess_env

logger = logging.getLogger(__name__)

_ALL_PACKAGES = "./..."
_GO_MOD = "go.mod"
COMMAND_TIMEOUT = 120.0
_OUTPUT_MAX_CHARS = 4_000
_COPY_IGNORE = shutil.ignore_patterns(".git", "node_modules", ".venv", "venv", "vendor", "dist", "build", ".worktrees")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    output: str = ""


@dataclass
class VerificationReport:
    """The outcome of the deterministic checks. ``passed`` is true only when every check that ran passed and at
    least one check ran; ``skipped`` says why nothing ran (no Go files, for example)."""

    checks: list[CheckResult] = field(default_factory=list)
    skipped: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def failures(self) -> list[str]:
        return [f"{c.name}: {c.output}".strip() for c in self.checks if not c.passed]

    def as_prompt(self) -> str:
        """A compact rendering for the LLM gates."""
        if self.skipped:
            return f"VERIFICATION: skipped ({self.skipped})"
        lines = [f"VERIFICATION: {'passed' if self.passed else 'FAILED'}"]
        for c in self.checks:
            lines.append(f"- {c.name}: {'ok' if c.passed else 'FAILED'}")
            if not c.passed and c.output:
                lines.append(c.output[:_OUTPUT_MAX_CHARS])
        return "\n".join(lines)


async def _run(name: str, args: list[str], cwd: str) -> CheckResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=sanitized_subprocess_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return CheckResult(name, False, f"'{args[0]}' is not installed")
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=COMMAND_TIMEOUT)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return CheckResult(name, False, f"'{' '.join(args)}' timed out after {COMMAND_TIMEOUT:.0f}s")
    output = f"{stdout.decode(errors='replace')}\n{stderr.decode(errors='replace')}".strip()
    return CheckResult(name, proc.returncode == 0, output[-_OUTPUT_MAX_CHARS:])


async def _go_checks(cwd: str, has_tests: bool) -> list[CheckResult]:
    checks = [await _run("go build", ["go", "build", _ALL_PACKAGES], cwd)]
    if not checks[0].passed:
        return checks  # vet and test would only repeat the compiler errors
    checks.append(await _run("go vet", ["go", "vet", _ALL_PACKAGES], cwd))
    if has_tests:
        checks.append(await _run("go test", ["go", "test", _ALL_PACKAGES], cwd))
    return checks


def _go_files(files: dict[str, str]) -> dict[str, str]:
    return {name: content for name, content in files.items() if name.endswith(".go")}


async def verify_artifacts(artifacts: list[Any]) -> VerificationReport:
    """Build, vet and test the Go files of the given artifacts in a fresh module."""
    files: dict[str, str] = {}
    for art in artifacts:
        files.update(getattr(art, "files", None) or {})
    go_files = _go_files(files)
    if not go_files:
        return VerificationReport(skipped="no Go files to verify")

    report = VerificationReport()
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            for name, content in go_files.items():
                await asyncio.to_thread(safe_write, temp_dir, name, content)
        except WRITE_NAME_ERRORS as e:
            report.checks.append(CheckResult("write files", False, str(e)))
            return report
        if not any(name == _GO_MOD or name.endswith(f"/{_GO_MOD}") for name in go_files):
            err = await init_and_tidy(temp_dir)
            if err:
                report.checks.append(CheckResult("go mod", False, err))
                return report
        report.checks.extend(await _go_checks(temp_dir, any(n.endswith("_test.go") for n in go_files)))
    return report


async def verify_tree(root: str, changed_files: dict[str, str]) -> VerificationReport:
    """Overlay the changed files on a copy of the codebase at ``root`` and build, vet and test the copy."""
    source = Path(root)
    if not source.is_dir():
        return VerificationReport(skipped=f"codebase root {root!r} is not a directory")
    if not _go_files(changed_files) and not (source / _GO_MOD).exists():
        return VerificationReport(skipped="no Go module and no Go files to verify")

    report = VerificationReport()
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "tree"
        try:
            await asyncio.to_thread(shutil.copytree, source, target, ignore=_COPY_IGNORE, symlinks=True)
            for name, content in changed_files.items():
                await asyncio.to_thread(safe_write, target, name, content)
        except WRITE_NAME_ERRORS as e:  # OSError is among them
            report.checks.append(CheckResult("prepare tree", False, str(e)))
            return report
        if not (target / _GO_MOD).exists():
            err = await init_and_tidy(str(target))
            if err:
                report.checks.append(CheckResult("go mod", False, err))
                return report
        report.checks.extend(await _go_checks(str(target), True))
    return report
