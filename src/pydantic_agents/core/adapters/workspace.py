import json
import logging
import os

from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.writer import ArtifactWriter

logger = logging.getLogger(__name__)


def _tree(path: str, max_depth: int = 2) -> str:
    """An indented listing of ``path`` (at most ``max_depth`` levels, ten files per directory); "" if it is missing."""
    if not os.path.exists(path):
        return ""
    tree = []
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        if level >= max_depth:
            dirs[:] = []
            continue
        indent = " " * 2 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for f in files[:10]:
            tree.append(f"{subindent}{f}")
    return "\n".join(tree)


def _labelled_tree(label: str, path: str) -> str | None:
    """``label`` followed by the directory tree, or None when there is nothing to show."""
    tree = _tree(path)
    return f"{label}\n{tree}" if tree else None


def _read_text(filepath: str, description: str) -> str:
    """The stripped contents of a file; "" (and a warning) if it cannot be read."""
    try:
        with open(filepath) as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Failed to read {description}: {e}")
        return ""


class WorkspaceContext:
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir

    def _js_dependencies(self) -> str | None:
        path = os.path.join(self.root_dir, "package.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            deps = [*data.get("dependencies", {}), *data.get("devDependencies", {})]
        except Exception:
            return None
        return f"JS/TS Dependencies: {', '.join(deps)}" if deps else None

    def _go_dependencies(self) -> str | None:
        path = os.path.join(self.root_dir, "go.mod")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                deps = [line.strip().split()[0] for line in f if "require" in line or line.startswith("\t")]
        except Exception:
            return None
        return f"Go Dependencies: {', '.join(deps[:10])}..." if deps else None

    def get_repo_context(self) -> str:
        sections = [
            self._js_dependencies(),
            self._go_dependencies(),
            _labelled_tree("Existing Go Packages (pkg/):", os.path.join(self.root_dir, "pkg")),
            _labelled_tree(
                "Existing Frontend Components (src/components/):", os.path.join(self.root_dir, "src/components")
            ),
        ]
        context_lines = [section for section in sections if section]

        if not context_lines:
            return "REPOSITORY CONTEXT: No existing repository context found."

        return "REPOSITORY CONTEXT:\n" + "\n\n".join(context_lines)

    def _standard_file_blocks(self) -> list[str]:
        blocks = []
        for filename in ["AGENTS.md", "CODING_STANDARDS.md"]:
            filepath = os.path.join(self.root_dir, filename)
            if not os.path.exists(filepath):
                continue
            content = _read_text(filepath, filename)
            if content:
                blocks.append(f"### {filename}\n{content}")
        return blocks

    def _adr_block(self) -> str | None:
        adrs_dir = os.path.join(self.root_dir, "adrs")
        if not os.path.isdir(adrs_dir):
            return None
        contents = []
        for filename in sorted(os.listdir(adrs_dir)):
            if not filename.endswith(".md"):
                continue
            content = _read_text(os.path.join(adrs_dir, filename), f"ADR {filename}")
            if content:
                contents.append(f"==== {filename} ====\n{content}")
        if not contents:
            return None
        return "### Architectural Decision Records (ADRs)\n" + "\n\n".join(contents)

    def get_markdown_context(self) -> str:
        """
        Reads local markdown files (AGENTS.md, CODING_STANDARDS.md, and adrs/*.md)
        to dynamically hydrate the agent prompts with repository-specific rules.
        """
        context_blocks = self._standard_file_blocks()
        adr_block = self._adr_block()
        if adr_block:
            context_blocks.append(adr_block)

        if not context_blocks:
            return ""

        return "LOCAL REPOSITORY GUIDELINES:\n" + "\n\n".join(context_blocks)


class ArtifactStorageAdapter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def write_artifacts(self, artifacts: list[Artifact]):
        if not artifacts:
            logger.warning("No artifacts to write.")
            return

        logger.info(f"Writing {len(artifacts)} artifacts to {self.output_dir}...")
        writer = ArtifactWriter(self.output_dir)
        for art in artifacts:
            writer.write(art)
        logger.info("Pipeline complete. Artifacts written to disk.")
