import json
import logging
import os

from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.writer import ArtifactWriter

logger = logging.getLogger(__name__)


class WorkspaceContext:
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir

    def get_repo_context(self) -> str:
        context_lines = []
        package_path = os.path.join(self.root_dir, "package.json")
        mod_path = os.path.join(self.root_dir, "go.mod")

        if os.path.exists(package_path):
            try:
                with open(package_path) as f:
                    data = json.load(f)
                    deps = list(data.get("dependencies", {}).keys()) + list(data.get("devDependencies", {}).keys())
                    if deps:
                        context_lines.append(f"JS/TS Dependencies: {', '.join(deps)}")
            except Exception:
                pass

        if os.path.exists(mod_path):
            try:
                with open(mod_path) as f:
                    deps = [line.strip().split()[0] for line in f if "require" in line or line.startswith("\t")]
                    if deps:
                        context_lines.append(f"Go Dependencies: {', '.join(deps[:10])}...")
            except Exception:
                pass

        def _tree(path, max_depth=2):
            tree = []
            if not os.path.exists(path):
                return ""
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

        pkg_tree = _tree(os.path.join(self.root_dir, "pkg"))
        if pkg_tree:
            context_lines.append("Existing Go Packages (pkg/):\n" + pkg_tree)

        components_tree = _tree(os.path.join(self.root_dir, "src/components"))
        if components_tree:
            context_lines.append("Existing Frontend Components (src/components/):\n" + components_tree)

        if not context_lines:
            return "REPOSITORY CONTEXT: No existing repository context found."

        return "REPOSITORY CONTEXT:\n" + "\n\n".join(context_lines)

    def get_markdown_context(self) -> str:
        """
        Reads local markdown files (AGENTS.md, CODING_STANDARDS.md, and adrs/*.md)
        to dynamically hydrate the agent prompts with repository-specific rules.
        """
        context_blocks = []

        # Core standard files
        for filename in ["AGENTS.md", "CODING_STANDARDS.md"]:
            filepath = os.path.join(self.root_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath) as f:
                        content = f.read().strip()
                        if content:
                            context_blocks.append(f"### {filename}\n{content}")
                except Exception as e:
                    logger.warning(f"Failed to read {filename}: {e}")

        # ADRs (Architectural Decision Records)
        adrs_dir = os.path.join(self.root_dir, "adrs")
        if os.path.exists(adrs_dir) and os.path.isdir(adrs_dir):
            adr_contents = []
            for filename in sorted(os.listdir(adrs_dir)):
                if filename.endswith(".md"):
                    filepath = os.path.join(adrs_dir, filename)
                    try:
                        with open(filepath) as f:
                            content = f.read().strip()
                            if content:
                                adr_contents.append(f"==== {filename} ====\n{content}")
                    except Exception as e:
                        logger.warning(f"Failed to read ADR {filename}: {e}")

            if adr_contents:
                context_blocks.append("### Architectural Decision Records (ADRs)\n" + "\n\n".join(adr_contents))

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
