from pathlib import Path

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.utils import safe_path_join


class ArtifactWriter:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def write(self, artifact: Artifact) -> None:
        base_resolved = self.base_dir.resolve()
        agent_dir = safe_path_join(base_resolved, artifact.agent_name)
        if agent_dir.parent != base_resolved or "/" in artifact.agent_name or "\\" in artifact.agent_name:
            raise PathTraversalError(f"Path traversal detected: invalid agent_name {artifact.agent_name!r}")
        for filepath, content in artifact.files.items():
            full_path = safe_path_join(agent_dir, filepath)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
