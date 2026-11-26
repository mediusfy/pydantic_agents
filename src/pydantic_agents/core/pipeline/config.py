"""Pipeline configuration: blueprint catalog and per-run config dataclass."""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel


class Framework(str, Enum):  # noqa: UP042
    LEGACY = "legacy"
    LANGGRAPH = "langgraph"


class StageName(str, Enum):  # noqa: UP042
    BA = "ba"
    ADVISORY = "advisory"
    IMPLEMENTATION = "implementation"
    QUALITY_GATE = "quality_gate"
    MERGE = "merge"


STAGES = StageName


class BlueprintSpec(BaseModel):
    name: str
    description: str
    framework: str = "legacy"
    stages: list[str] | None = None

    def model_post_init(self, _context: Any) -> None:
        self._validate_fields()

    def _validate_fields(self) -> None:
        if self.framework not in ("legacy", "langgraph"):
            raise ValueError(
                f"Blueprint '{self.name}' uses retired/unsupported framework {self.framework!r}; "
                "supported frameworks are 'legacy' and 'langgraph'."
            )
        if not self.stages:
            raise ValueError(f"Blueprint '{self.name}' (framework={self.framework}) requires 'stages'")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"description": self.description, "framework": self.framework}
        if self.stages is not None:
            d["stages"] = self.stages
        return d


# ── Project → Target Directory Mappings ───────────────────────────────────
PROJECT_ROOTS: dict[str, str] = {
    "default": ".",
    "rufkis": "/Users/jocham/programming/projects/mediusfy/platform/kube/view",
}


# ── Blueprint Catalog ─────────────────────────────────────────────────────
# Standardized on pydantic_ai + OrgitrationPipeline ("legacy") and LangGraph ("langgraph").
_BLUEPRINT_SPECS: list[BlueprintSpec] = [
    BlueprintSpec(
        name="new_feature",
        description="Full SDLC: BA → Advisory → Implementation → Quality Gate",
        stages=[STAGES.BA, STAGES.ADVISORY, STAGES.IMPLEMENTATION, STAGES.QUALITY_GATE],
        framework=Framework.LEGACY,
    ),
    BlueprintSpec(
        name="quick_backend",
        description="Backend-only: Implementation → Review",
        stages=[STAGES.IMPLEMENTATION, STAGES.QUALITY_GATE],
        framework=Framework.LEGACY,
    ),
    BlueprintSpec(
        name="docs_update",
        description="Documentation: Implementation (docs agent) → Review",
        stages=[STAGES.IMPLEMENTATION, STAGES.QUALITY_GATE],
        framework=Framework.LEGACY,
    ),
    BlueprintSpec(
        name="security_review",
        description="Security audit: Review (security agent)",
        stages=[STAGES.QUALITY_GATE],
        framework=Framework.LEGACY,
    ),
    BlueprintSpec(
        name="langgraph_full",
        description="Full SDLC via LangGraph state machine with retry loops",
        stages=[STAGES.BA, STAGES.ADVISORY, STAGES.IMPLEMENTATION, STAGES.QUALITY_GATE],
        framework=Framework.LANGGRAPH,
    ),
    BlueprintSpec(
        name="langgraph_quick",
        description="Backend-only via LangGraph with test-retry cycle",
        stages=[STAGES.IMPLEMENTATION, STAGES.QUALITY_GATE],
        framework=Framework.LANGGRAPH,
    ),
]

BLUEPRINTS: dict[str, dict[str, Any]] = {bp.name: bp.to_dict() for bp in _BLUEPRINT_SPECS}


# The project name becomes a directory under the output, checkpoint and approval roots, so it must be a plain name.
_PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SUPPORTED_FRAMEWORKS = frozenset({"legacy", "langgraph"})


@dataclass
class PipelineConfig:
    project: str = "default"
    step: str = "all"
    blueprint: str = "new_feature"
    framework: str = "legacy"
    resume: bool = False
    auto_yes: bool = False
    dry_run: bool = False
    requirement: str | None = None
    target_root: str = field(default=".")
    # The run's id (also its checkpoint key); None lets the orchestrator mint one.
    run_id: str | None = None

    blueprints: ClassVar[dict[str, dict[str, Any]]] = BLUEPRINTS

    def __post_init__(self):
        if not _PROJECT_NAME.fullmatch(self.project) or ".." in self.project:
            raise ValueError(
                f"Invalid project name {self.project!r}: letters, digits, '_', '.' and '-' only (at most 64 characters)"
            )
        self.target_root = PROJECT_ROOTS.get(self.project, self.project)
        self._validate_blueprint()

    def _validate_blueprint(self):
        if self.blueprint not in BLUEPRINTS:
            raise ValueError(f"Unknown blueprint '{self.blueprint}'. Available: {', '.join(BLUEPRINTS)}")
        if self.framework not in _SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported pipeline framework {self.framework!r}; supported frameworks are 'legacy' and 'langgraph'."
            )
        bp = BLUEPRINTS[self.blueprint]
        expected = bp.get("framework", "legacy")
        if self.framework == "legacy" and expected != "legacy":
            self.framework = expected
            return
        if self.framework != "legacy" and self.framework != expected:
            import logging

            logging.getLogger(__name__).warning(
                "Blueprint '%s' expects framework '%s' but '%s' was requested. Using '%s'.",
                self.blueprint,
                expected,
                self.framework,
                self.framework,
            )

    def output_dir(self, base: Path = Path("./generated-app")) -> Path:
        return base / self.project
