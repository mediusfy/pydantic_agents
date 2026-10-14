"""Pipeline configuration: blueprint catalog and per-run config dataclass."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel


class Framework(str, Enum):  # noqa: UP042
    LEGACY = "legacy"
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    COMPOSITE = "composite"


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
    crew_class: str | None = None
    group_class: str | None = None

    def model_post_init(self, _context: Any) -> None:
        self._validate_fields()

    def _validate_fields(self) -> None:
        if self.framework in ("legacy", "langgraph", "composite") and not self.stages:
            raise ValueError(f"Blueprint '{self.name}' (framework={self.framework}) requires 'stages'")
        elif self.framework == "crewai" and not self.crew_class:
            raise ValueError(f"Blueprint '{self.name}' requires 'crew_class'")
        elif self.framework == "autogen" and not self.group_class:
            raise ValueError(f"Blueprint '{self.name}' requires 'group_class'")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"description": self.description, "framework": self.framework}
        if self.stages is not None:
            d["stages"] = self.stages
        if self.crew_class is not None:
            d["crew_class"] = self.crew_class
        if self.group_class is not None:
            d["group_class"] = self.group_class
        return d


# ── Project → Target Directory Mappings ───────────────────────────────────
PROJECT_ROOTS: dict[str, str] = {
    "default": ".",
    "rufkis": "/Users/jocham/programming/projects/mediusfy/platform/kube/view",
}


# ── Blueprint Catalog ─────────────────────────────────────────────────────
# Defined as validated BlueprintSpec instances, then converted to a dict for
# backwards-compatible access (BLUEPRINTS[name], .keys(), iteration, ``in``).
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
    BlueprintSpec(
        name="crew_sdlc",
        description="Sequential SDLC via CrewAI role-based agents",
        crew_class="SdlcCrew",
        framework=Framework.CREWAI,
    ),
    BlueprintSpec(
        name="crew_research",
        description="Parallel research via CrewAI: weather + market + crypto",
        crew_class="ResearchCrew",
        framework=Framework.CREWAI,
    ),
    BlueprintSpec(
        name="autogen_sdlc",
        description="Conversational SDLC via AutoGen group chat",
        group_class="SdlcGroupChat",
        framework=Framework.AUTOGEN,
    ),
    BlueprintSpec(
        name="autogen_design_review",
        description="Design discussion via AutoGen: CTO + BA + Security",
        group_class="DesignGroupChat",
        framework=Framework.AUTOGEN,
    ),
    BlueprintSpec(
        name="composite",
        description="LangGraph orchestration with CrewAI implementation",
        stages=[STAGES.BA, STAGES.ADVISORY, STAGES.IMPLEMENTATION, STAGES.QUALITY_GATE],
        framework=Framework.COMPOSITE,
    ),
]

BLUEPRINTS: dict[str, dict[str, Any]] = {bp.name: bp.to_dict() for bp in _BLUEPRINT_SPECS}


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

    blueprints: ClassVar[dict[str, dict[str, Any]]] = BLUEPRINTS

    def __post_init__(self):
        self.target_root = PROJECT_ROOTS.get(self.project, self.project)
        self._validate_blueprint()

    def _validate_blueprint(self):
        if self.blueprint not in BLUEPRINTS:
            raise ValueError(f"Unknown blueprint '{self.blueprint}'. Available: {', '.join(BLUEPRINTS)}")
        bp = BLUEPRINTS[self.blueprint]
        expected = bp.get("framework", "legacy")
        if self.framework == "legacy" and expected != "legacy":
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
