"""Final ``approved`` status of a crew run. Kept apart from ``bridge`` so it imports without the ``crewai`` extra."""

from pydantic_agents.core.models.core import PipelineState

# Crews with no review verdict or test gate of their own: they finish when they finish without errors. A run that
# reports "failed: finished without approval" for a completed research or audit crew was a status bug.
UNGATED_CREWS = frozenset({"ResearchCrew", "AuditCrew"})


def finalize_approval(state: PipelineState, crew_class_name: str) -> None:
    """Set ``approved`` for crews whose output has no verdict (the others set it from their verdict or gate)."""
    if state.failure_report is not None:
        state.approved = False
    elif crew_class_name in UNGATED_CREWS:
        state.approved = not state.errors
    elif crew_class_name == "ArchitectureReviewCrew":
        state.approved = state.cto_approved and not state.errors
