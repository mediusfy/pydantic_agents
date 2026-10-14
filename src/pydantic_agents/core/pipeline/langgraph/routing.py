from typing import Literal

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.settings import settings


def route_after_ba(state: PipelineState) -> Literal["advisory", "end"]:
    if state.errors or not state.user_stories:
        return "end"
    return "advisory"


def route_after_advisory(state: PipelineState) -> Literal["implementation", "end"]:
    if not state.cto_approved:
        return "end"
    return "implementation"


def route_after_implementation(state: PipelineState) -> Literal["quality_gate", "retry_implementation", "end"]:
    if state.failure_report:
        if state.failure_report.source == "test_suite":
            return "retry_implementation"
        if state.failure_report.error_type == "runtime_error":
            return "end"
    return "quality_gate"


def route_after_quality_gate(state: PipelineState) -> Literal["retry_implementation", "end"]:
    if state.approved:
        return "end"
    if state.refinement_attempts < settings.max_retries:
        return "retry_implementation"
    return "end"


def route_retry(state: PipelineState) -> Literal["implementation", "end"]:
    if state.refinement_attempts < settings.max_retries:
        return "implementation"
    return "end"


def route_after_implementation_subgraph(state: PipelineState) -> Literal["merge", "end"]:
    if state.approved:
        return "merge"
    return "end"
