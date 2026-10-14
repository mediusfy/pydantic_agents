"""Pipeline protocol — shared interface for all framework orchestrators."""

from typing import Protocol, runtime_checkable

from pydantic_agents.core.models.core import PipelineState


@runtime_checkable
class PipelineOrchestrator(Protocol):
    """Every pipeline orchestrator must implement this protocol.

    LangGraph graphs are wrapped in ``_GraphPipelineWrapper`` (factory.py)
    to conform to this protocol. CrewAI and AutoGen pipelines implement
    it directly.
    """

    async def execute(self) -> PipelineState: ...
