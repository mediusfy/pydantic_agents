from pydantic_agents.core.adapters.agents import AgentProviderPort
from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig


class StageRunner:
    def __init__(self, agent_provider: AgentProviderPort | None = None):
        self.agent_provider = agent_provider

    @property
    def provider(self) -> AgentProviderPort:
        if self.agent_provider is None:
            raise RuntimeError("AgentProviderPort is not configured")
        return self.agent_provider

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        """
        Executes a pipeline stage taking the current state and returning the updated state.
        Must be implemented by all stage handlers.
        """
        raise NotImplementedError("StageRunner must implement run()")
