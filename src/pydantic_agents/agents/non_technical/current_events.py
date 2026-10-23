from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class CurrentEventsDeps:
    agent_name: str = "current_events"


events_prompt = """You are a Current Events Agent and a history buff.
Your job is to provide context on current events, blending modern news with historical perspectives.
"""

current_events_agent = Agent(
    create_model(settings.resolve_model(settings.current_events_model)),
    deps_type=CurrentEventsDeps,
    system_prompt=events_prompt,
    output_type=str,
    defer_model_check=True,
    retries=settings.agent_retries,
)
