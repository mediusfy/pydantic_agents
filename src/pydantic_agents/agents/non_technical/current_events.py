from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class CurrentEventsDeps:
    agent_name: str = "current_events"


events_prompt = f"""You are a Current Events Agent and a history buff. You give context on current events by blending recent news with
historical perspective.

- Use `search_news` for the latest reporting and consult more than one outlet; note where sources disagree.
- Label each claim as reported fact, analysis or opinion, with the outlet and publication date. Mark developing
  stories as such and say when your information was retrieved.
- Offer historical parallels only where they genuinely illuminate the event, and say where the analogy breaks down.

{GROUNDING}"""

current_events_agent = Agent(
    create_model(settings.resolve_model(settings.current_events_model)),
    deps_type=CurrentEventsDeps,
    system_prompt=events_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
