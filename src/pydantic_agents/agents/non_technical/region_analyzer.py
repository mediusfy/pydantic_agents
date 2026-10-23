from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class RegionAnalyzerDeps:
    agent_name: str = "region_analyzer"


analyzer_prompt = f"""You are a Region Analyzer Agent. You analyse regions on safety, cost of living, climate, infrastructure and local
attractions, for visitors and for people considering moving.

- Compare against explicit alternatives or a stated baseline, and say which factors matter most for the user's goal.
- For safety, cite official advisories (government travel advisories) and give their dates; do not generalise about
  groups of people.
- Present a short table of factors with source and confidence, then your conclusion.

{GROUNDING}"""

region_analyzer_agent = Agent(
    create_model(settings.resolve_model(settings.region_analyzer_model)),
    deps_type=RegionAnalyzerDeps,
    system_prompt=analyzer_prompt,
    output_type=str,
    defer_model_check=True,
    retries=settings.agent_retries,
)
