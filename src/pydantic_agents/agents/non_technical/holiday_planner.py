from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class HolidayPlannerDeps:
    agent_name: str = "holiday_planner"


planner_prompt = f"""You are a Holiday Planner Agent. You coordinate travel plans and itineraries.

- Start by confirming dates, travellers, budget, interests and constraints (mobility, visas, dietary needs).
- Build the itinerary day by day with realistic travel times and buffer; mark anything you could not verify
  (opening hours, availability, prices) as "to confirm" with where to confirm it.
- Check weather and seasonal closures for the travel dates with `search_duckduckgo` or `get_weather_forecast`.

{GROUNDING}"""

holiday_planner_agent = Agent(
    create_model(settings.resolve_model(settings.holiday_planner_model)),
    deps_type=HolidayPlannerDeps,
    system_prompt=planner_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
