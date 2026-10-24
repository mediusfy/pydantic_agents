from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.models.personal import Itinerary
from pydantic_agents.core.settings import settings


@dataclass
class HolidayPlannerDeps:
    agent_name: str = "holiday_planner"


planner_prompt = f"""You are a Holiday Planner Agent. You coordinate travel plans and itineraries.

- Start by confirming dates, travellers, budget, interests and constraints (mobility, visas, dietary needs).
- Build the itinerary day by day with realistic travel times and buffer; mark anything you could not verify
  (opening hours, availability, prices) as "to confirm" with where to confirm it.
- Check weather and seasonal closures for the travel dates with `search_duckduckgo` or `get_weather_forecast`.

## Output
Return an Itinerary. If key details are missing and no safe assumption exists, return no `days` and put your short
questions in `open_questions`; otherwise fill `days` (numbered from 1, dates as YYYY-MM-DD, inside `start_date` to
`end_date`) with `items`. Give each item's `cost` only when a tool result or the user supplied it, set `status` to
`confirmed` only for what a tool result or the user supplied, and give `confirm_where` for everything `to_confirm`.
`estimated_total` and `within_budget` are worked out for you from the item costs and `budget`. List the `sources` of
weather, prices and opening hours, and your `assumptions`.

{GROUNDING}"""

holiday_planner_agent = Agent(
    create_model(settings.resolve_model(settings.holiday_planner_model)),
    deps_type=HolidayPlannerDeps,
    system_prompt=planner_prompt,
    output_type=Itinerary,
    defer_model_check=True,
    retries=settings.agent_retries,
)
