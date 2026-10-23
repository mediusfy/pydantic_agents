from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class WeatherDeps:
    agent_name: str = "weather"


weather_prompt = f"""You are a Weather Agent. You provide forecasts, climate summaries and weather-aware travel advice.

- Use `get_weather_forecast` (it needs latitude and longitude; if the user gives a place name, find its coordinates
  with `search_duckduckgo` first and state the location you resolved).
- Report the forecast period, units and the time the data was fetched. Beyond about a week, describe climate
  norms rather than a forecast, and say that you are doing so.

{GROUNDING}"""

weather_agent = Agent(
    create_model(settings.resolve_model(settings.weather_model)),
    deps_type=WeatherDeps,
    system_prompt=weather_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
