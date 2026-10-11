from dataclasses import dataclass

from pydantic_ai import Agent

from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class WeatherDeps:
    agent_name: str = "weather"


weather_prompt = """You are a Weather Agent.
Your job is to provide accurate weather forecasts, climate summaries, and travel-related weather advice.
"""

weather_agent = Agent(
    create_model(settings.resolve_model(settings.weather_model)),
    deps_type=WeatherDeps,
    system_prompt=weather_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
