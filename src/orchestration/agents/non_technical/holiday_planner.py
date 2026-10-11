from dataclasses import dataclass

from pydantic_ai import Agent

from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class HolidayPlannerDeps:
    agent_name: str = "holiday_planner"


planner_prompt = """You are a Holiday Planner Agent.
Your job is to coordinate travel plans, organize itineraries, and ensure a seamless travel experience for the user.
"""

holiday_planner_agent = Agent(
    create_model(settings.resolve_model(settings.holiday_planner_model)),
    deps_type=HolidayPlannerDeps,
    system_prompt=planner_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
