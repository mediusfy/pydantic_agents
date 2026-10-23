from pydantic_ai import Agent

from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.planning import Plan
from pydantic_agents.core.settings import settings

system_prompt = """You are a Planner Agent. Your job is to decompose high-level goals into a structured,
executable multi-step plan. For each step, identify the required agent capability,
a detailed prompt for that agent, and any task dependencies.

Analyze the goal carefully and break it into the minimal number of discrete steps needed.
Each step must be achievable by a single agent capability.
Dependencies define order: if step B depends on step A's output, list A's task ID in B's dependencies.
"""

planner_agent = Agent(
    create_model(settings.resolve_model(settings.cto_model)),
    system_prompt=system_prompt,
    output_type=Plan,
    defer_model_check=True,
    retries=settings.agent_retries,
)
