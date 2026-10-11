from dataclasses import dataclass

from pydantic_ai import Agent

from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class AccountantDeps:
    agent_name: str = "accountant"


accountant_prompt = """You are an Accountant Agent.
Your job is to manage budgets, calculate expenses, and provide financial advice for travel and general purposes.
"""

accountant_agent = Agent(
    create_model(settings.resolve_model(settings.accountant_model)),
    deps_type=AccountantDeps,
    system_prompt=accountant_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
