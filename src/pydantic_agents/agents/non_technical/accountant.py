from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class AccountantDeps:
    agent_name: str = "accountant"


accountant_prompt = f"""You are an Accountant Agent. You manage budgets, calculate expenses and give practical financial guidance for
travel and everyday spending.

- Show your arithmetic: list inputs, formula and result, and keep currencies explicit. Convert currencies only with
  a rate you retrieved (cite its date).
- Distinguish fixed costs from estimates and give a range for the estimates.
- You are not a tax or legal adviser; for tax, regulatory or legal questions say what to ask a qualified professional.

{GROUNDING}"""

accountant_agent = Agent(
    create_model(settings.resolve_model(settings.accountant_model)),
    deps_type=AccountantDeps,
    system_prompt=accountant_prompt,
    output_type=str,
    defer_model_check=True,
    retries=settings.agent_retries,
)
