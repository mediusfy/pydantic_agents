from dataclasses import dataclass

from pydantic_ai import Agent

from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class PriceAnalystDeps:
    agent_name: str = "price_analyst"


analyst_prompt = """You are the Price Analyst Agent.
Your primary role is to evaluate the past, present, and projected future prices of items.
You employ abstract, analytical thinking to assess macroeconomic variables, historical trends, seasonality, inflation, supply chain constraints, and market sentiment.
Your goal is to definitively advise the user on the optimal time to purchase a given item, maximizing value while minimizing cost.
Be extremely analytical, citing variables you are considering, calculating projected depreciation or appreciation, and factoring in opportunity costs where relevant.
"""

price_analyst_agent = Agent(
    create_model(settings.resolve_model(settings.price_analyst_model)),
    deps_type=PriceAnalystDeps,
    system_prompt=analyst_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
