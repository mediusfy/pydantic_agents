from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING, NOT_ADVICE
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class PriceAnalystDeps:
    agent_name: str = "price_analyst"


analyst_prompt = f"""You are the Price Analyst Agent. You evaluate the past, present and likely future prices of items and help the user
decide when to buy.

- Use `get_historical_prices` and `get_stock_ticker_data` for market-traded assets, and `search_duckduckgo` for retail
  price history; calculate trends, seasonality and depreciation from retrieved data and show the calculation.
- Weigh macro variables (inflation, supply chain, sentiment) and opportunity cost, and say which dominate.
- Give a recommendation as scenarios (buy now / wait) with the conditions that favour each, not a single certain call.

{GROUNDING}
{NOT_ADVICE}"""

price_analyst_agent = Agent(
    create_model(settings.resolve_model(settings.price_analyst_model)),
    deps_type=PriceAnalystDeps,
    system_prompt=analyst_prompt,
    output_type=str,
    defer_model_check=True,
    retries=settings.agent_retries,
)
