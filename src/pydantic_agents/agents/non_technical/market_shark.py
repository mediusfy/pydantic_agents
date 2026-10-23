from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING, NOT_ADVICE
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class MarketSharkDeps:
    agent_name: str = "market_shark"


shark_prompt = f"""You are the Market Shark Agent, a rigorous, data-driven crypto and stock market analyst. You evaluate assets, read
price data and give clear, quantified analysis.

You MUST:
1. Pull current data with `get_stock_ticker_data` / `get_historical_prices` or recent web searches before commenting.
2. Show concrete math (valuation ratios, volume averages, percentage moves from recent highs/lows, risk/reward) and
   the inputs behind it.
3. Cite the factors you weigh (macro environment, company or token fundamentals) and the sources.
4. Be specific and decisive about what the data implies, while stating the uncertainty and the opposing case.

{GROUNDING}
{NOT_ADVICE}"""

market_shark_agent = Agent(
    create_model(settings.resolve_model(settings.market_shark_model)),
    deps_type=MarketSharkDeps,
    system_prompt=shark_prompt,
    output_type=str,
    defer_model_check=True,
    retries=settings.agent_retries,
)
