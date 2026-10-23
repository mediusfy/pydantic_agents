from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic_ai import RunContext
from tenacity import retry, stop_after_attempt, wait_exponential

from pydantic_agents.core.tools.registry import register_tool


async def get_current_datetime(ctx: RunContext[Any]) -> str:
    """Get the current date and time in UTC. Crucial for understanding 'now' in pricing or travel."""
    now = datetime.now(UTC)
    return f"Current UTC Date and Time: {now.isoformat()}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def search_duckduckgo(ctx: RunContext[Any], query: str, max_results: int = 5) -> str:
    """Perform a DuckDuckGo web search to find current pricing, news, or travel info."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return "No results found."

            formatted_results = []
            for i, res in enumerate(results, 1):
                formatted_results.append(
                    f"Result {i}:\nTitle: {res.get('title')}\nURL: {res.get('href')}\nSnippet: {res.get('body')}"
                )

            return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Web search failed: {e}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def search_news(ctx: RunContext[Any], query: str, max_results: int = 5) -> str:
    """Perform a DuckDuckGo News search specifically for recent articles, press releases, or legislation."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))

            if not results:
                return "No news found."

            formatted_results = []
            for i, res in enumerate(results, 1):
                formatted_results.append(
                    f"Result {i}:\nDate: {res.get('date')}\nTitle: {res.get('title')}\nSource: {res.get('source')}\nSnippet: {res.get('body')}"
                )

            return "\n\n".join(formatted_results)
    except Exception as e:
        return f"News search failed: {e}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_weather_forecast(ctx: RunContext[Any], lat: float, lon: float) -> str:
    """Get the 7-day weather forecast for a location (latitude, longitude) using Open-Meteo."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode&timezone=auto"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            if not daily:
                return "No daily forecast data available."

            forecasts = []
            for i in range(len(daily.get("time", []))):
                date = daily["time"][i]
                tmax = daily["temperature_2m_max"][i]
                tmin = daily["temperature_2m_min"][i]
                precip = daily["precipitation_sum"][i]
                forecasts.append(f"{date}: Max {tmax}C, Min {tmin}C, Precip {precip}mm")
            return "\n".join(forecasts)
    except Exception as e:
        return f"Weather fetch failed: {e}"


register_tool(
    "get_current_datetime",
    get_current_datetime,
    "Get current date and time",
    ["web", "search", "planning", "pricing", "news"],
)
register_tool(
    "search_duckduckgo",
    search_duckduckgo,
    "Search the web for current events, prices, and info",
    ["web", "search", "planning", "pricing", "news"],
)
register_tool(
    "search_news",
    search_news,
    "Search news articles and recent events",
    ["web", "search", "planning", "pricing", "news"],
)
register_tool("get_weather_forecast", get_weather_forecast, "Get weather forecast by lat/lon", ["web", "planning"])
