import asyncio
from typing import Any

import yfinance as yf
from pydantic_ai import RunContext
from tenacity import retry, stop_after_attempt, wait_exponential

from pydantic_agents.core.tools.registry import register_tool


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_stock_ticker_data(ctx: RunContext[Any], ticker: str) -> str:
    """Get core financial data, current price, and company info for a given stock or crypto ticker symbol (e.g., AAPL, BTC-USD)."""
    try:

        def fetch_info():
            asset = yf.Ticker(ticker)
            return asset.info, asset

        info, asset = await asyncio.to_thread(fetch_info)

        if not info or "regularMarketPrice" not in info and "currentPrice" not in info:
            hist = await asyncio.to_thread(asset.history, period="1d")
            if not hist.empty:
                last_price = hist["Close"].iloc[-1]
                return f"Ticker: {ticker}\nLatest Close Price: ${last_price:.2f}\nData limited via basic info fetch."
            return f"No financial data could be found for ticker '{ticker}'."

        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", "Unknown")
        market_cap = info.get("marketCap", "Unknown")
        pe_ratio = info.get("trailingPE", "N/A")
        volume = info.get("volume", "Unknown")
        fifty_two_high = info.get("fiftyTwoWeekHigh", "Unknown")
        fifty_two_low = info.get("fiftyTwoWeekLow", "Unknown")

        details = (
            f"--- Financial Overview for {ticker} ---\n"
            f"Current Price: ${price}\n"
            f"Market Cap: {market_cap}\n"
            f"P/E Ratio: {pe_ratio}\n"
            f"Volume: {volume}\n"
            f"52-Week High: ${fifty_two_high}\n"
            f"52-Week Low: ${fifty_two_low}\n"
            f"Short Description: {info.get('longBusinessSummary', 'N/A')[:500]}..."
        )
        return details
    except Exception as e:
        return f"Failed to fetch data for ticker {ticker}: {str(e)}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_historical_prices(ctx: RunContext[Any], ticker: str, period: str = "1mo") -> str:
    """Get historical price data for a ticker to analyze charts and trends. Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"""
    try:

        def fetch_hist():
            asset = yf.Ticker(ticker)
            return asset.history(period=period)

        hist = await asyncio.to_thread(fetch_hist)
        if hist.empty:
            return f"No historical data found for {ticker} over period {period}."

        summary = f"Historical Close Prices for {ticker} ({period}):\n"
        for date, row in hist.iterrows():
            summary += f"{date.strftime('%Y-%m-%d')}: ${row['Close']:.2f}\n"
        return summary
    except Exception as e:
        return f"Failed to fetch historical data for {ticker}: {str(e)}"


register_tool(
    "get_stock_ticker_data",
    get_stock_ticker_data,
    "Fetch live stock/crypto financial metrics.",
    ["finance", "pricing", "search"],
)
register_tool(
    "get_historical_prices",
    get_historical_prices,
    "Fetch historical price movements (the 'chart data') for a ticker.",
    ["finance", "pricing", "search"],
)
