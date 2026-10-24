"""Typed results for the investing agents (crypto_oracle, market_shark, price_analyst).

The rules the prompts used to ask for in prose are enforced here, so a result that breaks them fails validation and
pydantic-ai asks the model to correct it: scenario probabilities add up, ranges are ordered, every result cites at
least one source, and the not-financial-advice disclaimer is set by code rather than trusted to the model.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DISCLAIMER = "This is analysis based on public data, not financial advice."

# Scenario probabilities may be rounded by the model; anything further from 1.0 than this is a mistake.
PROBABILITY_TOLERANCE = 0.05


class Source(BaseModel):
    tool: str = Field(description="Tool or origin of the figure, e.g. get_historical_prices, search_news, user")
    reference: str = Field(default="", description="Ticker and date range, URL or headline that identifies it")


class Scenario(BaseModel):
    name: str = Field(description="e.g. base, bull, bear")
    low: float
    high: float
    probability: float = Field(ge=0.0, le=1.0)
    drivers: list[str] = Field(default_factory=list, description="What would have to happen for this scenario")

    @model_validator(mode="after")
    def _ordered_range(self) -> "Scenario":
        if self.low > self.high:
            raise ValueError(f"scenario '{self.name}': low ({self.low}) is above high ({self.high})")
        return self


class _Analysis(BaseModel):
    as_of: str = Field(description="Date of the analysis (ISO 8601), taken from get_current_datetime")
    summary: str
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list, description="Main risks and what would change the view")
    sources: list[Source] = Field(min_length=1, description="Where every number came from; at least one")
    disclaimer: str = Field(default=DISCLAIMER, description="Always the standard disclaimer; set for you")

    @model_validator(mode="after")
    def _fixed_disclaimer(self) -> "_Analysis":
        self.disclaimer = DISCLAIMER
        return self


def _check_probabilities(scenarios: list[Scenario]) -> None:
    total = sum(s.probability for s in scenarios)
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(f"scenario probabilities add up to {total:.2f}, expected 1.0 (+/- {PROBABILITY_TOLERANCE})")


class PreviousForecast(BaseModel):
    prediction: str = Field(description="What the earlier forecast said, as logged in history")
    actual: str = Field(description="What the price actually did since, from tool data")
    error_pct: float | None = Field(default=None, description="Margin of error in percent, if it can be computed")
    lesson: str = Field(description="Why it was off (or right) and what changes in this forecast")


class Forecast(_Analysis):
    asset: str
    horizon: str = Field(description="e.g. '30 days', 'end of Q1'")
    currency: str = "USD"
    scenarios: list[Scenario] = Field(min_length=2, description="base / bull / bear price ranges with probabilities")
    previous_forecast: PreviousForecast | None = Field(
        default=None, description="Accountability for the last forecast; None only if query_history found none"
    )

    @model_validator(mode="after")
    def _probabilities_add_up(self) -> "Forecast":
        _check_probabilities(self.scenarios)
        return self


class Metric(BaseModel):
    name: str = Field(description="e.g. P/E, 30d average volume, % below 52-week high")
    value: str
    source: str = Field(description="Tool or reference the inputs came from")


class MarketAnalysis(_Analysis):
    asset: str
    thesis: str = Field(description="What the data implies, stated specifically")
    metrics: list[Metric] = Field(min_length=1)
    bull_case: list[str] = Field(min_length=1)
    bear_case: list[str] = Field(min_length=1)
    risk_reward: str | None = Field(default=None, description="Quantified risk/reward, with its inputs")


class PurchaseScenario(BaseModel):
    action: Literal["buy_now", "wait"]
    conditions: list[str] = Field(min_length=1, description="What favours this choice")
    expected_price_change_pct: float | None = None
    rationale: str


class PurchaseAdvice(_Analysis):
    item: str
    currency: str = "USD"
    current_price: float | None = Field(default=None, description="From tool output; None if it could not be found")
    leaning: Literal["buy_now", "wait", "unclear"]
    trend: str = Field(description="Past and present price trend, with the calculation behind it")
    scenarios: list[PurchaseScenario] = Field(min_length=2, description="Buy-now and wait, each with its conditions")

    @model_validator(mode="after")
    def _both_choices_covered(self) -> "PurchaseAdvice":
        actions = {s.action for s in self.scenarios}
        if actions != {"buy_now", "wait"}:
            raise ValueError("scenarios must cover both buy_now and wait")
        return self
