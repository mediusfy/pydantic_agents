"""Typed results for the personal agents (holiday_planner, accountant).

As with the investing agents (core/models/analysis.py), rules the prompts used to ask for in prose are enforced here, so
a result that breaks them fails validation and pydantic-ai asks the model to correct it: unverified details say where
to confirm them, fixed costs are exact, totals are added up in code, and every figure is sourced. A request that is
missing key details is answered with ``open_questions`` instead of an invented plan.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from pydantic_agents.core.models.analysis import Source

ACCOUNTING_DISCLAIMER = "This is a calculation based on the figures provided, not tax, legal or financial advice."

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Totals are sums of floats the model typed, so allow for representation error but not for a real difference.
_MONEY_SLACK = 0.005


def _check_iso_date(value: str, field: str) -> None:
    if not _ISO_DATE.match(value):
        raise ValueError(f"{field} must be an ISO 8601 date (YYYY-MM-DD), got {value!r}")


# ── holiday_planner ───────────────────────────────────────────────────────


class ItineraryItem(BaseModel):
    time: str = Field(description="Start time or slot, e.g. '09:30' or 'morning'")
    activity: str
    location: str = ""
    duration_minutes: int | None = Field(default=None, gt=0)
    travel_minutes_from_previous: int | None = Field(
        default=None, ge=0, description="Realistic travel time from the previous item; None for the first item"
    )
    cost: float | None = Field(default=None, ge=0, description="Per group, in the itinerary currency; None if unknown")
    status: Literal["confirmed", "to_confirm"] = Field(
        description="'confirmed' only if a tool result or the user supplied it; otherwise 'to_confirm'"
    )
    confirm_where: str = Field(
        default="", description="Required for 'to_confirm': where to check opening hours, availability or price"
    )

    @model_validator(mode="after")
    def _unverified_items_say_where_to_confirm(self) -> "ItineraryItem":
        if self.status == "to_confirm" and not self.confirm_where.strip():
            raise ValueError(f"'{self.activity}' is to_confirm but does not say where to confirm it")
        return self


class ItineraryDay(BaseModel):
    day: int = Field(ge=1)
    date: str = Field(description="ISO 8601 date (YYYY-MM-DD)")
    weather: str = Field(default="", description="Forecast or seasonal note for this date, with its source")
    items: list[ItineraryItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_date(self) -> "ItineraryDay":
        _check_iso_date(self.date, f"day {self.day} date")
        return self


class Itinerary(BaseModel):
    destination: str
    start_date: str = Field(default="", description="ISO 8601 date (YYYY-MM-DD); empty only if there is no plan yet")
    end_date: str = Field(default="", description="ISO 8601 date (YYYY-MM-DD); empty only if there is no plan yet")
    travellers: int | None = Field(default=None, ge=1)
    currency: str = "USD"
    budget: float | None = Field(default=None, ge=0, description="The user's budget, if they gave one")
    estimated_total: float | None = Field(
        default=None, ge=0, description="Derived in code from the item costs; supplied value ignored"
    )
    within_budget: bool | None = Field(
        default=None, description="Derived in code: estimated_total <= budget; None without both"
    )
    summary: str = ""
    days: list[ItineraryDay] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(
        default_factory=list,
        description="Details still needed (dates, travellers, budget, constraints). With no days, this is the answer",
    )
    sources: list[Source] = Field(default_factory=list, description="Where weather, prices and hours came from")

    @model_validator(mode="after")
    def _consistent_plan(self) -> "Itinerary":
        if not self.days:
            if not self.open_questions:
                raise ValueError("an itinerary needs days, or open_questions saying what is missing")
            return self

        if not self.sources:
            raise ValueError("an itinerary with days must cite at least one source")
        for field in ("start_date", "end_date"):
            _check_iso_date(getattr(self, field), field)
        if self.start_date > self.end_date:
            raise ValueError(f"start_date {self.start_date} is after end_date {self.end_date}")

        numbers = [d.day for d in self.days]
        if numbers != list(range(1, len(self.days) + 1)):
            raise ValueError(f"days must be numbered 1..{len(self.days)} in order, got {numbers}")
        dates = [d.date for d in self.days]
        if dates != sorted(set(dates)):
            raise ValueError("day dates must be unique and in order")
        if dates[0] < self.start_date or dates[-1] > self.end_date:
            raise ValueError(f"day dates {dates[0]}..{dates[-1]} fall outside {self.start_date}..{self.end_date}")

        costs = [i.cost for d in self.days for i in d.items if i.cost is not None]
        self.estimated_total = round(sum(costs), 2) if costs else None
        if self.estimated_total is not None and self.budget is not None:
            self.within_budget = self.estimated_total <= self.budget + _MONEY_SLACK
        else:
            self.within_budget = None
        return self


# ── accountant ────────────────────────────────────────────────────────────


class ExchangeRate(BaseModel):
    from_currency: str
    to_currency: str
    rate: float = Field(gt=0)
    as_of: str = Field(description="Date the rate was retrieved (ISO 8601)")
    source: str

    @model_validator(mode="after")
    def _valid_date(self) -> "ExchangeRate":
        _check_iso_date(self.as_of, "exchange rate as_of")
        return self


class CostLine(BaseModel):
    name: str
    kind: Literal["fixed", "estimate"] = Field(description="'fixed' is an exact known cost; 'estimate' has a range")
    low: float = Field(ge=0, description="In the breakdown currency (converted with a listed rate if needed)")
    high: float = Field(ge=0)
    basis: str = Field(description="Inputs and formula, e.g. '3 nights x 120 EUR x 1.08 USD/EUR'")
    source: str = Field(default="", description="Tool or reference the inputs came from; 'user' if the user gave them")

    @model_validator(mode="after")
    def _range_matches_kind(self) -> "CostLine":
        if self.low > self.high:
            raise ValueError(f"'{self.name}': low ({self.low}) is above high ({self.high})")
        if self.kind == "fixed" and abs(self.low - self.high) > _MONEY_SLACK:
            raise ValueError(f"'{self.name}' is fixed but has a range ({self.low}..{self.high}); mark it an estimate")
        return self


class CostBreakdown(BaseModel):
    as_of: str = Field(description="Date of the calculation (ISO 8601), from get_current_datetime")
    summary: str = Field(default="", description="The answer in a sentence or two, in the breakdown currency")
    currency: str = Field(description="The one currency every line and total is expressed in")
    lines: list[CostLine] = Field(default_factory=list)
    total_low: float | None = Field(
        default=None, description="Derived in code: sum of the lines; supplied value ignored"
    )
    total_high: float | None = Field(
        default=None, description="Derived in code: sum of the lines; supplied value ignored"
    )
    exchange_rates: list[ExchangeRate] = Field(
        default_factory=list, description="Every rate used, with date and source"
    )
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(
        default_factory=list,
        description="Missing inputs (currency, dates, quantities). With no lines, this is the answer",
    )
    sources: list[Source] = Field(default_factory=list)
    disclaimer: str = Field(default=ACCOUNTING_DISCLAIMER, description="Always the standard disclaimer; set for you")

    @model_validator(mode="after")
    def _totals_in_code(self) -> "CostBreakdown":
        self.disclaimer = ACCOUNTING_DISCLAIMER
        _check_iso_date(self.as_of, "as_of")
        if not self.lines:
            if not self.open_questions:
                raise ValueError("a breakdown needs lines, or open_questions saying what is missing")
            self.total_low = self.total_high = None
            return self

        if not self.sources:
            raise ValueError("a breakdown with lines must cite at least one source")
        self.total_low = round(sum(line.low for line in self.lines), 2)
        self.total_high = round(sum(line.high for line in self.lines), 2)
        return self
