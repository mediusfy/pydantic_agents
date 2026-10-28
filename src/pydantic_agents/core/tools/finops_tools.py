"""FinOps tools: spend reporting from the LiteLLM proxy."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from pydantic_ai import RunContext

from pydantic_agents.core.tools.registry import register_tool

SPEND_TIMEOUT_SECONDS = 15.0
MAX_SPEND_DAYS = 90


def _spend_logs_url(base_url: str) -> str:
    """`/spend/logs` lives at the proxy root, not under the OpenAI-compatible `/v1` prefix the model factory adds."""
    return f"{base_url.rstrip('/').removesuffix('/v1')}/spend/logs"


def _spend_rows(payload: Any) -> list[dict[str, Any]]:
    """The endpoint returns a list of log rows; tolerate a dict wrapper and ignore anything that is not a row."""
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("logs") or []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _summarise(rows: list[dict[str, Any]], key_alias: str) -> dict[str, Any]:
    per_model: dict[str, float] = {}
    per_key: dict[str, float] = {}
    total = 0.0
    counted = 0
    for row in rows:
        alias = str(row.get("key_alias") or row.get("api_key") or "unknown")
        if key_alias and alias != key_alias:
            continue
        try:
            spend = float(row.get("spend") or 0.0)
        except (TypeError, ValueError):
            continue
        model = str(row.get("model") or "unknown")
        per_model[model] = per_model[model] + spend if model in per_model else spend
        per_key[alias] = per_key[alias] + spend if alias in per_key else spend
        total += spend
        counted += 1

    def rounded(table: dict[str, float]) -> dict[str, float]:
        return {k: round(v, 6) for k, v in sorted(table.items(), key=lambda kv: -kv[1])}

    return {
        "total_usd": round(total, 6),
        "requests": counted,
        "per_model": rounded(per_model),
        "per_key_alias": rounded(per_key),
    }


async def litellm_spend_report(ctx: RunContext[Any], days: int = 7, key_alias: str = "") -> str:
    """Summarise LiteLLM proxy spend over the last N days, per model group and per key alias.

    Args:
        days: Look-back window in days (1..90). Defaults to 7.
        key_alias: Only count requests made with this key alias. Empty means every key.

    Returns a compact JSON summary (total_usd, requests, per_model, per_key_alias) or an error string. Never raises.
    """
    from pydantic_agents.core.models.factory import _litellm_connection

    try:
        base_url, api_key = _litellm_connection("openai:finops")
    except Exception as e:
        return f"Error: LiteLLM proxy is not configured: {e}"

    days = max(1, min(int(days), MAX_SPEND_DAYS))
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    url = _spend_logs_url(base_url)
    params = {"start_date": start.isoformat(), "end_date": end.isoformat()}
    try:
        async with httpx.AsyncClient(timeout=SPEND_TIMEOUT_SECONDS) as client:
            resp = await client.get(url, params=params, headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} fetching spend logs from {url}"
    except httpx.RequestError as e:
        return f"Error: request for spend logs to {url} failed: {type(e).__name__}"
    except Exception as e:
        return f"Error: could not read spend logs from {url}: {type(e).__name__}: {e}"

    summary = {"start_date": params["start_date"], "end_date": params["end_date"], "days": days}
    if key_alias:
        summary["key_alias"] = key_alias
    summary.update(_summarise(_spend_rows(payload), key_alias))
    return json.dumps(summary, separators=(",", ":"))


register_tool(
    "litellm_spend_report",
    litellm_spend_report,
    "Summarise LiteLLM proxy spend over the last N days per model and per key alias",
    ["finops"],
)
