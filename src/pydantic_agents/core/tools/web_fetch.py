from typing import Any

import httpx
from pydantic_ai import RunContext

from pydantic_agents.core.tools.registry import register_tool


async def fetch_url(ctx: RunContext[Any], url: str, format: str = "text") -> str:
    """Fetch content from a URL via HTTP GET.

    Args:
        url: The HTTP or HTTPS URL to fetch. Do NOT pass local file paths or file:// URLs. Use read_file for local files.
        format: The response format requested (e.g. 'text', 'json').
    """
    if not url.startswith(("http://", "https://")):  # also rejects file:// and local paths
        return f"Error: Invalid URL '{url}'. This tool requires an http:// or https:// URL. For local files, use the read_file tool instead."

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Orgitration-Agent/1.0"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                import json

                return json.dumps(resp.json(), indent=2)
            return resp.text
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} fetching {url}: {e}"
    except httpx.RequestError as e:
        return f"Request failed for {url}: {e}"
    except Exception as e:
        return f"Unexpected error fetching {url}: {e}"


register_tool("fetch_url", fetch_url, "Fetch content from a URL via HTTP GET", ["web", "docs", "research"])
