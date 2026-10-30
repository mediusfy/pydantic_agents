import asyncio
import ipaddress
import json
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from pydantic_ai import RunContext

from pydantic_agents.core.tools.registry import register_tool

_MAX_REDIRECTS = 5
_MAX_RESPONSE_CHARS = 500_000
_BLOCKED_HOSTNAMES = frozenset({"localhost", "metadata.google.internal", "metadata", "instance-data"})
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal")


def _is_forbidden_ip(raw_ip: str) -> bool:
    """True when ``raw_ip`` is loopback, private, link-local, multicast, reserved, or unspecified."""
    try:
        ip = ipaddress.ip_address(raw_ip)
    except ValueError:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and _is_forbidden_ip(str(mapped)):
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def validate_public_http_url(url: str, *, resolve_dns: bool = True) -> str | None:
    """Return an error string if ``url`` is not a safe public HTTP(S) URL, or ``None`` if allowed."""
    if not url.startswith(("http://", "https://")):
        return (
            f"Error: Invalid URL '{url}'. This tool requires an http:// or https:// URL. "
            "For local files, use the read_file tool instead."
        )
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return f"Error: Invalid URL '{url}': missing hostname."

    host = parsed.hostname.lower().rstrip(".")
    if host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_SUFFIXES):
        return f"Error: Access to internal host '{host}' is forbidden."

    # Literal IP check
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if _is_forbidden_ip(host):
            return f"Error: Access to non-public IP address '{host}' is forbidden."
        return None

    if not resolve_dns:
        return None

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        # Let the HTTP client report DNS resolution errors naturally if not resolvable here
        return None

    for info in infos:
        sockaddr = info[4]
        if sockaddr and _is_forbidden_ip(str(sockaddr[0])):
            return f"Error: Host '{host}' resolves to non-public IP address '{sockaddr[0]}', which is forbidden."
    return None


# Bytes read from a response. `resp.text` loaded the whole body before it was cut to _MAX_RESPONSE_CHARS, so one
# large response could use all the memory; 4 bytes per character is the UTF-8 worst case.
_MAX_RESPONSE_BYTES = _MAX_RESPONSE_CHARS * 4


async def _get_bounded(client: httpx.AsyncClient, url: str) -> tuple[httpx.Response, bytes]:
    """GET ``url`` and read at most ``_MAX_RESPONSE_BYTES`` of the body."""
    async with client.stream("GET", url, headers={"User-Agent": "Orgitration-Agent/1.0"}) as resp:
        body = bytearray()
        if not resp.is_redirect:
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) >= _MAX_RESPONSE_BYTES:
                    break
        return resp, bytes(body[:_MAX_RESPONSE_BYTES])


async def fetch_url(ctx: RunContext[Any], url: str, format: str = "text") -> str:
    """Fetch content from a URL via HTTP GET.

    Args:
        url: The HTTP or HTTPS URL to fetch. Do NOT pass local file paths or file:// URLs. Use read_file for local files.
        format: The response format requested (e.g. 'text', 'json').
    """
    err = await asyncio.to_thread(validate_public_http_url, url)
    if err:
        return err

    try:
        current_url = url
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                resp, body = await _get_bounded(client, current_url)
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        return f"Error: Redirect from {current_url} had no Location header."
                    next_url = urljoin(current_url, location)
                    redirect_err = await asyncio.to_thread(validate_public_http_url, next_url)
                    if redirect_err:
                        return redirect_err
                    current_url = next_url
                    continue
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                text = body.decode(resp.encoding or "utf-8", errors="replace")
                if "json" in content_type:
                    try:
                        return json.dumps(json.loads(text), indent=2)[:_MAX_RESPONSE_CHARS]
                    except ValueError:
                        pass  # truncated or invalid JSON: return it as text
                return text[:_MAX_RESPONSE_CHARS]
            return f"Error: Too many redirects fetching {url}."
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} fetching {url}: {e}"
    except httpx.RequestError as e:
        return f"Request failed for {url}: {e}"
    except Exception as e:
        return f"Unexpected error fetching {url}: {e}"


register_tool("fetch_url", fetch_url, "Fetch content from a URL via HTTP GET", ["web", "docs", "research"])
