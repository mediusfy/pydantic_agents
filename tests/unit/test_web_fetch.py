"""Unit tests for SSRF validation and redirect/size handling in web_fetch."""

import json
import socket
from unittest.mock import MagicMock

import httpx
import pytest

from pydantic_agents.core.tools import web_fetch
from pydantic_agents.core.tools.web_fetch import fetch_url, validate_public_http_url


def _addrinfo(*ips: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]


@pytest.mark.parametrize(
    "blocked_url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1:8080/admin",
        "http://localhost:4000/v1/models",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/status",
        "http://[::1]:8000/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://service.internal/api",
    ],
)
def test_validate_public_http_url_blocks_ssrf_targets(blocked_url):
    err = validate_public_http_url(blocked_url, resolve_dns=False)
    assert err is not None
    assert err.startswith("Error:")


def test_validate_public_http_url_allows_public_ip():
    assert validate_public_http_url("https://93.184.216.34/index.html", resolve_dns=False) is None


@pytest.mark.parametrize(
    ("ips", "blocked"),
    [(("93.184.216.34",), False), (("93.184.216.34", "10.0.0.5"), True), (("127.0.0.1",), True)],
)
def test_validate_public_http_url_checks_every_resolved_address(monkeypatch, ips, blocked):
    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", lambda *_a, **_k: _addrinfo(*ips))
    err = web_fetch.validate_public_http_url("https://example.com/x")
    assert (err is not None) is blocked
    if blocked:
        assert "resolves to non-public IP" in err


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        ("ftp://example.com", "requires an http"),
        ("http://", "missing hostname"),
        ("http://svc.internal/x", "internal host"),
        ("http://[::ffff:127.0.0.1]/", "non-public IP"),
    ],
)
def test_validate_public_http_url_rejections(url, fragment):
    err = validate_public_http_url(url, resolve_dns=False)
    assert err is not None
    assert fragment in err


def test_validate_public_http_url_unresolvable_host_is_left_to_the_client(monkeypatch):
    def boom(*_a, **_k):
        raise socket.gaierror("nope")

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", boom)
    assert web_fetch.validate_public_http_url("https://no-such-host.example") is None
    assert web_fetch.validate_public_http_url("https://example.com", resolve_dns=False) is None


def _patch_client(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        web_fetch.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setattr(web_fetch, "validate_public_http_url", lambda url, **_k: None if "evil" not in url else "Err")
    return web_fetch


@pytest.mark.asyncio
async def test_fetch_url_blocks_redirect_to_loopback(monkeypatch):
    monkeypatch.setattr(
        "pydantic_agents.core.tools.web_fetch.validate_public_http_url",
        lambda u, **_kw: validate_public_http_url(u, resolve_dns=False),
    )

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, **_kwargs):
            req = httpx.Request("GET", url)
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"}, request=req)

    monkeypatch.setattr("pydantic_agents.core.tools.web_fetch.httpx.AsyncClient", _FakeClient)
    out = await fetch_url(MagicMock(), "https://93.184.216.34/start")
    assert out.startswith("Error: Access to non-public IP address")


@pytest.mark.asyncio
async def test_fetch_url_follows_safe_redirects_and_caps_the_body(monkeypatch):
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(200, text="x" * 600_000, headers={"content-type": "text/plain"})

    wf = _patch_client(monkeypatch, handler)
    out = await wf.fetch_url(MagicMock(), "https://example.com/start")
    assert len(out) == wf._MAX_RESPONSE_CHARS


@pytest.mark.asyncio
async def test_fetch_url_formats_json(monkeypatch):
    wf = _patch_client(monkeypatch, lambda _r: httpx.Response(200, json={"a": 1}))
    assert json.loads(await wf.fetch_url(MagicMock(), "https://example.com/j")) == {"a": 1}


@pytest.mark.parametrize(
    ("handler", "fragment"),
    [
        (lambda _r: httpx.Response(302), "no Location header"),
        (lambda _r: httpx.Response(302, headers={"location": "https://evil.example/"}), "Err"),
        (lambda _r: httpx.Response(302, headers={"location": "/loop"}), "Too many redirects"),
        (lambda _r: httpx.Response(404), "HTTP error 404"),
    ],
)
@pytest.mark.asyncio
async def test_fetch_url_failure_modes(monkeypatch, handler, fragment):
    wf = _patch_client(monkeypatch, handler)
    assert fragment in await wf.fetch_url(MagicMock(), "https://example.com/loop")


@pytest.mark.asyncio
async def test_fetch_url_rejects_a_blocked_initial_url():
    assert "non-public" in await fetch_url(MagicMock(), "http://169.254.169.254/latest/meta-data/")


@pytest.mark.asyncio
async def test_fetch_url_reports_request_errors(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    wf = _patch_client(monkeypatch, handler)
    assert "Request failed" in await wf.fetch_url(MagicMock(), "https://example.com/")
