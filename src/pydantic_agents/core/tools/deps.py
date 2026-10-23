from typing import Any

import httpx
from pydantic_ai import RunContext

from pydantic_agents.core.tools.registry import register_tool


async def check_go_package(ctx: RunContext[Any], package: str) -> str:
    try:
        url = f"https://proxy.golang.org/{package}/@latest"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("Version", "unknown")
            return f"not found (HTTP {resp.status_code})"
    except Exception as e:
        return f"error: {e}"


async def check_pypi_package(ctx: RunContext[Any], package: str) -> dict:
    try:
        url = f"https://pypi.org/pypi/{package}/json"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                info = data.get("info", {})
                return {
                    "name": info.get("name", package),
                    "version": info.get("version", "unknown"),
                    "summary": (info.get("summary", "") or "")[:200],
                    "home_page": info.get("home_page", ""),
                    "license": info.get("license", ""),
                }
            return {"name": package, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"name": package, "error": str(e)}


async def check_npm_package(ctx: RunContext[Any], package: str) -> dict:
    try:
        url = f"https://registry.npmjs.org/{package}/latest"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "name": data.get("name", package),
                    "version": data.get("version", "unknown"),
                    "description": (data.get("description", "") or "")[:200],
                }
            return {"name": package, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"name": package, "error": str(e)}


async def check_package_version(ctx: RunContext[Any], package: str, language: str = "go") -> str:
    if language == "go":
        return await check_go_package(ctx, package)
    elif language == "python":
        result = await check_pypi_package(ctx, package)
        return result.get("version", result.get("error", "unknown"))
    elif language == "npm":
        result = await check_npm_package(ctx, package)
        return result.get("version", result.get("error", "unknown"))
    return "unknown"


register_tool("check_go_package", check_go_package, "Check latest Go module version via proxy.golang.org", ["deps"])
register_tool("check_pypi_package", check_pypi_package, "Check PyPI package details", ["deps"])
register_tool("check_npm_package", check_npm_package, "Check NPM package details", ["deps"])
register_tool(
    "check_package_version", check_package_version, "Check latest version of a package (go/python/npm)", ["deps"]
)
