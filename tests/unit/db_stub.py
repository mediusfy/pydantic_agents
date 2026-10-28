"""In-memory stand-in for the asyncpg pool that the best-effort database helpers (``run_store``,
``platform_metrics``, ``platform_findings``) get from ``vector_db._get_pool()``.

Each test module supplies its own connection class, which answers the SQL that module's helper issues against a
plain dict; the pool, the ``async with pool.acquire()`` context and the monkeypatching are the same everywhere.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

from pydantic_agents.core.settings import settings


class StubCtx:
    def __init__(self, conn: Any):
        self.conn = conn

    async def __aenter__(self) -> Any:
        return self.conn

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class StubPool:
    """``acquire()`` hands out a fresh connection over one shared ``db`` dict."""

    def __init__(self, conn_factory: Callable[[dict[str, Any]], Any]):
        self.db: dict[str, Any] = {}
        self._conn_factory = conn_factory

    def acquire(self) -> StubCtx:
        return StubCtx(self._conn_factory(self.db))


def stub_database(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, conn_factory: Callable[[dict[str, Any]], Any]
) -> StubPool:
    """Enable ``module`` against an in-memory pool.

    ``module`` must be one of the best-effort database helpers: it needs a module-level ``_table_ready`` flag (reset
    here so each test re-creates its table) and a ``vector_db`` import whose ``_get_pool`` is replaced. A module
    without either fails at ``monkeypatch.setattr`` with an AttributeError, before any test runs against it.
    """
    pool = StubPool(conn_factory)
    monkeypatch.setattr(settings, "database_url", "postgresql://stub/db")
    monkeypatch.setattr(module, "_table_ready", False)

    async def get_pool() -> StubPool:
        return pool

    monkeypatch.setattr(module.vector_db, "_get_pool", get_pool)
    return pool
