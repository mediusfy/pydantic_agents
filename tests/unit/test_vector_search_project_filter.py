import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pydantic_agents.core import vector_db as vdb


class _Conn:
    def __init__(self, calls):
        self.calls = calls

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return [{"content": "row"}]


class _Pool:
    def __init__(self):
        self.calls = []

    def acquire(self):
        return _Ctx(_Conn(self.calls))


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.parametrize(
    "project,expect_filter,expect_args",
    [
        (None, False, 2),
        ("", False, 2),
        ("openclaw", True, 3),
    ],
)
def test_search_filters_by_project(project, expect_filter, expect_args):
    pool = _Pool()

    async def _noop(*_a, **_k):
        return None

    with (
        patch.object(vdb, "_require_vector_deps"),
        patch.object(vdb, "_ensure_table", _noop),
        patch.object(vdb, "_to_embedding", return_value=[0.1, 0.2]),
        patch.object(vdb, "_get_pool", AsyncMock(return_value=pool)),
    ):
        out = asyncio.run(vdb.vector_db._search("adrs", "q", 3, project))

    assert out == ["row"]
    sql, args = pool.calls[0]
    assert ("metadata->>'project'" in sql) is expect_filter
    assert len(args) == expect_args
    if expect_filter:
        assert args[-1] == project
