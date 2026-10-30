"""Hybrid retrieval, off-loop embedding, LiteLLM embeddings/rerank and the dimension guard in vector_db."""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from pydantic_agents.core import vector_db as vdb
from pydantic_agents.core.exceptions import ConfigurationError, OrgitrationDependencyError
from pydantic_agents.core.settings import settings

# ── fakes ──────────────────────────────────────────────────────────────────


class _Conn:
    """Records every statement; ``fetch_results`` is consumed in call order, ``fetchval_result`` is returned as-is."""

    def __init__(self, calls, fetch_results=None, fetchval_result=None):
        self.calls = calls
        self.fetch_results = list(fetch_results or [])
        self.fetchval_result = fetchval_result

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        if self.fetch_results:
            return self.fetch_results.pop(0)
        return []

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        return self.fetchval_result

    async def executemany(self, sql, rows):
        self.calls.append(("executemany", sql, rows))

    def transaction(self):
        return _Ctx(self)


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_exc):
        return False


class _Pool:
    def __init__(self, fetch_results=None, fetchval_result=None):
        self.calls = []
        self._fetch_results = list(fetch_results or [])
        self._fetchval_result = fetchval_result

    def acquire(self):
        # One connection per acquire(); the shared fetch queue is consumed across acquires.
        return _Ctx(_Conn(self.calls, self._fetch_results, self._fetchval_result))


def _rows(*contents):
    return [{"content": c} for c in contents]


def _fetches(pool):
    return [(sql, args) for kind, sql, args in pool.calls if kind == "fetch"]


class _FakeResponse:
    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient: records posts, answers from a queue of responses."""

    posts: list[tuple[str, dict, dict]] = []
    responses: list[_FakeResponse] = []
    kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).posts.append((url, json, headers))
        return type(self).responses.pop(0)


@pytest.fixture
def fake_http(monkeypatch):
    _FakeAsyncClient.posts = []
    _FakeAsyncClient.responses = []
    _FakeAsyncClient.kwargs = {}
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(vdb, "_litellm_connection", lambda: ("http://proxy/v1", "sk-test"))
    return _FakeAsyncClient


@pytest.fixture
def no_table(monkeypatch):
    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(vdb, "_require_vector_deps", lambda: None)
    monkeypatch.setattr(vdb, "_ensure_table", _noop)


# ── hybrid candidate queries ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "n_results,filters,project,expect_limit,expect_params",
    [
        (3, None, None, 20, ()),
        (3, None, "openclaw", 20, ("openclaw",)),
        (10, {"agent_name": "a", "run_id": None}, "p", 40, ("a", "p")),
    ],
)
async def test_hybrid_issues_vector_and_text_queries_with_same_filters(
    monkeypatch, no_table, n_results, filters, project, expect_limit, expect_params
):
    monkeypatch.setattr(settings, "retrieval_hybrid", True)
    monkeypatch.setattr(settings, "rerank_model", "")
    pool = _Pool(fetch_results=[_rows("v1"), _rows("t1")])
    monkeypatch.setattr(vdb, "_get_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(vdb, "_embed_texts", AsyncMock(return_value=[[0.1, 0.2]]))

    out = await vdb.vector_db._search("adrs", "the query", n_results, project, filters)

    assert out == ["v1", "t1"]
    fetches = _fetches(pool)
    assert len(fetches) == 2
    (vec_sql, vec_args), (txt_sql, txt_args) = fetches
    assert "embedding <=> $1::vector" in vec_sql
    assert "LIMIT $2" in vec_sql
    assert vec_args == ("[0.1,0.2]", expect_limit, *expect_params)
    assert "ts_rank_cd(content_tsv, to_tsquery('english', $1))" in txt_sql
    assert "content_tsv @@ to_tsquery('english', $1)" in txt_sql
    assert txt_args == ("the | query", expect_limit, *expect_params)
    # Metadata placeholders start at $3 in both queries and the filter clauses are identical.
    for key, value in {**(filters or {}), **({"project": project} if project else {})}.items():
        clause = f"metadata->>'{key}' IS NULL" if value is None else f"metadata->>'{key}' = $"
        assert clause in vec_sql
        assert clause in txt_sql
    for i in range(len(expect_params)):
        assert f"= ${i + 3}" in vec_sql
        assert f"= ${i + 3}" in txt_sql


async def test_hybrid_disabled_issues_single_vector_query(monkeypatch, no_table):
    monkeypatch.setattr(settings, "retrieval_hybrid", False)
    pool = _Pool(fetch_results=[_rows("a", "b", "c")])
    monkeypatch.setattr(vdb, "_get_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(vdb, "_embed_texts", AsyncMock(return_value=[[0.5]]))

    out = await vdb.vector_db._search("adrs", "q", 2, "proj")

    assert out == ["a", "b"]
    fetches = _fetches(pool)
    assert len(fetches) == 1
    sql, args = fetches[0]
    assert "embedding <=> $1::vector" in sql
    assert "ts_rank_cd" not in sql
    assert args == ("[0.5]", 2, "proj")


# ── reciprocal rank fusion ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rankings,expected",
    [
        # Hand-computed with k=60: c = 1/62 + 1/61 ~ 0.0325, b = 1/63 + 1/62 ~ 0.0320, a = 1/61 ~ 0.0164.
        ([["a", "c", "b"], ["c", "b"]], ["c", "b", "a"]),
        # Identical rankings keep their order.
        ([["x", "y"], ["x", "y"]], ["x", "y"]),
        # One empty ranking leaves the other untouched.
        ([["x", "y", "z"], []], ["x", "y", "z"]),
        # Ties (same rank in exactly one list each) keep first-appearance order: vector ranking first.
        ([["only_vec"], ["only_txt"]], ["only_vec", "only_txt"]),
        ([], []),
    ],
)
def test_rrf_fuse(rankings, expected):
    assert vdb._rrf_fuse(rankings) == expected


@pytest.mark.parametrize("k", [1, 60])
def test_rrf_fuse_honours_k(k):
    # With k=1: a = 1/2 ~ 0.5 beats b = 1/3 + 1/2 ~ 0.83? No: b wins either way; a tie-break needs a different shape.
    # Here a appears first in list one (rank 1) and nowhere else; b is rank 2 in list one and rank 1 in list two.
    assert vdb._rrf_fuse([["a", "b"], ["b"]], k=k) == ["b", "a"]


# ── embedding providers ────────────────────────────────────────────────────


async def test_local_embedding_runs_off_the_event_loop(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "local")
    loop_thread = threading.get_ident()
    seen: list[tuple[str, int]] = []

    def fake_embed(text):
        seen.append((text, threading.get_ident()))
        return [1.0, 2.0]

    monkeypatch.setattr(vdb, "_to_embedding", fake_embed)
    out = await vdb._embed_texts(["one", "two"])

    assert out == [[1.0, 2.0], [1.0, 2.0]]
    assert [t for t, _ in seen] == ["one", "two"]
    assert all(ident != loop_thread for _, ident in seen)


async def test_local_embedding_uses_to_thread(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(vdb, "_to_embedding", lambda _t: [0.0])
    to_thread = AsyncMock(side_effect=lambda fn: fn())
    monkeypatch.setattr(asyncio, "to_thread", to_thread)

    assert await vdb._embed_texts(["x"]) == [[0.0]]
    to_thread.assert_awaited_once()


async def test_litellm_embedding_posts_expected_body(monkeypatch, fake_http):
    monkeypatch.setattr(settings, "embedding_provider", "litellm")
    monkeypatch.setattr(settings, "embedding_model", "mistral-embed")
    fake_http.responses.append(
        _FakeResponse(
            {"data": [{"index": 1, "embedding": [0.3, 0.4]}, {"index": 0, "embedding": [0.1, 0.2]}]},
        )
    )
    to_thread = AsyncMock()
    monkeypatch.setattr(asyncio, "to_thread", to_thread)

    out = await vdb._embed_texts(["first", "second"])

    assert out == [[0.1, 0.2], [0.3, 0.4]]  # sorted back by index
    assert fake_http.posts == [
        (
            "http://proxy/v1/embeddings",
            {"model": "mistral-embed", "input": ["first", "second"]},
            {"Authorization": "Bearer sk-test"},
        )
    ]
    assert fake_http.kwargs == {"timeout": 30.0}
    to_thread.assert_not_awaited()


async def test_litellm_embedding_count_mismatch_raises(monkeypatch, fake_http):
    monkeypatch.setattr(settings, "embedding_provider", "litellm")
    fake_http.responses.append(_FakeResponse({"data": [{"index": 0, "embedding": [0.1]}]}))
    with pytest.raises(ConfigurationError):
        await vdb._embed_texts(["a", "b"])


async def test_add_documents_embeds_batch_once_and_inserts_each(monkeypatch, no_table):
    pool = _Pool()
    monkeypatch.setattr(vdb, "_get_pool", AsyncMock(return_value=pool))
    embed = AsyncMock(return_value=[[0.1], [0.2]])
    monkeypatch.setattr(vdb, "_embed_texts", embed)

    await vdb.vector_db._add_documents("c", ["d1", "d2"], [{"k": "v"}, {}], ["i1", "i2"])

    embed.assert_awaited_once_with(["d1", "d2"])
    (batch,) = [rows for kind, _sql, rows in pool.calls if kind == "executemany"]
    assert batch == [("i1", '{"k": "v"}', "[0.1]", "d1"), ("i2", "{}", "[0.2]", "d2")]


# ── table setup and dimension guard ────────────────────────────────────────


@pytest.mark.parametrize(
    "existing_dim,configured_dim,raises",
    [
        (384, 384, False),
        (None, 384, False),  # column lookup returned nothing: nothing to compare
        (-1, 1024, False),  # unconstrained vector column
        (384, 1024, True),
        (1024, 384, True),
    ],
)
async def test_ensure_table_checks_dimension(monkeypatch, existing_dim, configured_dim, raises):
    monkeypatch.setattr(settings, "embedding_dim", configured_dim)
    monkeypatch.setattr(vdb, "_initialized_tables", set())
    pool = _Pool(fetchval_result=existing_dim)
    monkeypatch.setattr(vdb, "_get_pool", AsyncMock(return_value=pool))

    if raises:
        with pytest.raises(ConfigurationError, match="re-seeding"):
            await vdb._ensure_table("adrs")
        assert "adrs" not in vdb._initialized_tables
        return

    await vdb._ensure_table("adrs")

    assert "adrs" in vdb._initialized_tables
    executed = [sql for kind, sql, _ in pool.calls if kind == "execute"]
    assert any(f"embedding vector({configured_dim}) NOT NULL" in sql for sql in executed)
    assert any("ADD COLUMN IF NOT EXISTS content_tsv tsvector" in sql for sql in executed)
    assert any("idx_adrs_tsv" in sql and "USING gin (content_tsv)" in sql for sql in executed)
    assert any("idx_adrs_embedding" in sql for sql in executed)


async def test_ensure_table_is_cached(monkeypatch):
    monkeypatch.setattr(vdb, "_initialized_tables", {"adrs"})
    get_pool = AsyncMock()
    monkeypatch.setattr(vdb, "_get_pool", get_pool)
    await vdb._ensure_table("adrs")
    get_pool.assert_not_awaited()


# ── rerank ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,model,response,expected,expect_post",
    [
        # Success: ordered by relevance_score, cut to n_results.
        (
            "litellm",
            "rerank-1",
            _FakeResponse(
                {"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5}]},
            ),
            ["c", "a"],
            True,
        ),
        # HTTP error: fused order, first n_results.
        ("litellm", "rerank-1", _FakeResponse(error=RuntimeError("502 Bad Gateway")), ["a", "b"], True),
        # Malformed body: fused order.
        ("litellm", "rerank-1", _FakeResponse({"unexpected": []}), ["a", "b"], True),
        # Out-of-range indexes only: fused order.
        ("litellm", "rerank-1", _FakeResponse({"results": [{"index": 9, "relevance_score": 1.0}]}), ["a", "b"], True),
        # No model configured: no call.
        ("litellm", "", None, ["a", "b"], False),
        # Local provider: no call even with a model set.
        ("local", "rerank-1", None, ["a", "b"], False),
    ],
)
async def test_rerank(monkeypatch, fake_http, caplog, provider, model, response, expected, expect_post):
    monkeypatch.setattr(settings, "embedding_provider", provider)
    monkeypatch.setattr(settings, "rerank_model", model)
    if response is not None:
        fake_http.responses.append(response)

    out = await vdb._rerank("q", ["a", "b", "c"], 2)

    assert out == expected
    assert bool(fake_http.posts) is expect_post
    if expect_post:
        url, body, _headers = fake_http.posts[0]
        assert url == "http://proxy/v1/rerank"
        assert body == {"model": "rerank-1", "query": "q", "documents": ["a", "b", "c"], "top_n": 2}


async def test_rerank_connection_error_is_swallowed(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "litellm")
    monkeypatch.setattr(settings, "rerank_model", "rerank-1")

    def boom():
        raise ConfigurationError("no key")

    monkeypatch.setattr(vdb, "_litellm_connection", boom)
    assert await vdb._rerank("q", ["a", "b"], 1) == ["a"]


async def test_search_applies_rerank_to_fused_candidates(monkeypatch, no_table):
    monkeypatch.setattr(settings, "retrieval_hybrid", True)
    pool = _Pool(fetch_results=[_rows("v1", "shared"), _rows("shared", "t1")])
    monkeypatch.setattr(vdb, "_get_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(vdb, "_embed_texts", AsyncMock(return_value=[[0.1]]))
    rerank = AsyncMock(side_effect=lambda _q, docs, n: list(reversed(docs))[:n])
    monkeypatch.setattr(vdb, "_rerank", rerank)

    out = await vdb.vector_db._search("adrs", "q", 2)

    rerank.assert_awaited_once_with("q", ["shared", "v1", "t1"], 2)
    assert out == ["t1", "v1"]


# ── missing dependencies ───────────────────────────────────────────────────


async def test_search_returns_empty_when_vector_deps_missing(monkeypatch, caplog):
    def missing():
        raise OrgitrationDependencyError("numpy missing")

    monkeypatch.setattr(vdb, "_require_vector_deps", missing)
    with caplog.at_level("WARNING", logger=vdb.__name__):
        out = await vdb.vector_db._search("adrs", "q", 3)

    assert out == []
    assert "not installed" in caplog.text
    assert not any("install heavy dependencies" in r for r in out)


def test_require_vector_deps_skips_numpy_for_litellm(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "litellm")
    import builtins

    real_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("no numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    vdb._require_vector_deps()  # must not raise
    monkeypatch.setattr(settings, "embedding_provider", "local")
    with pytest.raises(OrgitrationDependencyError):
        vdb._require_vector_deps()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("the query", "the | query"),
        ("add a favourites panel; DROP TABLE x --", "add | a | favourites | panel | drop | table | x"),
        ("repeat repeat Repeat", "repeat"),
        ("!!! ???", ""),
        ("", ""),
    ],
)
def test_or_tsquery_ors_plain_words_only(query, expected):
    assert vdb._or_tsquery(query) == expected


async def test_a_query_without_words_skips_the_text_search(monkeypatch, no_table):
    monkeypatch.setattr(settings, "retrieval_hybrid", True)
    monkeypatch.setattr(settings, "rerank_model", "")
    pool = _Pool(fetch_results=[_rows("v1")])
    monkeypatch.setattr(vdb, "_get_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(vdb, "_embed_texts", AsyncMock(return_value=[[0.1]]))

    assert await vdb.vector_db._search("adrs", "???", 3) == ["v1"]
    assert len(_fetches(pool)) == 1


async def test_search_domains_embeds_the_query_once(monkeypatch, no_table):
    monkeypatch.setattr(settings, "retrieval_hybrid", False)
    pool = _Pool(fetch_results=[_rows("a")])
    monkeypatch.setattr(vdb, "_get_pool", AsyncMock(return_value=pool))
    embed = AsyncMock(return_value=[[0.1]])
    monkeypatch.setattr(vdb, "_embed_texts", embed)

    out = await vdb.vector_db._search_domains(["adrs", "architecture"], "q", 3, None)

    embed.assert_awaited_once_with(["q"])
    assert out == [["a"], ["a"]]  # one result list per domain, in order
