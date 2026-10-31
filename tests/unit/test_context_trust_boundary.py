"""AP-102: retrieval must not serve unapproved ADR proposals as architecture context.

``innovation_scout.propose_adr`` writes drafts into the ``adrs`` collection tagged ``status: "proposed"``, and a
human CTO approval gate (``core/cto_gate.py``) decides whether a proposal ever becomes a decision. Nothing read
that status back on the retrieval path, so a draft reached every agent querying the ``adrs`` domain — including
``security``, ``review`` and ``quality_gate``, the three gates that exist to catch it. See ADR-0002.

The filter must keep rows that carry no ``status`` key at all: approved and seeded ADRs have none, so an equality
filter on ``status`` would hide the entire real knowledge base.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pydantic_agents.core import vector_db as vdb
from pydantic_agents.core.context_engine import AGENT_CONTEXT_DOMAINS, retrieve_context

# The three quality gates that must never be fed an unapproved proposal.
GATE_AGENTS = ("security", "review", "quality_gate")


class _Conn:
    def __init__(self, calls):
        self.calls = calls

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return [{"content": "row"}]


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_exc):
        return False


class _Pool:
    def __init__(self):
        self.calls = []

    def acquire(self):
        return _Ctx(_Conn(self.calls))


def _run_search(**kwargs):
    """Drive ``_search`` against a stub pool and return the SQL/args of every query it issued."""
    pool = _Pool()

    async def _noop(*_a, **_k):
        return None

    with (
        patch.object(vdb, "_require_vector_deps"),
        patch.object(vdb, "_ensure_table", _noop),
        patch.object(vdb, "_to_embedding", return_value=[0.1, 0.2]),
        patch.object(vdb, "_get_pool", AsyncMock(return_value=pool)),
    ):
        asyncio.run(vdb.vector_db._search("adrs", "q", 3, None, **kwargs))
    return pool.calls


def _captured_domain_filters(agent_name="security"):
    """What ``retrieve_context`` hands to ``async_search_domains``."""
    captured: dict = {}

    async def _fake(collections, query, n_results=3, project=None, **kwargs):
        captured["collections"] = list(collections)
        captured["kwargs"] = kwargs
        return [[] for _ in collections]

    with patch.object(vdb.vector_db, "async_search_domains", _fake):
        asyncio.run(retrieve_context(agent_name, "add a payments endpoint"))
    return captured


def test_retrieve_context_excludes_proposed_adrs_by_default():
    """The default retrieval path must ask the store to leave proposals out."""
    captured = _captured_domain_filters("security")

    assert "adrs" in captured["collections"], "the security agent should still query the adrs domain"
    domain_filters = captured["kwargs"].get("domain_filters")
    assert domain_filters is not None, "retrieve_context sent no metadata filter at all (AP-102)"

    adrs_filter = domain_filters.get("adrs")
    assert adrs_filter is not None, "the adrs domain was searched with no filter"
    assert dict(adrs_filter.exclude) == {"status": "proposed"}


def test_proposed_adrs_require_an_explicit_opt_in():
    """A caller that genuinely wants drafts passes its own filters; nothing else can reach a proposal."""
    captured: dict = {}

    async def _fake(collections, query, n_results=3, project=None, **kwargs):
        captured["kwargs"] = kwargs
        return [[] for _ in collections]

    with patch.object(vdb.vector_db, "async_search_domains", _fake):
        asyncio.run(retrieve_context("security", "q", domain_filters={}))

    assert captured["kwargs"].get("domain_filters") == {}


def test_search_excludes_proposed_with_is_distinct_from():
    """The exclusion keeps NULL-status rows, so approved and seeded ADRs survive it."""
    calls = _run_search(exclude={"status": "proposed"})

    sql, args = calls[0]
    assert "metadata->>'status'" in sql
    assert "IS DISTINCT FROM" in sql, "a plain = / != would also drop every ADR that has no status key"
    # The value is bound, never interpolated: the query must stay injection-safe.
    assert "proposed" not in sql
    assert "proposed" in args


def test_exclusion_is_applied_to_the_hybrid_text_query_too():
    """Hybrid retrieval issues a second, full-text query; a filter on only one of them still leaks."""
    calls = _run_search(exclude={"status": "proposed"})

    assert len(calls) == 2, "expected the vector query and the hybrid full-text query"
    for sql, args in calls:
        assert "IS DISTINCT FROM" in sql
        assert "proposed" in args


def test_placeholder_numbering_survives_project_filters_and_exclusions_together():
    """The three condition sources share one parameter list, so their ``$n`` numbering must stay aligned.

    ``$1`` is the query vector (or the tsquery), ``$2`` the LIMIT, ``$3..`` the metadata values. An IS NULL
    condition consumes no parameter, which is exactly where an off-by-one would hide.
    """
    pool = _Pool()

    async def _noop(*_a, **_k):
        return None

    with (
        patch.object(vdb, "_require_vector_deps"),
        patch.object(vdb, "_ensure_table", _noop),
        patch.object(vdb, "_to_embedding", return_value=[0.1, 0.2]),
        patch.object(vdb, "_get_pool", AsyncMock(return_value=pool)),
    ):
        asyncio.run(
            vdb.vector_db._search(
                "adrs",
                "payments",
                3,
                "openclaw",
                filters={"kind": "decision", "legacy": None},
                exclude={"status": "proposed"},
            )
        )

    for sql, args in pool.calls:
        assert "metadata->>'kind' = $3" in sql
        assert "metadata->>'legacy' IS NULL" in sql
        assert "metadata->>'project' = $4" in sql
        assert "metadata->>'status' IS DISTINCT FROM $5" in sql
        # $1 and $2 are the query and the limit; the metadata values follow in order.
        assert args[2:] == ("decision", "openclaw", "proposed")


def test_exclude_key_must_be_a_plain_identifier():
    """Metadata keys are interpolated into SQL, so they stay behind the _METADATA_KEY allowlist."""
    with pytest.raises(ValueError, match="metadata filter key"):
        _run_search(exclude={"status'; DROP TABLE v_adrs; --": "proposed"})


def test_search_domains_applies_the_filter_to_the_right_collection():
    """The seam between the policy and the SQL: ``_search_domains`` unpacking a ``DomainFilter`` per collection.

    Without this, both ends are tested and the middle is not — swapping ``filters=`` and ``exclude=`` here would
    leave the trust boundary wide open with every other test still green.
    """
    from pydantic_agents.core.context_engine import DEFAULT_DOMAIN_FILTERS

    pool = _Pool()

    async def _noop(*_a, **_k):
        return None

    with (
        patch.object(vdb, "_require_vector_deps"),
        patch.object(vdb, "_ensure_table", _noop),
        patch.object(vdb, "_to_embedding", return_value=[0.1, 0.2]),
        patch.object(vdb, "_embed_texts", AsyncMock(return_value=[[0.1, 0.2]])),
        patch.object(vdb, "_get_pool", AsyncMock(return_value=pool)),
    ):
        asyncio.run(
            vdb.vector_db._search_domains(
                ["adrs", "architecture"], "q", 3, None, DEFAULT_DOMAIN_FILTERS
            )
        )

    adrs = [(sql, args) for sql, args in pool.calls if "v_adrs" in sql]
    architecture = [(sql, args) for sql, args in pool.calls if "v_architecture" in sql]

    assert adrs, "the adrs collection was not searched"
    assert architecture, "the architecture collection was not searched"
    # The exclusion lands on adrs only, as an exclusion and not as an equality match.
    for sql, args in adrs:
        assert "metadata->>'status' IS DISTINCT FROM $3" in sql
        assert args[2:] == ("proposed",)
    # A domain with no filter entry is searched exactly as before: no metadata predicate, no extra parameter.
    for sql, args in architecture:
        assert "metadata->>" not in sql
        assert args[2:] == ()


def test_the_default_filter_covers_every_gate_that_reads_adrs():
    """Policy-level guard: the three gates read the adrs domain, and the default filter covers that domain."""
    from pydantic_agents.core.context_engine import ADR_PROPOSED_STATUS, DEFAULT_DOMAIN_FILTERS

    for agent in GATE_AGENTS:
        assert "adrs" in AGENT_CONTEXT_DOMAINS[agent]

    assert ADR_PROPOSED_STATUS == "proposed"
    assert dict(DEFAULT_DOMAIN_FILTERS["adrs"].exclude) == {"status": ADR_PROPOSED_STATUS}
