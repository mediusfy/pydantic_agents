"""innovation_scout_agent (AP-81, ADR-0001): registration, its own tools called directly (no LLM), and the
AP-81 acceptance criterion that no code path in the module bypasses the CTO approval gate.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.agents import innovation_scout
from pydantic_agents.agents.innovation_scout import (
    build_report,
    innovation_scout_agent,
    propose_adr,
    search_existing_adrs,
)
from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.models.core import InnovationProposal
from pydantic_agents.core.registry import registry
from pydantic_agents.core.tool_assignments import AGENT_TOOL_CATEGORIES

CTX = MagicMock()

# Any write path that would let a draft skip human/CTO review: creating implementation tickets, triggering a
# pipeline run, or writing/reading the CTO approval marker directly. None of these may appear anywhere in the
# module's source, so a future edit cannot quietly add one.
FORBIDDEN_BYPASS_TOKENS = [
    "write_approval",
    "cto_gate.apply",
    "ORCHESTRATION_CTO_OVERRIDE",
    "run_orchestration_pipeline",
    "scaffold_and_build_service",
    "create_issue",
    "create_pull_request",
]


def test_innovation_scout_is_registered():
    register_all_agents()
    assert "innovation_scout" in registry.list_agents()
    assert registry.get("innovation_scout") is innovation_scout_agent


def test_innovation_scout_has_a_tool_category_entry():
    assert "innovation_scout" in AGENT_TOOL_CATEGORIES
    assert AGENT_TOOL_CATEGORIES["innovation_scout"] == ["web", "search", "memory", "documentation"]


async def test_search_existing_adrs_queries_the_adr_store(monkeypatch):
    mock_search = AsyncMock(return_value=["past decision: use X"])
    monkeypatch.setattr(innovation_scout.vector_db, "async_search", mock_search)

    result = await search_existing_adrs(CTX, "caching strategy")

    mock_search.assert_awaited_once_with("adrs", "caching strategy", n_results=5)
    assert "past decision: use X" in result


async def test_search_existing_adrs_reports_no_results(monkeypatch):
    monkeypatch.setattr(innovation_scout.vector_db, "async_search", AsyncMock(return_value=[]))

    result = await search_existing_adrs(CTX, "quantum caching")

    assert "No ADRs found" in result


async def test_propose_adr_persists_a_draft_via_the_adr_store(monkeypatch):
    mock_add = AsyncMock(return_value=None)
    monkeypatch.setattr(innovation_scout.vector_db, "async_add_documents", mock_add)

    result = await propose_adr(CTX, "ADR-0099", "## Context\n...")

    mock_add.assert_awaited_once_with(
        "adrs", ["## Context\n..."], [{"id": "ADR-0099", "status": "proposed"}], ["ADR-0099"]
    )
    assert "ADR-0099" in result
    assert "pending human/CTO approval" in result


def test_build_report_assembles_proposals_without_an_llm_call():
    report = build_report([InnovationProposal(decision_id="ADR-0099", title="Adopt X", summary="...")], narrative="n")
    assert report.narrative == "n"
    assert report.proposals[0].decision_id == "ADR-0099"


@pytest.mark.parametrize("token", FORBIDDEN_BYPASS_TOKENS)
def test_no_source_reference_to_a_gate_bypassing_write_path(token):
    source = inspect.getsource(innovation_scout)
    assert token not in source, f"innovation_scout.py references a gate-bypassing write path: {token!r}"


def test_propose_adr_is_the_only_tool_with_a_vector_db_write_call():
    """The acceptance criterion in prose: `propose_adr` is the sole write path; `search_existing_adrs` only reads."""
    assert "async_add_documents" not in inspect.getsource(search_existing_adrs)
    assert "async_add_documents" in inspect.getsource(propose_adr)
