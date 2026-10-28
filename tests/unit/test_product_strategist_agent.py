"""product_strategist_agent (AP-82, ADR-0001): registration, its own tools called directly (no LLM) against
stubbed `pipeline_runs`/`platform_findings` pools, and the AP-82 acceptance criterion that the module has no
write path at all (report-only output).
"""

import inspect
import json
from unittest.mock import MagicMock

import pytest

from pydantic_agents.agents import product_strategist
from pydantic_agents.agents.product_strategist import (
    open_findings,
    product_strategist_agent,
    recent_requirements,
)
from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core import platform_findings, run_store
from pydantic_agents.core.registry import registry
from pydantic_agents.core.tool_assignments import AGENT_TOOL_CATEGORIES
from tests.unit.db_stub import stub_database

CTX = MagicMock()

# Any write-shaped call this report-only agent must never make: creating tickets/PRs, writing code, or
# triggering/approving pipeline work. None of these may appear anywhere in the module's source.
FORBIDDEN_WRITE_TOKENS = [
    "save_run",
    "save_finding",
    "write_approval",
    "create_issue",
    "create_pull_request",
    "run_orchestration_pipeline",
    "scaffold_and_build_service",
]


class _Conn:
    """Answers both `pipeline_runs` (run_store) and `platform_findings` SQL over one shared ``db`` dict, matching
    production where both share the one real database pool via ``vector_db._get_pool()``.
    """

    def __init__(self, db):
        self.db = db

    async def execute(self, sql, *args):
        if "INSERT INTO platform_findings" in sql:
            finding_id, kind, severity, agent, summary, evidence = args
            self.db[finding_id] = {
                "id": finding_id,
                "kind": kind,
                "severity": severity,
                "agent": agent,
                "summary": summary,
                "evidence": evidence,
                "status": "open",
                "_t": "f",
            }

    async def fetch(self, sql, *args):
        if "pipeline_runs" in sql:
            rows = sorted((r for r in self.db.values() if r["_t"] == "r"), key=lambda r: r["run_id"], reverse=True)
            (limit,) = args
            return rows[:limit]
        rows = sorted((r for r in self.db.values() if r["_t"] == "f"), key=lambda r: r["id"], reverse=True)
        *filters, limit = args
        if "status =" in sql:
            (status,) = filters
            rows = [r for r in rows if r["status"] == status]
        return rows[:limit]


@pytest.fixture
def pool(monkeypatch):
    stub_database(monkeypatch, run_store, _Conn)
    return stub_database(monkeypatch, platform_findings, _Conn)


def test_product_strategist_is_registered():
    register_all_agents()
    assert "product_strategist" in registry.list_agents()
    assert registry.get("product_strategist") is product_strategist_agent


def test_product_strategist_has_a_tool_category_entry():
    assert "product_strategist" in AGENT_TOOL_CATEGORIES
    assert AGENT_TOOL_CATEGORIES["product_strategist"] == ["memory"]


async def test_recent_requirements_returns_json(pool):
    pool.db["run_1"] = {
        "run_id": "run_1",
        "project": "demo",
        "requirement": "Add dark mode",
        "status": "completed",
        "completed_at": None,
        "_t": "r",
    }

    rows = json.loads(await recent_requirements(CTX, limit=10))

    assert rows[0]["requirement"] == "Add dark mode"


async def test_open_findings_returns_json(pool):
    await platform_findings.save_finding("health", "low", "db pool exhausted")

    rows = json.loads(await open_findings(CTX, limit=10))

    assert rows[0]["summary"] == "db pool exhausted"


@pytest.mark.parametrize("token", FORBIDDEN_WRITE_TOKENS)
def test_no_source_reference_to_a_write_path(token):
    source = inspect.getsource(product_strategist)
    assert token not in source, f"product_strategist.py references a write path: {token!r}"


def test_the_agent_has_both_of_its_own_read_tools():
    tool_names = set(product_strategist_agent._function_toolset.tools)
    assert {"recent_requirements", "open_findings"} <= tool_names


def test_no_granted_or_own_tool_writes_to_the_filesystem_or_repo():
    """AP-82 acceptance in prose: output is a report structure only, no direct repo/Jira writes. The
    `documentation` category is deliberately excluded from this agent's categories (see CHANGELOG) because
    `update_documentation` is a real filesystem write; this guards against it being re-added.
    """
    tool_names = set(product_strategist_agent._function_toolset.tools)
    assert "update_documentation" not in tool_names
    assert "documentation" not in AGENT_TOOL_CATEGORIES["product_strategist"]
