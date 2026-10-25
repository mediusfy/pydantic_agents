"""Tests for the agent registry.

Contract under test (post-fix):
- Capability index supports multiple owners per capability (list, not 1:1).
- query() resolves to the first registered owner (deterministic).
- A second register() for the same capability is allowed but logs a warning.
- query() raises AgentNotFoundError on unknown capability,
  AgentExecutionError on timeout / agent failure (no more error-as-string).
- unregister() removes only the named agent from each capability's owner list,
  leaving sibling owners intact.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.core.exceptions import AgentExecutionError, AgentNotFoundError, AmbiguousCapabilityError
from pydantic_agents.core.registry import AgentRegistry

# ─── Capability Index ─────────────────────────────────────────────────────────


class TestCapabilityIndex:
    def test_register_builds_index(self):
        registry = AgentRegistry()
        registry.register("agent-a", MagicMock(), ["cap1", "cap2"])

        assert registry._cap_index["cap1"] == ["agent-a"]
        assert registry._cap_index["cap2"] == ["agent-a"]

    @pytest.mark.asyncio
    async def test_index_provides_o1_lookup(self):
        registry = AgentRegistry()
        a1 = MagicMock()
        a1.run = AsyncMock(return_value=MagicMock(output="from a1"))
        a2 = MagicMock()
        a2.run = AsyncMock(return_value=MagicMock(output="from a2"))

        registry.register("agent-two", a2, ["search"])
        registry.register("agent-one", a1, ["other-cap"])

        result = await registry.query("search", "find x")
        assert result == "from a2"
        a1.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_returns_correct_agent(self):
        registry = AgentRegistry()
        a1 = MagicMock()
        a1.run = AsyncMock(return_value=MagicMock(output="from a1"))
        a2 = MagicMock()
        a2.run = AsyncMock(return_value=MagicMock(output="from a2"))

        registry.register("agent-one", a1, ["search"])
        registry.register("agent-two", a2, ["audit"])

        result1 = await registry.query("search", "find x")
        result2 = await registry.query("audit", "check y")

        assert result1 == "from a1"
        assert result2 == "from a2"

    @pytest.mark.asyncio
    async def test_query_unknown_capability_raises(self):
        registry = AgentRegistry()
        with pytest.raises(AgentNotFoundError, match="nonexistent"):
            await registry.query("nonexistent", "anything")

    @pytest.mark.asyncio
    async def test_same_capability_raises_ambiguity_on_query(self, caplog):
        """When two agents share a capability, query() raises AmbiguousCapabilityError.

        The second registration is permitted but logs a warning so the operator
        can disambiguate.  Querying by capability now raises so the caller
        must pick — either use ``instrumented_query(capability=...)`` with an
        explicit ``owner``, or query a specific agent directly.
        """
        registry = AgentRegistry()
        a1 = MagicMock()
        a1.run = AsyncMock(return_value=MagicMock(data="first"))
        a2 = MagicMock()
        a2.run = AsyncMock(return_value=MagicMock(data="second"))

        registry.register("agent-a", a1, ["search"])
        with caplog.at_level("WARNING"):
            registry.register("agent-b", a2, ["search"])

        # Both owners are tracked, in registration order.
        assert registry._cap_index["search"] == ["agent-a", "agent-b"]
        # Querying an ambiguous capability raises.
        with pytest.raises(AmbiguousCapabilityError) as exc_info:
            await registry.query("search", "find x")
        assert exc_info.value.owners == ["agent-a", "agent-b"]
        assert "multiple agents" in str(exc_info.value)
        # An ERROR was logged about the collision.
        assert any("multiple agents" in rec.message for rec in caplog.records)

    def test_owners_of_returns_list_in_registration_order(self):
        registry = AgentRegistry()
        registry.register("alpha", MagicMock(), ["shared"])
        registry.register("beta", MagicMock(), ["shared"])

        assert registry.owners_of("shared") == ["alpha", "beta"]
        assert registry.owners_of("unknown") == []

    def test_unregister_removes_index_entries(self):
        registry = AgentRegistry()
        registry.register("agent-x", MagicMock(), ["cap-a", "cap-b", "cap-c"])

        registry.unregister("agent-x")

        assert "cap-a" not in registry._cap_index
        assert "cap-b" not in registry._cap_index
        assert "cap-c" not in registry._cap_index

    def test_unregister_preserves_shared_capability_for_other_owner(self):
        """unregister removes only the named agent from each owner list.

        Sibling owners survive — this is the post-fix contract that addresses
        the prior 'last-write-wins clobber on unregister' foot-gun.
        """
        registry = AgentRegistry()
        registry.register("alpha", MagicMock(), ["common", "alpha-only"])
        registry.register("beta", MagicMock(), ["common", "beta-only"])

        registry.unregister("alpha")

        # "common" still resolves to beta — the only remaining owner.
        assert registry._cap_index["common"] == ["beta"]
        assert "alpha-only" not in registry._cap_index
        assert registry._cap_index.get("beta-only") == ["beta"]

    def test_unregister_nonexistent_does_not_raise(self):
        registry = AgentRegistry()
        registry.unregister("ghost")

    def test_unregister_clears_initialized_flag(self):
        registry = AgentRegistry()
        registry.register("agent-z", MagicMock(), ["cap"])
        registry._initialized.add("agent-z")

        registry.unregister("agent-z")

        assert "agent-z" not in registry._initialized


# ─── query() error semantics ──────────────────────────────────────────────────


class TestQueryEdgeCases:
    @pytest.mark.asyncio
    async def test_query_timeout_raises_agent_execution_error(self):
        registry = AgentRegistry()
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=TimeoutError())
        registry.register("slow", agent, ["think"])

        with pytest.raises(AgentExecutionError) as excinfo:
            await registry.query("think", "hard problem", timeout_seconds=0.001)
        assert "timed out" in str(excinfo.value)
        assert excinfo.value.is_retryable is True

    @pytest.mark.asyncio
    async def test_query_agent_error_raises_agent_execution_error(self):
        registry = AgentRegistry()
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        registry.register("crashy", agent, ["implode"])

        with pytest.raises(AgentExecutionError) as excinfo:
            await registry.query("implode", "trigger error")
        assert "boom" in str(excinfo.value)
        # Non-retryable: the underlying model raised a logic error, not a transient one.
        assert excinfo.value.is_retryable is False

    @pytest.mark.asyncio
    async def test_query_no_data_attr_returns_str_of_result(self):
        registry = AgentRegistry()
        agent = MagicMock()
        agent.run = AsyncMock(return_value="raw string result")
        registry.register("simple", agent, ["echo"])

        result = await registry.query("echo", "hello")

        assert result == "raw string result"


# ─── dispatch_task removed ────────────────────────────────────────────────────


class TestDispatchTaskRemoved:
    def test_dispatch_task_does_not_exist(self):
        registry = AgentRegistry()
        assert not hasattr(registry, "dispatch_task")
