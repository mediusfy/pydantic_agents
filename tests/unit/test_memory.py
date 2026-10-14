"""Tests for the VectorMemory class.

These tests cover the meaningful behaviour changes in memory.py:
- The hash is stable across calls (same input = same hash prefix)
- consolidate() method returns None on empty store
- consolidate() formats memories into a text block with timestamps
- The global long_term_memory uses "activity_logs" collection
"""

import hashlib
from unittest.mock import AsyncMock, patch

import pytest

from pydantic_agents.core.memory import VectorMemory

# ─── Content hashing ──────────────────────────────────────────────────────────


class TestContentHashing:
    """Hash stability: same input must produce same 12-char md5 prefix."""

    def test_hash_is_deterministic(self):
        """Same content always produces the same hash prefix."""
        content = "The quick brown fox jumps over the lazy dog"
        h1 = hashlib.md5(content.encode()).hexdigest()[:12]
        h2 = hashlib.md5(content.encode()).hexdigest()[:12]
        assert h1 == h2

    def test_hash_differs_for_different_content(self):
        h1 = hashlib.md5(b"hello").hexdigest()[:12]
        h2 = hashlib.md5(b"world").hexdigest()[:12]
        assert h1 != h2

    def test_hash_prefix_is_12_hex_chars(self):
        prefix = hashlib.md5(b"hello").hexdigest()[:12]
        assert len(prefix) == 12
        assert all(c in "0123456789abcdef" for c in prefix)

    def test_add_memory_embeds_hash(self):
        """add_memory produces a doc_id containing a hash of the content."""
        content = "some memorable event"
        expected_hash = hashlib.md5(content.encode()).hexdigest()[:12]

        import datetime

        ts = datetime.datetime.now().isoformat()
        doc_id = f"memory_{ts}_{expected_hash}"
        assert expected_hash in doc_id


# ─── consolidate() method ─────────────────────────────────────────────────────


class TestVectorMemoryConsolidate:
    """consolidate() method behaviour."""

    @pytest.mark.asyncio
    async def test_consolidate_empty_returns_none(self):
        """When no memories match, consolidate returns None."""
        vm = VectorMemory("test_collection")

        with patch.object(vm, "get_relevant_memories", AsyncMock(return_value=[])):
            result = await vm.consolidate(n_recent=5)

        assert result is None

    @pytest.mark.asyncio
    async def test_consolidate_formats_timestamp_and_content(self):
        """Each memory is formatted as '[timestamp] content'."""
        from pydantic_agents.core.memory import Memory

        vm = VectorMemory("test_collection")
        fake_memories = [
            Memory(content="first note", timestamp="2025-05-22T10:00:00.000000"),
            Memory(content="second note", timestamp="2025-05-22T10:05:00.000000"),
        ]

        with patch.object(vm, "get_relevant_memories", AsyncMock(return_value=fake_memories)):
            result = await vm.consolidate(n_recent=10)

        assert result is not None
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "[2025-05-22T10:00:00] first note"
        assert lines[1] == "[2025-05-22T10:05:00] second note"

    @pytest.mark.asyncio
    async def test_consolidate_includes_metadata_when_present(self):
        """Metadata type and agent_name are rendered as [type by agent]."""
        from pydantic_agents.core.memory import Memory

        vm = VectorMemory("test_collection")
        fake_memories = [
            Memory(
                content="agent reported progress",
                timestamp="2025-05-22T10:00:00",
                metadata={"type": "observation", "agent_name": "backend"},
            ),
        ]

        with patch.object(vm, "get_relevant_memories", AsyncMock(return_value=fake_memories)):
            result = await vm.consolidate(n_recent=10)

        assert result is not None
        # The meta tag has no leading space due to .strip()
        assert result == "[2025-05-22T10:00:00][observation by backend] agent reported progress"

    @pytest.mark.asyncio
    async def test_consolidate_handles_timestamps_shorter_than_19_chars(self):
        """Short timestamps (no microseconds) are used as-is."""
        from pydantic_agents.core.memory import Memory

        vm = VectorMemory("test_collection")
        fake_memories = [
            Memory(content="short", timestamp="2025-05-22T10:00:00"),
        ]

        with patch.object(vm, "get_relevant_memories", AsyncMock(return_value=fake_memories)):
            result = await vm.consolidate(n_recent=10)

        assert result == "[2025-05-22T10:00:00] short"

    @pytest.mark.asyncio
    async def test_consolidate_handles_empty_metadata(self):
        """No metadata bracket when metadata is empty."""
        from pydantic_agents.core.memory import Memory

        vm = VectorMemory("test_collection")
        fake_memories = [
            Memory(content="bare note", timestamp="2025-05-22T10:00:00", metadata={}),
        ]

        with patch.object(vm, "get_relevant_memories", AsyncMock(return_value=fake_memories)):
            result = await vm.consolidate(n_recent=10)

        assert result == "[2025-05-22T10:00:00] bare note"

    @pytest.mark.asyncio
    async def test_consolidate_n_recent_passed_through(self):
        """n_recent is forwarded to get_relevant_memories."""
        vm = VectorMemory("test_collection")

        with patch.object(vm, "get_relevant_memories", AsyncMock(return_value=[])) as mock_get:
            await vm.consolidate(n_recent=99)

        mock_get.assert_awaited_once_with("activity:", n_results=99)

    @pytest.mark.asyncio
    async def test_consolidate_returns_none_on_no_results(self):
        """Edge case: if query returns nothing, method returns None."""
        vm = VectorMemory("test_collection")

        with patch.object(vm, "get_relevant_memories", AsyncMock(return_value=[])):
            result = await vm.consolidate()

        assert result is None


# ─── Global instance ──────────────────────────────────────────────────────────


class TestVectorMemoryGlobalInstance:
    def test_global_collection_name_is_activity_logs(self):
        from pydantic_agents.core.memory import long_term_memory

        assert long_term_memory.collection_name == "activity_logs"

    def test_global_is_vector_memory_instance(self):
        from pydantic_agents.core.memory import long_term_memory

        assert isinstance(long_term_memory, VectorMemory)


# ─── add_memory doc_id construction ───────────────────────────────────────────


class TestAddMemory:
    @pytest.mark.asyncio
    async def test_add_memory_calls_vector_db(self):
        """add_memory delegates to vector_db.async_add_documents with correct args."""
        vm = VectorMemory("test_collection")

        with patch("pydantic_agents.core.memory.vector_db.async_add_documents", AsyncMock()) as mock_add:
            await vm.add_memory("hello world", metadata={"key": "val"})

        mock_add.assert_awaited_once()
        args, _ = mock_add.await_args
        collection_name, documents, metadatas, doc_ids = args
        assert collection_name == "test_collection"
        assert documents == ["hello world"]
        assert metadatas[0]["content"] == "hello world"
        assert metadatas[0]["metadata.key"] == "val"
        # doc_id contains a hash prefix of the content
        expected_hash = hashlib.md5(b"hello world").hexdigest()[:12]
        assert expected_hash in doc_ids[0]
