"""Unit tests for SQL identifier validation on pgvector collection names."""

import pytest

from pydantic_agents.core.vector_db import _ensure_table, vector_db


@pytest.mark.parametrize(
    "bad_collection",
    [
        "adrs; DROP TABLE pipeline_runs; --",
        "adrs--",
        "123bad",
        "../escape",
        "",
        "a" * 65,
    ],
)
@pytest.mark.asyncio
async def test_vector_db_rejects_invalid_collection_names(bad_collection):
    with pytest.raises(ValueError, match="Invalid collection name"):
        await _ensure_table(bad_collection)
    with pytest.raises(ValueError, match="Invalid collection name"):
        await vector_db._search(bad_collection, "query")
    with pytest.raises(ValueError, match="Invalid collection name"):
        await vector_db._add_documents(bad_collection, ["doc"], [{}], ["id1"])
