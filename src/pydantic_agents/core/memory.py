import asyncio
import datetime
import hashlib
import logging
from typing import Any

from pydantic import BaseModel, Field

from pydantic_agents.core.settings import settings
from pydantic_agents.core.vector_db import vector_db

logger = logging.getLogger(__name__)


def _flatten_metadata(meta: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict values for ChromaDB (requires scalar values only).

    Converts {"content": "...", "metadata": {"type": "x", "agent": "y"}}
    into {"content": "...", "metadata.type": "x", "metadata.agent": "y"}.
    """
    flat: dict[str, Any] = {}
    for key, value in meta.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metadata(value, full_key))
        else:
            flat[full_key] = value
    return flat


class Memory(BaseModel):
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorMemory:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    async def add_memory(self, content: str, metadata: dict[str, Any] | None = None):
        """Adds a new memory to the vector store with a stable content-based ID."""
        memory = Memory(content=content, metadata=metadata or {})
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        doc_id = f"memory_{memory.timestamp}_{content_hash}"
        # Flatten metadata for ChromaDB compatibility — Chroma requires
        # scalar values (str, int, float, bool), not nested dicts.
        chroma_metadata = _flatten_metadata(memory.model_dump())
        await vector_db.async_add_documents(self.collection_name, [content], [chroma_metadata], [doc_id])

    async def get_relevant_memories(self, query: str, n_results: int = 5) -> list[Memory]:
        """Retrieves the most relevant memories for a given query."""
        results = await vector_db.async_search(self.collection_name, query, n_results=n_results)
        return [Memory(content=res) for res in results]

    async def consolidate(self, n_recent: int = 50) -> str | None:
        """Fetch recent memories and return them as a single text block for summarization.

        Callers (agents or scheduled tasks) can pass the result to an LLM for
        summarization, then call add_memory() with the summary.
        Returns None if no memories found.
        """
        recent = await self.get_relevant_memories("activity:", n_results=n_recent)
        if not recent:
            logger.info("No memories to consolidate.")
            return None

        lines = []
        for mem in recent:
            ts = mem.timestamp[:19] if len(mem.timestamp) > 19 else mem.timestamp
            meta = ""
            if mem.metadata:
                mtype = mem.metadata.get("type", "")
                agent = mem.metadata.get("agent_name", "")
                meta = f" [{mtype} by {agent}]".strip() if mtype or agent else ""
            lines.append(f"[{ts}]{meta} {mem.content}")

        logger.info("Fetched %d recent memories for consolidation.", len(lines))
        return "\n".join(lines)


# Global memory instance — uses same collection as tools/memory.py log_activity
long_term_memory = VectorMemory("activity_logs")


class RedisIrisMemory:
    """Redis-backed long-term memory via redis-agent-memory package.

    Provides persistent, distributed memory storage. Falls back gracefully
    to VectorMemory if redis-agent-memory is not installed or unreachable.
    """

    def __init__(self, server_url: str, store_id: str, api_key: str):
        self.server_url = server_url
        self.store_id = store_id
        self.api_key = api_key

    @staticmethod
    def _require_package() -> None:
        """Lazy import: raises ImportError if redis-agent-memory is not installed."""
        import redis_agent_memory  # noqa: F401

    async def add_memory(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a memory to the Redis IRIS store."""
        await asyncio.to_thread(self._require_package)  # the first import is blocking disk I/O
        logger.warning(
            "RedisIrisMemory.add_memory is not implemented; dropped %d characters (metadata keys: %s)",
            len(content),
            sorted(metadata or {}),
        )

    async def get_relevant_memories(self, query: str, n_results: int = 5) -> list["Memory"]:
        """Retrieve relevant memories from the Redis IRIS store."""
        await asyncio.to_thread(self._require_package)
        raise NotImplementedError(
            "redis_agent_memory integration not yet implemented "
            f"(query of {len(query)} characters, n_results={n_results})"
        )


_memory_backend: VectorMemory | RedisIrisMemory | None = None


def create_memory_backend() -> VectorMemory | RedisIrisMemory:
    """Create the appropriate memory backend based on configuration.

    Returns RedisIrisMemory if REDIS_IRIS_TOKEN is configured and the
    redis-agent-memory package is available. Falls back to VectorMemory
    otherwise.
    """
    if settings.redis_iris_token:
        try:
            return RedisIrisMemory(
                server_url=settings.redis_iris_url or "http://localhost:6380",
                store_id=settings.redis_iris_store_id or "orchestration",
                api_key=settings.redis_iris_token,
            )
        except Exception:
            logger.warning("Failed to initialize RedisIrisMemory, falling back to VectorMemory", exc_info=True)
    return VectorMemory("activity_logs")


def get_memory_backend() -> VectorMemory | RedisIrisMemory:
    """Get the singleton memory backend, creating it lazily on first call."""
    global _memory_backend
    if _memory_backend is None:
        _memory_backend = create_memory_backend()
    return _memory_backend
