"""Vector similarity search via PostgreSQL + pgvector.

Replaces the previous ChromaDB-backed VectorDBManager with direct SQL
INSERT / SELECT using pgvector's <=> (cosine distance) operator.

Collection names map to pgvector tables, each containing:
  - id        TEXT PRIMARY KEY
  - metadata  JSONB (any scalar-level metadata)
  - embedding vector(384)  — all-MiniLM-L6-v2 produces 384-dim vectors
  - content   TEXT

All heavy dependencies (asyncpg, numpy, sentence-transformers, langchain)
are imported lazily — importing this module never fails.  Runtime paths
that touch those deps raise ``OrgitrationDependencyError`` with an install hint
when the extras are missing.
"""

import asyncio
import json
import logging
import os
import re
import threading
from typing import Any

from pydantic_agents.core.exceptions import OrgitrationDependencyError
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

# Metadata filter keys are interpolated into SQL, so they must be plain identifiers.
_METADATA_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# ── Embedding ──────────────────────────────────────────────────────

_embeddings: Any = None


def _require_vector_deps():
    """Import numpy lazily; raise ``OrgitrationDependencyError`` if missing.  asyncpg is a core dep."""
    _missing = []

    try:
        import numpy as _  # noqa: F401
    except ImportError as exc:
        raise OrgitrationDependencyError(
            "vector_db requires numpy.  Install with:  pip install pydantic-agents[vector]"
        ) from exc


def _get_embeddings():
    """Lazy-load HuggingFace embeddings (same model as before, 384-dim)."""
    global _embeddings
    if _embeddings is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OrgitrationDependencyError(
                "vector_db requires sentence-transformers.  Install with:  pip install pydantic-agents[vector]"
            ) from exc

        _embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    return _embeddings


# ── pgvector helpers ───────────────────────────────────────────────

_connection_pools: dict[str, Any] = {}
_initialized_tables: set[str] = set()

# Persistent background loop that every sync call bridges onto. The asyncpg
# pool cached in _get_pool() is bound to whichever loop created it; bridging
# via a fresh asyncio.run() loop per call (the old behavior) left that pool
# bound to a loop that was closed as soon as the call returned, so the next
# call's queries died with "Event loop is closed". Routing every call
# through the same never-closed loop keeps the pool valid across calls.
_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None
_bg_lock = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent background event loop, starting it on first use."""
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is None or not _bg_loop.is_running():
            _bg_loop = asyncio.new_event_loop()
            _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
            _bg_thread.start()
        return _bg_loop


def _bridge(coro: Any) -> "asyncio.Future[Any]":
    """Run a coroutine on the persistent background loop, awaitable from any loop.

    The cached asyncpg pool lives on ``_bg_loop``, so every pool-touching call —
    sync or async entrypoint — must execute there too. wrap_future() exposes the
    threadsafe future as an awaitable on the caller's own loop instead of
    blocking it, which run_coroutine_threadsafe(...).result() would do.
    """
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return asyncio.wrap_future(future)


# Test-mode override for _get_pool — set by _set_test_pool() during testing.
# When set, _get_pool returns this pool instead of connecting to PostgreSQL.
_test_pool_override: Any = None


def _set_test_pool(pool: Any):
    """Override _get_pool for testing (bypasses real PostgreSQL).

    The ``pool`` parameter is typed loosely to avoid requiring an
    ``asyncpg.Pool`` import at module scope.  Pass any pool-like object
    that can be used as a context manager via ``.acquire()``.
    """
    global _test_pool_override
    _test_pool_override = pool


def _dsn() -> str:
    """Return the PostgreSQL connection string."""
    return settings.database_url or os.environ.get("DATABASE_URL", "postgresql://localhost:5432/orchestration")


async def _get_pool():
    """Return a singleton asyncpg connection pool for the configured DSN.

    In test mode (when _set_test_pool() was called), returns the overridden pool.
    """
    if _test_pool_override is not None:
        return _test_pool_override

    import asyncpg

    dsn = _dsn()
    if dsn not in _connection_pools:
        _connection_pools[dsn] = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    return _connection_pools[dsn]


async def _ensure_table(collection_name: str):
    """CREATE TABLE IF NOT EXISTS for the named collection, with pgvector extension."""
    if collection_name in _initialized_tables:
        return
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS v_{collection_name} (
                id        TEXT PRIMARY KEY,
                metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector(384) NOT NULL,
                content   TEXT NOT NULL
            )
            """
        )
        # Index for cosine-distance (<=>) search — ivfflat with cosine distance
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{collection_name}_embedding
            ON v_{collection_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )
    _initialized_tables.add(collection_name)


def _to_embedding(text: str) -> list[float]:
    """Return a 384-dim embedding vector for the given text."""
    import numpy as np

    model = _get_embeddings()
    vec = model.encode(text, normalize_embeddings=True)
    if isinstance(vec, np.ndarray):
        return vec.tolist()
    return list(vec)


# ── Public interface (same API as old Chroma-based VectorDBManager) ─


class VectorDBManager:
    """Drop-in replacement for the old ChromaDB-backed VectorDBManager.

    All same public methods: add_documents, search, async_add_documents,
    async_search, ingest_url, ingest_pdf, async_ingest_url, async_ingest_pdf.
    """

    @staticmethod
    def _run_sync(coro):
        """Run a coroutine from synchronous code, bridging onto the persistent background loop."""
        loop = _get_background_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    async def _add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ):
        try:
            _require_vector_deps()
            await _ensure_table(collection_name)
            pool = await _get_pool()
            async with pool.acquire() as conn:
                for doc_text, meta, doc_id in zip(documents, metadatas, ids, strict=True):
                    embedding = _to_embedding(doc_text)
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    await conn.execute(
                        f"""
                        INSERT INTO v_{collection_name} (id, metadata, embedding, content)
                        VALUES ($1, $2::jsonb, $3::vector, $4)
                        ON CONFLICT (id) DO UPDATE SET
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            content   = EXCLUDED.content
                        """,
                        doc_id,
                        _serialize_meta(meta),
                        embedding_str,
                        doc_text,
                    )
        except OrgitrationDependencyError as exc:
            logger.warning("Vector DB add_documents called but vector dependencies are not installed: %s", exc)

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ):
        """Synchronous convenience wrapper (avoids event-loop deadlock)."""
        return self._run_sync(self._add_documents(collection_name, documents, metadatas, ids))

    async def async_add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ):
        await _bridge(self._add_documents(collection_name, documents, metadatas, ids))

    async def _search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 3,
        project: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[str]:
        # `project` is the long-standing filter; `filters` adds any other metadata equality filters.
        conditions = dict(filters or {})
        if project:
            conditions["project"] = project
        for key in conditions:
            if not _METADATA_KEY.fullmatch(key):
                raise ValueError(f"Invalid metadata filter key: {key!r}")
        try:
            _require_vector_deps()
            await _ensure_table(collection_name)
            embedding = _to_embedding(query)
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            pool = await _get_pool()
            where = "".join(
                f"{'WHERE' if i == 0 else 'AND'} metadata->>'{key}' = ${i + 3}\n" for i, key in enumerate(conditions)
            )
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT content
                    FROM v_{collection_name}
                    {where}ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    embedding_str,
                    n_results,
                    *conditions.values(),
                )
                return [row["content"] for row in rows]
        except OrgitrationDependencyError as exc:
            logger.warning("Vector DB search called but vector dependencies are not installed: %s", exc)
            return ["Vector DB not available — install heavy dependencies (sentence-transformers, asyncpg, pgvector)"]

    def search(self, collection_name: str, query: str, n_results: int = 3, project: str | None = None) -> list[str]:
        """Synchronous convenience wrapper (avoids event-loop deadlock).

        ``project`` restricts results to documents tagged with that project at seed time.
        """
        return self._run_sync(self._search(collection_name, query, n_results, project))

    async def async_search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 3,
        project: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[str]:
        return await _bridge(self._search(collection_name, query, n_results, project, filters))

    def ingest_url(self, collection_name: str, url: str) -> str:
        """Fetch a URL, split into chunks, and index into pgvector."""
        try:
            from langchain_community.document_loaders import WebBaseLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise OrgitrationDependencyError(
                "vector_db.ingest_url requires extra dependencies. Install with:  pip install pydantic-agents[ingest]"
            ) from exc
        else:
            _require_vector_deps()
            loader = WebBaseLoader(url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            texts = [d.page_content for d in splits]
            metas = [{"source": url} for _ in splits]
            ids = [f"url_{id(s)}" for s in splits]

            self._run_sync(self._add_documents(collection_name, texts, metas, ids))
            return f"Ingested {len(splits)} chunks from {url}"

    def ingest_pdf(self, collection_name: str, file_path: str) -> str:
        """Load a PDF, split into chunks, and index into pgvector."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise OrgitrationDependencyError(
                "vector_db.ingest_pdf requires extra dependencies. Install with:  pip install pydantic-agents[ingest]"
            ) from exc
        else:
            _require_vector_deps()
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            texts = [d.page_content for d in splits]
            metas = [{"source": file_path} for _ in splits]
            ids = [f"pdf_{id(s)}" for s in splits]

            self._run_sync(self._add_documents(collection_name, texts, metas, ids))
            return f"Ingested {len(splits)} chunks from PDF."

    async def async_ingest_url(self, collection_name: str, url: str) -> str:
        try:
            from langchain_community.document_loaders import WebBaseLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise OrgitrationDependencyError(
                "vector_db.async_ingest_url requires extra dependencies. "
                "Install with:  pip install pydantic-agents[ingest]"
            ) from exc
        else:
            _require_vector_deps()
            loader = WebBaseLoader(url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            texts = [d.page_content for d in splits]
            metas = [{"source": url} for _ in splits]
            ids = [f"url_{id(s)}" for s in splits]

            await _bridge(self._add_documents(collection_name, texts, metas, ids))
            return f"Ingested {len(splits)} chunks from {url}"

    async def async_ingest_pdf(self, collection_name: str, file_path: str) -> str:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise OrgitrationDependencyError(
                "vector_db.async_ingest_pdf requires extra dependencies. "
                "Install with:  pip install pydantic-agents[ingest]"
            ) from exc
        else:
            _require_vector_deps()
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            texts = [d.page_content for d in splits]
            metas = [{"source": file_path} for _ in splits]
            ids = [f"pdf_{id(s)}" for s in splits]

            await _bridge(self._add_documents(collection_name, texts, metas, ids))
            return f"Ingested {len(splits)} chunks from PDF."


# ── Singleton ──────────────────────────────────────────────────────

vector_db = VectorDBManager()


# ── Helpers ────────────────────────────────────────────────────────


def _serialize_meta(meta: dict[str, Any] | None) -> str:
    """Serialize metadata dict to JSON string for JSONB column.

    Handles nested dicts by flattening them (ChromaDB required scalar metadata;
    pgvector's JSONB happily takes nested structures).

    All values are coerced to JSON-serializable types.
    """
    if meta is None:
        return "{}"
    cleaned: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif v is None:
            cleaned[k] = None
        else:
            try:
                json.dumps(v)
                cleaned[k] = v
            except (TypeError, ValueError):
                cleaned[k] = str(v)
    return json.dumps(cleaned)
