"""Vector similarity search via PostgreSQL + pgvector.

Replaces the previous ChromaDB-backed VectorDBManager with direct SQL
INSERT / SELECT using pgvector's <=> (cosine distance) operator.

Collection names map to pgvector tables ``v_<collection>``, each containing:
  - id          TEXT PRIMARY KEY
  - metadata    JSONB (any scalar-level metadata)
  - embedding   vector(<settings.embedding_dim>)
  - content     TEXT
  - content_tsv tsvector generated from ``content`` (GIN-indexed, for hybrid search)

Embeddings come from ``settings.embedding_provider``:
  - ``local``   runs sentence-transformers in-process (``settings.embedding_model``,
                default ``all-MiniLM-L6-v2`` = 384 dims). The async paths run the
                model in a worker thread so the event loop is never blocked.
  - ``litellm`` POSTs the texts to the LiteLLM proxy's OpenAI-compatible
                ``/embeddings`` endpoint (connection taken from
                ``core.models.factory._litellm_connection``).

Switching the provider, the embedding model or ``embedding_dim`` changes the
vector space, so every collection must be re-seeded afterwards (drop the
``v_<collection>`` tables and run the seed scripts again). When the configured
dimension differs from an existing table's column, ``_ensure_table`` raises
``ConfigurationError`` instead of writing incompatible vectors.

Retrieval is hybrid by default (``settings.retrieval_hybrid``): the top
candidates by cosine distance and by Postgres full-text rank are fused with
reciprocal rank fusion. With ``settings.rerank_model`` set (litellm provider
only) the fused candidates are reordered by the proxy's ``/rerank`` endpoint;
rerank failures fall back to the fused order and never raise.

All heavy dependencies (asyncpg, numpy, sentence-transformers, langchain)
are imported lazily — importing this module never fails.  Runtime paths
that touch those deps raise ``OrgitrationDependencyError`` with an install hint
when the extras are missing.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic_agents.core.exceptions import ConfigurationError, OrgitrationDependencyError
from pydantic_agents.core.settings import settings

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

# Metadata filter keys and collection names are interpolated into SQL, so they must be plain identifiers.
_METADATA_KEY = re.compile(r"[A-Za-z_]\w*", re.ASCII)
_COLLECTION_NAME = re.compile(r"[A-Za-z_]\w{0,60}", re.ASCII)


def _validate_collection_name(collection_name: str) -> str:
    """Reject collection names that are not plain ASCII SQL identifiers."""
    if not _COLLECTION_NAME.fullmatch(collection_name):
        raise ValueError(f"Invalid collection name: {collection_name!r}")
    return collection_name


# Reciprocal rank fusion constant (Cormack et al. 2009); 60 is the conventional value.
_RRF_K = 60
# Timeout for the LiteLLM /embeddings and /rerank calls.
_LITELLM_TIMEOUT_SECONDS = 30.0

# ── Embedding ──────────────────────────────────────────────────────

_embeddings: Any = None


def _require_vector_deps():
    """Import numpy lazily; raise ``OrgitrationDependencyError`` if missing.  asyncpg is a core dep.

    The ``litellm`` embedding provider needs neither numpy nor sentence-transformers.
    """
    if settings.embedding_provider == "litellm":
        return
    try:
        import numpy as _  # noqa: F401
    except ImportError as exc:
        raise OrgitrationDependencyError(
            "vector_db requires numpy.  Install with:  pip install pydantic-agents[vector]"
        ) from exc


def _get_embeddings():
    """Lazy-load the local sentence-transformers model named by ``settings.embedding_model``."""
    global _embeddings
    if _embeddings is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OrgitrationDependencyError(
                "vector_db requires sentence-transformers.  Install with:  pip install pydantic-agents[vector]"
            ) from exc

        _embeddings = SentenceTransformer(settings.embedding_model)
    return _embeddings


def _litellm_connection() -> tuple[str, str]:
    """Base URL (ending in /v1) and API key of the LiteLLM proxy; imported lazily to keep this module light."""
    from pydantic_agents.core.models.factory import _litellm_connection as _connection

    return _connection(settings.embedding_model)


async def _litellm_post(path: str, body: dict[str, Any]) -> Any:
    """POST ``body`` to ``{base_url}{path}`` on the LiteLLM proxy and return the decoded JSON."""
    import httpx

    base_url, api_key = _litellm_connection()
    async with httpx.AsyncClient(timeout=_LITELLM_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{base_url}{path}",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        return response.json()


async def _litellm_embed(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` in one OpenAI-compatible ``/embeddings`` call through the LiteLLM proxy."""
    payload = await _litellm_post("/embeddings", {"model": settings.embedding_model, "input": list(texts)})
    data = sorted(payload["data"], key=lambda item: item.get("index", 0))
    vectors = [[float(x) for x in item["embedding"]] for item in data]
    if len(vectors) != len(texts):
        raise ConfigurationError(
            f"LiteLLM /embeddings returned {len(vectors)} vectors for {len(texts)} inputs "
            f"(model {settings.embedding_model!r})"
        )
    return vectors


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` with the configured provider without blocking the event loop.

    The local model is CPU-bound, so it runs in a worker thread; the litellm provider is
    a single batched HTTP call.
    """
    if settings.embedding_provider == "litellm":
        return await _litellm_embed(texts)
    return await asyncio.to_thread(lambda: [_to_embedding(text) for text in texts])


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


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


async def with_connection(fn: Callable[[Any], Awaitable[_T]]) -> _T:
    """Run ``fn(conn)`` with a pooled connection, on the background loop that owns the cached pool.

    For callers outside this module (``run_store``, ``platform_metrics``, ``platform_findings``): the pool is bound
    to ``_bg_loop`` once any vector search has created it, so acquiring it from the caller's own loop fails.
    """

    async def _op() -> _T:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            return await fn(conn)

    return await _bridge(_op())


async def _ensure_table(collection_name: str):
    """CREATE TABLE IF NOT EXISTS for the named collection, with pgvector extension.

    Also adds the generated ``content_tsv`` column and its GIN index used by hybrid
    search, and raises ``ConfigurationError`` when an existing table's vector column
    does not match ``settings.embedding_dim`` (the operator must re-seed).
    """
    _validate_collection_name(collection_name)
    if collection_name in _initialized_tables:
        return
    dim = int(settings.embedding_dim)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS v_{collection_name} (
                id        TEXT PRIMARY KEY,
                metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dim}) NOT NULL,
                content   TEXT NOT NULL
            )
            """
        )
        # pgvector stores the declared dimension as the column's typmod (-1 when unconstrained).
        existing_dim = await conn.fetchval(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = $1::regclass AND attname = 'embedding' AND NOT attisdropped
            """,
            f"v_{collection_name}",
        )
        if existing_dim is not None and int(existing_dim) > 0 and int(existing_dim) != dim:
            raise ConfigurationError(
                f"Collection {collection_name!r} stores {int(existing_dim)}-dim vectors but EMBEDDING_DIM={dim} "
                f"(provider {settings.embedding_provider!r}, model {settings.embedding_model!r}). "
                f"Switching the embedding provider, model or dimension requires re-seeding: drop table "
                f"v_{collection_name} and run the seed scripts again."
            )
        # Index for cosine-distance (<=>) search. HNSW needs no training data: an ivfflat index built on an empty
        # table (as here, at first use) keeps meaningless centroids and loses recall until it is rebuilt. Older
        # pgvector without HNSW falls back to ivfflat.
        try:
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_hnsw
                ON v_{collection_name}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            await conn.execute(f"DROP INDEX IF EXISTS idx_{collection_name}_embedding")
        except Exception as exc:
            logger.warning("HNSW index unavailable for %s (%s); using ivfflat", collection_name, exc)
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_embedding
                ON v_{collection_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        # Full-text column + index for the lexical half of hybrid search.
        await conn.execute(
            f"""
            ALTER TABLE v_{collection_name}
            ADD COLUMN IF NOT EXISTS content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            """
        )
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{collection_name}_tsv
            ON v_{collection_name}
            USING gin (content_tsv)
            """
        )
    _initialized_tables.add(collection_name)


def _to_embedding(text: str) -> list[float]:
    """Return the local model's embedding vector for ``text`` (synchronous, CPU-bound).

    Async callers go through ``_embed_texts`` so this never runs on the event loop.
    """
    import numpy as np

    model = _get_embeddings()
    vec = model.encode(text, normalize_embeddings=True)
    if isinstance(vec, np.ndarray):
        return vec.tolist()
    return list(vec)


_TSQUERY_WORD = re.compile(r"\w+", re.ASCII)
_TSQUERY_MAX_TERMS = 32


def _or_tsquery(query: str) -> str:
    """The query's words joined with OR, as a ``to_tsquery`` string ('' when it has none).

    ``plainto_tsquery`` ANDs every word, so a request-length query (the context engine passes the whole task) almost
    never matches a chunk lexically and the text half of hybrid search returned nothing. With OR, ``ts_rank_cd``
    still ranks the chunks that match most terms first. The words are ``\\w`` characters only, so the string is a
    valid, injection-free tsquery.
    """
    words = list(dict.fromkeys(w.lower() for w in _TSQUERY_WORD.findall(query)))[:_TSQUERY_MAX_TERMS]
    return " | ".join(words)


def _rrf_fuse(rankings: list[list[str]], k: int = _RRF_K) -> list[str]:
    """Reciprocal rank fusion: score(doc) = sum over rankings of 1 / (k + rank), rank starting at 1.

    Ties keep first-appearance order (the first ranking wins), so the result is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc: scores[doc], reverse=True)


async def _rerank(query: str, documents: list[str], n_results: int) -> list[str]:
    """Reorder ``documents`` with the proxy's ``/rerank`` endpoint; on any failure keep the given order.

    Only active when ``settings.rerank_model`` is set and the provider is ``litellm``.
    """
    if not documents or not settings.rerank_model or settings.embedding_provider != "litellm":
        return documents[:n_results]
    try:
        payload = await _litellm_post(
            "/rerank",
            {"model": settings.rerank_model, "query": query, "documents": documents, "top_n": n_results},
        )
        results = sorted(payload["results"], key=lambda item: float(item["relevance_score"]), reverse=True)
        ordered = [documents[int(item["index"])] for item in results if 0 <= int(item["index"]) < len(documents)]
    except Exception as exc:  # rerank is best-effort and must never break retrieval
        logger.warning("Rerank via %r failed, keeping fused order: %s", settings.rerank_model, exc)
        return documents[:n_results]
    if not ordered:
        return documents[:n_results]
    return ordered[:n_results]


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
        _validate_collection_name(collection_name)
        try:
            _require_vector_deps()
            await _ensure_table(collection_name)
            pool = await _get_pool()
            embeddings = await _embed_texts(list(documents)) if documents else []
            rows = [
                (doc_id, _serialize_meta(meta), _vector_literal(embedding), doc_text)
                for doc_text, meta, doc_id, embedding in zip(documents, metadatas, ids, embeddings, strict=True)
            ]
            # One transaction and one round trip: a failure part-way no longer leaves half a document set behind.
            async with pool.acquire() as conn, conn.transaction():
                await conn.executemany(
                    f"""
                    INSERT INTO v_{collection_name} (id, metadata, embedding, content)
                    VALUES ($1, $2::jsonb, $3::vector, $4)
                    ON CONFLICT (id) DO UPDATE SET
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        content   = EXCLUDED.content
                    """,
                    rows,
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
        filters: dict[str, str | None] | None = None,
        embedding: list[float] | None = None,
    ) -> list[str]:
        """``embedding`` is the query's vector when the caller already computed it (``_search_domains``)."""
        _validate_collection_name(collection_name)
        # `project` is the long-standing filter; `filters` adds any other metadata equality filters.
        conditions: dict[str, str | None] = dict(filters or {})
        if project:
            conditions["project"] = project
        for key in conditions:
            if not _METADATA_KEY.fullmatch(key):
                raise ValueError(f"Invalid metadata filter key: {key!r}")
        try:
            _require_vector_deps()
            await _ensure_table(collection_name)
            if embedding is None:
                (embedding,) = await _embed_texts([query])
            embedding_str = _vector_literal(embedding)
            pool = await _get_pool()
            # Both candidate queries take ($1 = query vector / query text, $2 = LIMIT, $3.. = metadata values),
            # so one clause list serves both. A None value matches rows that lack the key (e.g. entries
            # written before it was recorded).
            clauses: list[str] = []
            params: list[str] = []
            for key, value in conditions.items():
                if value is None:
                    clauses.append(f"metadata->>'{key}' IS NULL")
                else:
                    params.append(value)
                    clauses.append(f"metadata->>'{key}' = ${len(params) + 2}")
            hybrid = bool(settings.retrieval_hybrid)
            limit = max(20, 4 * n_results) if hybrid else n_results
            where = f"WHERE {' AND '.join(clauses)}\n" if clauses else ""
            async with pool.acquire() as conn:
                vector_rows = await conn.fetch(
                    f"""
                    SELECT content
                    FROM v_{collection_name}
                    {where}ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    embedding_str,
                    limit,
                    *params,
                )
                by_vector = [row["content"] for row in vector_rows]
                if not hybrid:
                    return by_vector[:n_results]
                by_text: list[str] = []
                if ts_query := _or_tsquery(query):
                    text_clauses = [*clauses, "content_tsv @@ to_tsquery('english', $1)"]
                    text_rows = await conn.fetch(
                        f"""
                        SELECT content
                        FROM v_{collection_name}
                        WHERE {" AND ".join(text_clauses)}
                        ORDER BY ts_rank_cd(content_tsv, to_tsquery('english', $1)) DESC
                        LIMIT $2
                        """,
                        ts_query,
                        limit,
                        *params,
                    )
                    by_text = [row["content"] for row in text_rows]
            fused = _rrf_fuse([by_vector, by_text])
            return await _rerank(query, fused, n_results)
        except OrgitrationDependencyError as exc:
            # Return nothing rather than an explanatory string: callers paste results straight into prompts.
            logger.warning("Vector DB search called but vector dependencies are not installed: %s", exc)
            return []

    def search(self, collection_name: str, query: str, n_results: int = 3, project: str | None = None) -> list[str]:
        """Synchronous convenience wrapper (avoids event-loop deadlock).

        ``project`` restricts results to documents tagged with that project at seed time.
        """
        return self._run_sync(self._search(collection_name, query, n_results, project))

    async def _search_domains(
        self, collections: list[str], query: str, n_results: int, project: str | None
    ) -> list[list[str] | BaseException]:
        """Search several collections for one query: it is embedded once, and the collections are queried
        concurrently. A collection that fails yields its exception in its slot instead of failing the others."""
        try:
            _require_vector_deps()
            (embedding,) = await _embed_texts([query])
        except OrgitrationDependencyError as exc:
            logger.warning("Vector DB search called but vector dependencies are not installed: %s", exc)
            return [[] for _ in collections]
        return await asyncio.gather(
            *(self._search(c, query, n_results, project, embedding=embedding) for c in collections),
            return_exceptions=True,
        )

    async def async_search_domains(
        self, collections: list[str], query: str, n_results: int = 3, project: str | None = None
    ) -> list[list[str] | BaseException]:
        """``_search`` over several collections with one query embedding (see ``_search_domains``)."""
        return await _bridge(self._search_domains(collections, query, n_results, project))

    async def async_search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 3,
        project: str | None = None,
        filters: dict[str, str | None] | None = None,
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
            from pydantic_agents.core.tools.web_fetch import validate_public_http_url

            err = validate_public_http_url(url)
            if err:
                raise ValueError(err)
            _require_vector_deps()
            loader = WebBaseLoader(url, requests_kwargs={"timeout": 15, "allow_redirects": False})
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            texts = [d.page_content for d in splits]
            metas = [{"source": url} for _ in splits]
            ids = [
                f"url_{hashlib.sha256(f'{url}:{i}:{s.page_content}'.encode()).hexdigest()[:16]}"
                for i, s in enumerate(splits)
            ]

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
            ids = [
                f"pdf_{hashlib.sha256(f'{file_path}:{i}:{s.page_content}'.encode()).hexdigest()[:16]}"
                for i, s in enumerate(splits)
            ]

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
            from pydantic_agents.core.tools.web_fetch import validate_public_http_url

            err = validate_public_http_url(url)
            if err:
                raise ValueError(err)
            _require_vector_deps()
            loader = WebBaseLoader(url, requests_kwargs={"timeout": 15, "allow_redirects": False})
            docs = await asyncio.to_thread(loader.load)  # blocking HTTP: not on the event loop
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            texts = [d.page_content for d in splits]
            metas = [{"source": url} for _ in splits]
            ids = [
                f"url_{hashlib.sha256(f'{url}:{i}:{s.page_content}'.encode()).hexdigest()[:16]}"
                for i, s in enumerate(splits)
            ]

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
            docs = await asyncio.to_thread(loader.load)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            texts = [d.page_content for d in splits]
            metas = [{"source": file_path} for _ in splits]
            ids = [
                f"pdf_{hashlib.sha256(f'{file_path}:{i}:{s.page_content}'.encode()).hexdigest()[:16]}"
                for i, s in enumerate(splits)
            ]

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
