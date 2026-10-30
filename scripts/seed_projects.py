#!/usr/bin/env python3
"""Refresh the agents' vector knowledge base from recently active GitHub repos.

For each non-archived repo in ``--org`` pushed within ``--days`` (minus
``--exclude``), shallow-clone or update it under ``--clones-dir``, refresh that
project's chunks with ``seed_project_files`` (upsert), then delete only chunks whose
ids no longer exist upstream. Any seeding error aborts before anything is deleted.

Dry run by default (prints what would be indexed). Pass ``--apply`` to write; that
needs DATABASE_URL pointing at the pgvector ``agents`` database, e.g. through
``kubectl -n pgvector port-forward pod/pgvector-0 15432:5432``.

    DATABASE_URL=postgresql://...@127.0.0.1:15432/agents \\
        python scripts/seed_projects.py --apply
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from pydantic_agents.core.utils import safe_path_join

COLLECTIONS = ("adrs", "architecture", "conventions", "project")
_REPO_NAME = re.compile(r"[A-Za-z0-9_.\-]+", re.ASCII)


def _validate_repo_identifier(value: str, label: str = "name") -> str:
    if not _REPO_NAME.fullmatch(value) or value in (".", ".."):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def active_repos(org: str, days: int, exclude: set[str]) -> list[str]:
    _validate_repo_identifier(org, "org")
    out = subprocess.check_output(
        ["gh", "repo", "list", org, "--limit", "500", "--json", "name,pushedAt,isArchived"],
        text=True,
        timeout=60,
    )
    cutoff = datetime.now(UTC) - timedelta(days=days)
    names = [
        _validate_repo_identifier(r["name"], "repo name")
        for r in json.loads(out)
        if not r["isArchived"]
        and r["name"] not in exclude
        and datetime.fromisoformat(r["pushedAt"].replace("Z", "+00:00")) >= cutoff
    ]
    return sorted(names)


def sync_clone(org: str, name: str, clones_dir: Path) -> Path:
    _validate_repo_identifier(org, "org")
    _validate_repo_identifier(name, "repo name")
    dest = safe_path_join(clones_dir, name)
    if (dest / ".git").is_dir():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only", "-q"], check=False, timeout=120)
    else:
        subprocess.run(
            ["gh", "repo", "clone", f"{org}/{name}", str(dest), "--", "--depth", "1", "-q"],
            check=True,
            timeout=120,
        )
    return dest


async def project_ids(project: str) -> set[tuple[str, str]]:
    from pydantic_agents.core.vector_db import _get_pool

    pool = await _get_pool()
    found: set[tuple[str, str]] = set()
    async with pool.acquire() as conn:
        for c in COLLECTIONS:
            if await conn.fetchval("SELECT to_regclass($1)", f"v_{c}"):
                rows = await conn.fetch(f"SELECT id FROM v_{c} WHERE metadata->>'project' = $1", project)
                found.update((c, r["id"]) for r in rows)
    return found


async def delete_ids(stale: set[tuple[str, str]]) -> None:
    from pydantic_agents.core.vector_db import _get_pool

    pool = await _get_pool()
    async with pool.acquire() as conn:
        for collection, doc_id in stale:
            await conn.execute(f"DELETE FROM v_{collection} WHERE id = $1", doc_id)


def refresh_project(mcp_server, name: str, path: Path) -> str:
    """Upsert the project's chunks, then delete only ids that no longer exist upstream.

    All DB access goes through vector_db's persistent background loop (``_run_sync``);
    using ``asyncio.run`` here would bind the shared asyncpg pool to a loop that then closes.
    """
    vdb = mcp_server.vector_db
    before = vdb._run_sync(project_ids(name))
    written: set[tuple[str, str]] = set()
    real_add = vdb.add_documents

    def recording_add(collection, docs, metas, ids):
        written.update((collection, i) for i in ids)
        return real_add(collection, docs, metas, ids)

    with patch.object(vdb, "add_documents", side_effect=recording_add):
        summary = mcp_server.seed_project_files(str(path), name)
    if "Errors:" in summary:
        raise RuntimeError(f"{name}: seeding failed, nothing deleted\n{summary}")
    stale = before - written
    if stale:
        vdb._run_sync(delete_ids(stale))
    return f"{summary.splitlines()[0]}; upserted {len(written)} chunks, removed {len(stale)} stale"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", default="mediusfy")
    ap.add_argument("--days", type=int, default=31, help="repos pushed within this many days (default 31)")
    ap.add_argument("--exclude", nargs="*", default=["chatbot"], help="repo names to skip (default: chatbot)")
    ap.add_argument("--only", nargs="*", help="restrict to these repo names")
    ap.add_argument("--clones-dir", type=Path, help="reuse clones here (default: a temp dir)")
    ap.add_argument("--apply", action="store_true", help="write to the vector DB (default: dry run)")
    args = ap.parse_args()

    from pydantic_agents import mcp_server

    repos = args.only or active_repos(args.org, args.days, set(args.exclude))
    clones = args.clones_dir or Path(tempfile.mkdtemp(prefix="seed-clones-"))
    clones.mkdir(parents=True, exist_ok=True)
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(repos)} repos -> {', '.join(repos)}")

    for name in repos:
        path = sync_clone(args.org, name, clones)
        if args.apply:
            print(f"{name}: {refresh_project(mcp_server, name, path)}")
        else:
            with patch.object(mcp_server.vector_db, "add_documents") as add:
                mcp_server.seed_project_files(str(path), name)
            chunks = sum(len(c.args[1]) for c in add.call_args_list)
            print(f"{name}: would index {chunks} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
