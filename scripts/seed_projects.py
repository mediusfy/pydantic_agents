#!/usr/bin/env python3
"""Refresh the agents' vector knowledge base from recently active GitHub repos.

For each non-archived repo in ``--org`` pushed within ``--days`` (minus
``--exclude``), shallow-clone or update it under ``--clones-dir``, delete that
project's existing chunks, and re-seed it with ``seed_project_context`` so docs
removed upstream disappear and nothing is duplicated.

Dry run by default (prints what would be indexed). Pass ``--apply`` to write; that
needs DATABASE_URL pointing at the pgvector ``agents`` database, e.g. through
``kubectl -n pgvector port-forward pod/pgvector-0 15432:5432``.

    DATABASE_URL=postgresql://...@127.0.0.1:15432/agents \\
        python scripts/seed_projects.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

COLLECTIONS = ("adrs", "architecture", "conventions", "project")


def active_repos(org: str, days: int, exclude: set[str]) -> list[str]:
    out = subprocess.check_output(
        ["gh", "repo", "list", org, "--limit", "500", "--json", "name,pushedAt,isArchived"], text=True
    )
    cutoff = datetime.now(UTC) - timedelta(days=days)
    names = [
        r["name"]
        for r in json.loads(out)
        if not r["isArchived"]
        and r["name"] not in exclude
        and datetime.fromisoformat(r["pushedAt"].replace("Z", "+00:00")) >= cutoff
    ]
    return sorted(names)


def sync_clone(org: str, name: str, clones_dir: Path) -> Path:
    dest = clones_dir / name
    if (dest / ".git").is_dir():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only", "-q"], check=False)
    else:
        subprocess.run(["gh", "repo", "clone", f"{org}/{name}", str(dest), "--", "--depth", "1", "-q"], check=True)
    return dest


async def delete_project_rows(project: str) -> int:
    from pydantic_agents.core.vector_db import _get_pool

    pool = await _get_pool()
    deleted = 0
    async with pool.acquire() as conn:
        for c in COLLECTIONS:
            exists = await conn.fetchval("SELECT to_regclass($1)", f"v_{c}")
            if exists:
                res = await conn.execute(f"DELETE FROM v_{c} WHERE metadata->>'project' = $1", project)
                deleted += int(res.split()[-1])
    return deleted


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
            removed = asyncio.run(delete_project_rows(name))
            summary = mcp_server.seed_project_context(str(path), name).splitlines()[0]
            print(f"{name}: removed {removed} old chunks; {summary}")
        else:
            with patch.object(mcp_server.vector_db, "add_documents") as add:
                mcp_server.seed_project_context(str(path), name)
            chunks = sum(len(c.args[1]) for c in add.call_args_list)
            print(f"{name}: would index {chunks} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
