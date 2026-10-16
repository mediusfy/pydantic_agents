# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
`project.version` in `pyproject.toml`.

Every pull request that changes behaviour, the image, CI, dependencies or
docs adds a bullet under **Unreleased** (CI enforces this; see `AGENTS.md`).
When a version is cut, rename **Unreleased** to the version and date and open a
fresh **Unreleased** section above it.

## [Unreleased]

### Added
- promptfoo agent evals: `promptfooconfig.yaml`, an in-process provider in
  `evals/provider.py`, and `make eval` (#18).
- README section on running the evals, including the gateway key requirements.
- CI image smoke test: builds the linux/arm64 image on pull requests, gates its
  size and checks torch/turso/Go/Node inside it (#19).
- Changelog, `AGENTS.md`, a pull request template and a CI check that every
  pull request updates this file.

### Changed
- `uv.lock` is now tracked (local `uv sync` / `make install` reproduce CI's resolution), and
  `make lock-check` fails when it is stale.
- Image dependencies install from a hash-pinned lockfile with CPU-only torch;
  the image shrank from 10.8 GB to 3.6 GB (#16).

## [0.2.0]

### Added
- Go and Node toolchains in the image (#15).
- `turso` extra installed in the image (AP-37, #14).
- Optional embedded Turso checkpoint backend (`STATE_BACKEND=turso`) and
  pipeline stage events recorded in Turso (AP-33, AP-34, AP-35; #11, #12).
- `scripts/seed_projects.py` to refresh the vector knowledge base from active
  repos, and a project filter for vector search and `query_context` (#7, #8).
- Makefile and docs on where the agents run (#10).

### Fixed
- Seed script upserts before pruning on the persistent loop and aborts on
  errors without deleting (#9).
