# Agent instructions

Applies to every coding agent and human contributor working in this repo.

## Changelog is mandatory

Every pull request must add at least one bullet to `CHANGELOG.md` under
`## [Unreleased]`, in the matching `### Added` / `### Changed` / `### Fixed` /
`### Removed` subsection (create the subsection if it is missing).

- Write for a reader of the release notes: what changed and why it matters, not
  which files you touched. Reference the PR or ticket, e.g. `(#123)`, `(AP-37)`.
- Applies to behaviour, agents, MCP tools, the image, CI, dependencies and docs.
- The `changelog` CI check fails the PR if `CHANGELOG.md` has no new bullet under
  Unreleased. For a change with genuinely nothing to record (typo, formatting),
  a maintainer can apply the `skip-changelog` label; do not add the label
  yourself to get past the check.

## Before pushing

`make lint && make typecheck && make test` must pass locally; see the
`Makefile` for other targets. Never commit secrets: credentials come from the
environment or a secret manager.
