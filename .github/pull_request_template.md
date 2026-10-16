## What and why

<!-- What changed and why. Link the ticket or issue. -->

## Checklist

<!-- Criteria: .ai/guidance/testing.md, "Merge and verification criteria". Tick only what you checked; for anything not applicable, say why. -->

- [ ] `CHANGELOG.md` has a new bullet under `## [Unreleased]` (required; CI checks it, see `AGENTS.md`)
- [ ] `make lint && make typecheck && make test` pass locally
- [ ] Docs (`README.md`, `docs/`) updated if behaviour or setup changed
- [ ] No secrets, tokens or credentials in the diff
- [ ] **CI status:** CI is green on the latest commit
- [ ] **Sonar analysis:** the SonarCloud quality gate passes and new findings are addressed
- [ ] **Tests:** `make test-integration` for pipeline or MCP changes; `make eval` for prompt, output model or routing changes (or not required); `make smoke` passes
- [ ] **Coverage and gaps:** coverage of the changed code was checked (`make cov`), with no missing branches or edge cases left unexplained

## AI Context & Rules Update

<!-- One to five lines for coding agents: a new convention, an architectural shift, a rule that changed or a gotcha found. Write "None" if nothing changed. When this PR merges, this section is copied into .ai/recent_context.md. -->

None
