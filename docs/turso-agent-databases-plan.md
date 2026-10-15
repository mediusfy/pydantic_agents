# Plan: Turso agent databases for pipeline state

Status: proposed · Date: 2026-10-15 · Jira: epic in project AP, label `turso-agent-db`
Reference: https://docs.turso.tech/guides/agent-databases

## Goal

Persist coding-pipeline (and PR/existing-code pipeline) run state in an embedded
Turso database instead of one JSON file per `{run_id}_{stage}`, with optional
Turso Sync so a run can resume after a pod restart.

## Background

- `core/checkpoint.py::CheckpointManager` delegates to `StateRepositoryPort`
  (`core/adapters/state_repository.py`): `save`, `load`, `exists`, `clear`,
  `find_latest_run_id`.
- The port's docstring already anticipates DB-backed persistence. The only
  implementation is `LocalDiskStateRepository` (JSON files in
  `settings.checkpoint_dir/<project>`).
- `find_latest_run_id` sorts files by mtime and parses the UUID out of the
  filename; `clear` matches by filename prefix. Both are fragile.
- `settings.checkpoint_dir` is likely ephemeral in-cluster, so resume does not
  survive a pod restart.
- Vector/ADR context lives in Postgres (`vector_db.py`) and is out of scope.

## What Turso gives us (from the guide)

- Embedded mode: a single local SQLite-compatible file, no network.
- Turso Sync: local file replicated to Turso Cloud (`sync` / `pull` / `push`).
- Database-per-agent via the Platform API, and hub-and-spoke aggregation.
- Caveat: the guide's examples are JS only and it does not document a Python
  SDK or branching. Python support must be verified first (spike).

## Non-goals

- Replacing the Postgres vector DB.
- Database-per-agent cloud provisioning. One embedded file per project/run is
  enough isolation for in-process LangGraph nodes.
- Hub-and-spoke across pods (stages run in one process today).

## Design

Add `TursoStateRepository(StateRepositoryPort)`; no pipeline code changes.

```sql
CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
  run_id      TEXT NOT NULL,
  stage       TEXT NOT NULL,
  state_json  TEXT NOT NULL,
  updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON pipeline_checkpoints(updated_at DESC);
```

- `save` = upsert; `load`/`exists` = keyed lookup; `clear` = `DELETE WHERE run_id=?`;
  `find_latest_run_id` = `ORDER BY updated_at DESC LIMIT 1`.
- Corrupt rows are logged and return `None`, matching the disk repository.
- Backend selected by a setting (`state_backend = disk | turso`, default `disk`).
- Optional second table `pipeline_stage_events` (run_id, stage, attempt, verdict,
  duration_ms, created_at) as a queryable audit log of remediation retries and
  quality-gate outcomes (`MAX_RETRIES` in `existing_code_handler.py`).
- Phase 3 only: enable Turso Sync (`sync="full"`, token via env/SOPS, never
  committed) if checkpoints must outlive the pod.

## Phases

1. **Spike**: confirm a usable Python package/SDK (e.g. `pyturso`), async story,
   sync support, license, arm64 wheels (image is linux/arm64).
2. **Embedded repository**: `TursoStateRepository`, settings flag, table-driven
   tests parity-checked against `LocalDiskStateRepository`.
3. **Audit log**: `pipeline_stage_events` written from stage runner and
   remediation loop in `existing_code_handler.py`.
4. **Sync + deployment**: Turso Sync with credentials via SOPS-managed secret in
   the `pydantic-agents` Helm release (`k8s/charts/agents`), PVC or sync for the
   db file, resume-after-restart verified.
5. **Rollout**: enable in the coding pipeline, then the existing-code/PR flow;
   keep `disk` as fallback for one release.

## Risks

- Python SDK maturity (Turso DB is still young) → spike gates the epic.
- Sync credentials: must go through SOPS; no plaintext tokens.
- Dual-backend drift → shared contract test suite run against both repositories.
- Concurrency: one writer per run file; confirm locking behavior under async.

## Acceptance for the epic

- Pipeline resumes from the latest checkpoint using the Turso backend.
- Resume works after pod restart (with Sync or PVC).
- Contract tests pass for both backends; `make test` clean.
- Stage events queryable for a run.

## Spike result (AP-33) — 2026-10-15: GO for embedded

- `pyturso` 0.8.3 (import name `turso`) is a synchronous DB-API-style package with
  wheels for linux aarch64, macOS arm64 and x86_64.
- Verified on the embedded path: `connect`, `execute` with `?` params,
  `INSERT ... ON CONFLICT DO UPDATE`, `rowcount`, `commit`, reopen persistence.
- It is sync-only, so the repository serialises access with a lock (the pipeline
  calls the port synchronously anyway).
- Turso Sync from Python was **not** evaluated; that belongs to the deploy ticket (AP-36).
- Shipped as the optional extra `pydantic-agents[turso]`; select with `STATE_BACKEND=turso`
  (default `disk`).

## Stage events (AP-35)

`pipeline_stage_events(id, run_id, stage, attempt, verdict, duration_ms, detail, created_at)` lives in the
same per-project `state.db`. `run_existing_code_pipeline` records service_engineer, test_engineer,
quality_gate, review and a final `pipeline` verdict per attempt via `StageEventRecorderPort`. The recorder is a
no-op unless `STATE_BACKEND=turso`, and recording failures are swallowed so they cannot break a run.
`ExistingCodeResult.run_id` identifies the run; query with `TursoStageEventRecorder(project).list_events(run_id)`.
The project name is the basename of the codebase path.
