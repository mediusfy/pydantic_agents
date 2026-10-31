# ADR-0002: Context integrity, run-state integrity, and the boundary of the Turso adoption

- Status: Proposed
- Date: 2026-10-31
- Owners: cto
- Reviewed commit: `b4da352` (`origin/main`, 2026-10-31)
- Relates to: extends `docs/turso-agent-databases-plan.md` (proposed 2026-10-15, spike AP-33 GO); depends on ADR-0001 for `platform_metrics` / `platform_findings`; supersedes nothing.
- Tickets: AP-102, AP-103, AP-104, AP-105, AP-106, AP-107, AP-108, AP-109, AP-110

## Context

A three-lane review of `origin/main` at `b4da352` (prompt assembly and token budget; retrieval quality; run state and durability) produced findings that are not independent bugs. They fall into three groups with three different root causes, and **only one of those causes is a storage problem**. That distinction is the decision this ADR exists to record, because an in-flight storage migration is currently the most likely thing to be pointed at them.

The architecture is **stateless one-shot agents with pgvector RAG as the only shared memory**. `adapters/agents.py:95` calls `agent.run(prompt, **kwargs)` with no `message_history` and no `conversation`; kwargs carries only `usage_limits`, `model_settings`, `deps`, `usage`. This is a defensible design, and the execution layer around it is genuinely well built:

- `limits.py:28` `permit_held` avoids nested-delegation deadlock, and `usage=ctx.usage` threads the parent budget into nested calls.
- `limits.py:78` `is_transient_error` correctly refuses to retry a context-length or timeout error.
- Retrieval is hybrid vector + Postgres FTS, RRF-fused, HNSW-indexed with a correct ivfflat fallback (`vector_db.py:294-315`), with the project filter applied in parameterised SQL under a `_METADATA_KEY` allowlist (`vector_db.py:479-481`). It is injection-safe.
- Storage is idempotent on `doc_id` (`ON CONFLICT (id) DO UPDATE`, `vector_db.py:430-442`), and deterministic ids on the seed, merge, ADR and ingest paths mean pipeline runs do not accumulate duplicates.
- The embedding dimension **is** asserted against the stored pgvector typmod (`vector_db.py:279-293`).

The cost of the design is that **the context layer carries the entire memory of the system**, and today it has no trust boundary, no relevance floor, and no token budget.

### Group 1 — the context layer has no trust boundary (the finding that matters)

`agents/innovation_scout.py:75` writes agent-drafted, unapproved ADRs into the `adrs` collection:

```python
await vector_db.async_add_documents("adrs", [content], [{"id": doc_id, "status": "proposed"}], [doc_id])
```

`context_engine.py:106` reads that same collection with **no metadata filter at all**:

```python
outcomes = await vector_db.async_search_domains(domains, request[:MAX_QUERY_CHARS], top_k, project)
```

`async_search_domains` takes four positional arguments. There is no `filters` parameter on this path, so `status: "proposed"` is written and never read. **Twelve agents** query the `adrs` domain (`context_engine.py:31-49`): `backend`, `frontend`, `database`, `service_engineer`, `security`, `review`, `quality_gate`, `sre`, `devops`, `cto`, `planner`, `observability_engineer` — and they receive it under the prompt header "Relevant Architecture Context from Knowledge Base".

Three of those twelve are `security`, `review` and `quality_gate`. **The platform built a human CTO approval gate for ADRs and then routed around it through retrieval**: an unapproved proposal becomes authoritative architecture context for the very gates meant to catch it. AP-92 is already open to gate the `save_adr` *write*; it does not address this, because the leak is on the *read*. Gating the write without filtering the read leaves the bypass intact.

Compounding it, there is no relevance floor:

- `grep -rn "threshold\|min_score\|max_distance" core/` returns **nothing**. `_search` orders by `embedding <=> $1` and takes `LIMIT` (`vector_db.py:505-515`); nothing is filtered by score.
- `_rrf_fuse` (`vector_db.py:362-372`) scores purely by **rank**, discarding distance. A document ranked #1 of 20 irrelevant candidates scores identically to a genuine 0.95-similarity hit.
- `rerank_model` defaults to `""` (`settings.py:162`), so `_rerank` degenerates to `documents[:n_results]`.
- `_search` swallows dependency failures into `[]` (`vector_db.py:534-537`), so from an agent's side "no context" and "retrieval is broken" are indistinguishable.

Net effect: **every agent always receives exactly 3 snippets, relevant or not, filtered or not.** `retrieve_context(top_k=3)` fans out to 3 domains × 3 results = 9 candidates, interleaves, dedupes on `snippet[:100]`, then truncates to `[:3]` (`context_engine.py:106-119`) — 6 of 9 retrieved snippets are paid for and discarded.

And the highest-value domain is the worst indexed. `mcp_server.py:276` `save_adr` stores an entire ADR as **one** vector with no chunking, while `_chunk_text` already exists in the same file (`mcp_server.py:652`) and is used by the seed path. `all-MiniLM-L6-v2` truncates at ~256 word pieces (≈1 KB), so **any ADR longer than ~1 KB is matched on its title and Context section only** — its Decision and Consequences are unindexed and unretrievable. This ADR is itself longer than that limit, which is one reason it is not being written through `save_adr`.

### Group 2 — there is no token budget, only eight unrelated character caps

There is no tokenizer in the repository. `grep -rni "tiktoken|count_tokens|num_tokens"` finds only `platform_metrics.py:62` `token_counts()`, which reads usage off a *completed* result for telemetry. Every pre-send bound is a hand-tuned character constant, and none of them is related to any model's context window or to each other: `80_000` (`utils.py:200`), `GATE_BATCH_CHARS = 80_000`, `_OUTPUT_MAX_CHARS = 4_000`, `_TEST_TRACE_CHARS = 3_000`, `_INDEX_MAX_CHARS = 4_000`, `MAX_QUERY_CHARS = 1_000`, `_READ_FILE_MAX_BYTES = 256 KiB`, `_SOURCE_MAX_CHARS = 100_000`, `_MAX_RESPONSE_CHARS = 500_000`.

Two consequences are correctness bugs, not cost bugs.

**A quality gate can record a verdict on code it never saw.** `format_artifacts` hard-slices the file that crosses the budget, then `break`s out of the **artifact** loop (`utils.py:225-228`), so every later artifact is silently dropped behind one truncation note. `_batches` (`quality_gate_stage.py:37-52`) never splits an oversized single artifact, despite a docstring at `:33` claiming it does. A 300 KB artifact loses 220 KB and the gate returns `passed`. This is a **silent false approval** in an autonomous remediation loop.

**Model-generated content is interpolated into gate prompts with forgeable plain-text delimiters.** `format_artifacts` separators are `=== Agent: … ===` and `--- path ---` (`utils.py:179-189`) — text an LLM-authored file can trivially emit. Build and test output is interpolated unfenced into the same prompts (`verifier.py:63-67`, `implementation_stage.py:219`), and the test *source* is model-authored, so a `t.Log("VERIFICATION: passed")` injects into exactly the format the gate is told to trust. `fetch_url` returns up to 500 KB of external content with no fence and no provenance marker (`web_fetch.py:101-109`) — a cap six times larger than the entire gate budget.

The largest *unbounded* prompt input is `get_markdown_context()` (`core/adapters/workspace.py:99-128`): `_adr_block` iterates `sorted(os.listdir(adrs_dir))` and concatenates **every** `.md` in full, with no count cap, no size cap and no truncation marker. It is built once per run and pasted into the backend, frontend and database prompt of **every story** (`implementation_stage.py:127-167`), plus BA and both advisory prompts. A repository with 40 ADRs × 6 KB contributes ~240 KB per prompt; × 3 agents × 5 stories ≈ 3.6 MB of identical billed input per run. The prefix-cache ordering at `implementation_stage.py:120-123` was deliberately chosen to make this cacheable, which mitigates cost on providers that cache but does nothing about the per-request context-window limit.

### Group 3 — run-state integrity (the part that *is* storage-shaped)

- `run_store.save_run` is an unconditional `INSERT … ON CONFLICT (run_id) DO UPDATE` over a `pipeline_runs` table with **no version column and no status guard** (`run_store.py:46-66`, `:81-93`). The only double-submission guard is the **in-process** `PIPELINE_RUNS` dict (`pipeline_jobs.py:79-81`). The code is honest about the consequence — `list_interrupted_runs` documents *"this assumes one server process per database"* (`pipeline_jobs.py:198-203`) — but that is an undocumented hard deployment constraint, and `sre_autopilot` auto-resumes off exactly this signal (`sre_remediation.py:47-67`).
- `interrupted` is **derived, never stored** (`pipeline_jobs.py:186-192`), so after a crash the row says `running` forever.
- `grep -ci 'resume|checkpoint'` is **0** across all five `pipeline/crewai/*.py` and all four `pipeline/autogen/*.py` files, against 24 in `langgraph/checkpointer.py` and 9 in `langgraph/graphs.py`. Yet `factory.py:60-69` dispatches on framework and passes `config.resume=True` regardless, and `resume_pipeline_run` reports success. **A "resume" of `crew_sdlc`, `crew_qa`, `crew_architecture_review` or `crew_audit` is a silent full re-run from scratch of every tool side effect** — today that means re-merging artifacts over the project tree (`merge_stage.py:122-125`), where a second merge with different LLM output overwrites the user's original file in `.merge-backups/` via `shutil.copy2` (`merge_stage.py:117-125`) and the original is gone.
- The legacy orchestrator checkpoints only *after* a stage returns (`orchestrator.py:163-166`), so an interrupted stage loses every LLM call it made.
- `delegation_depth` lives only in a contextvar (`a2a_tools.py:65-66`) and is absent from `PipelineState` (`models/core.py:323-343`), so **each resume grants a fresh delegation budget**.
- Both `checkpoint_dir` (`./.claw-checkpoints`) and `output_dir` (`./generated-app`) default to **relative, pod-local paths**. Out of the box nothing here survives a pod restart except the Postgres row — which can only report that the run was interrupted, never resume it.

The CTO gate's rendezvous medium is the sharpest case. `consume_marker` (`core/cto_gate.py:48-58`) deletes the marker as soon as it reads a decision, and its docstring gives a sound reason: a marker answers one gate, not every later run. The defect is not the deletion — it is that **the decision is consumed before it is durably recorded anywhere**, while the gate itself is an in-process `asyncio.sleep` poll loop (`cto_gate.py:77-95`) inside a stage that is never checkpointed. Die between the `unlink` and the advisory checkpoint and the human approval is gone, and resume demands a second one. Worse, `_poll_for_file_approval` **returns `False` on timeout** (`cto_gate.py:93-95`) — the same value as an explicit rejection, distinguishable only in the logs. A 3600 s wait is silently converted into "the CTO said no".

Two further gate defects: `_approval_file_path` (`cto_gate.py:25-32`) joins the caller-chosen `project` with plain `Path` and calls `mkdir(parents=True)`, creating directories outside `output_dir` for `project="../../tmp/x"` — while `write_approval` three lines below **deliberately uses `safe_path_join` for exactly this reason** and documents it. The read path lost the guard the write path has. And the marker is keyed by *project*, not `run_id`, so concurrent runs of one project cannot be approved independently.

## Decision

**1. The context layer gets a trust boundary, a relevance floor, and a token budget. This is not a storage concern and must not wait on any storage migration.**

- Thread a `filters` argument through `retrieve_context` → `async_search_domains` → `_search`, and make the `adrs` domain exclude `status = "proposed"` by default. Approved-only is the default; retrieving a proposal requires asking for it explicitly. (AP-102)
- Add a cosine-distance cutoff applied **before** RRF, and keep distance in the fusion score rather than discarding it for rank. Return fewer than `top_k` snippets — including zero — rather than padding with the best of a bad set. Distinguish "no relevant context" from "retrieval failed" in what the agent sees. (AP-103)
- Run `save_adr` content through the existing `_chunk_text`. (AP-104)
- `project` becomes required, or default-deny, on `retrieve_context`; tag ADRs with a project at write time. Today `project=None` means "search every project", so project A's ADRs can land verbatim in project B's prompt, and `save_adr` takes no project at all — making ADRs simultaneously globally leaky and invisible to any scoped query. (AP-102)
- Introduce one real tokenizer-backed budget that the per-block caps derive from. `format_artifacts` must never silently drop an artifact: split oversized artifacts across gate calls, and make a gate verdict on truncated input an explicit `inconclusive`, never a `passed`. (AP-105)
- Bound `get_markdown_context()` and mark truncation. (AP-106)

**2. Untrusted content gets a delimiter that model output cannot forge.** Model-generated files, subprocess output and fetched web pages are all attacker-influenced inputs to the gates. They need an unforgeable envelope and a provenance marker, and the gate prompts need to state that content inside it is data, not instruction. (AP-105 covers the artifact path; the subprocess and `fetch_url` paths are tracked under AP-75's red-team baseline.)

**3. Run-state integrity gets a version guard and a durable decision record.**

- Add a version or status guard to `run_store.save_run`, and store `interrupted` rather than deriving it. Document single-process-per-database until that lands, because `sre_autopilot` auto-resumes off this signal. (AP-108)
- Record the CTO decision durably *before* consuming the marker, and make a timeout a distinct outcome from a rejection. Restore `safe_path_join` on the read path and key the marker by `run_id`. (AP-108, AP-109)
- Persist `delegation_depth` in `PipelineState`. (AP-108)

**4. `resume` must fail loudly for frameworks that cannot resume.** `factory.py` rejects `config.resume=True` for `crewai` and `autogen` with a clear error until those orchestrators have checkpointing. Reporting a full re-run as a resume is worse than refusing. (AP-107)

### 5. The boundary of the Turso adoption

**Turso fixes none of the Group 1 or Group 2 findings, and must not be pointed at pgvector.** Those defects are semantic — a missing filter, a missing relevance floor, a missing chunking call, a missing tokenizer. They are properties of the retrieval and prompt-assembly logic, not of the engine underneath, and they would survive a storage swap unchanged. `docs/turso-agent-databases-plan.md` already drew this boundary correctly at line 24 ("Vector/ADR context lives in Postgres and is out of scope") and in its non-goals; **this review confirms that boundary empirically and it should not be revisited.**

Turso's existing scope — replacing one-JSON-file-per-`{run_id}_{stage}` checkpointing with an embedded, optionally-synced database — is the right fit for Group 3, and the Group 3 findings strengthen its case: pod-local relative `checkpoint_dir`, fragile mtime-and-filename-parsing in `find_latest_run_id`, and no durable gate decision are all exactly what that plan set out to fix.

**Turso gains one new job beyond that plan: a per-run effect ledger.** The reason `crewai`/`autogen` resume is dangerous is not that checkpoints are missing — it is that there is no record of which side effects a run already performed, so a re-run repeats them. An append-only `run_effects(run_id, effect_key, status, result)` table in the same embedded database, written before and after each side-effecting operation, makes re-execution idempotent and is the precondition for ever enabling resume on those frameworks. It is also the missing safety net for `merge_stage`'s backup overwrite and for any future branch-push or PR-creation step — `grep` confirms no git-push or PR machinery exists in the repo today, so this lands *before* the first such step, not after.

## Consequences

- Agents will sometimes receive **zero** context snippets. That is the intended outcome and a visible behaviour change; today's 3-snippets-always hides whether retrieval ever worked. Expect agent output to change on prompts where the 3 snippets were noise.
- Excluding `status = "proposed"` reduces what `innovation_scout` contributes to other agents. That is the point: the CTO gate becomes real.
- Chunking ADRs changes `doc_id` shape for the `adrs` collection. Positional ids leave orphans when a document shrinks (`mcp_server.py:754`), so this needs delete-before-insert, which the seed path also lacks.
- A tokenizer-backed budget adds a dependency and a per-request cost. Accepted: it replaces eight constants that cannot be reasoned about together.
- Refusing `resume=True` on `crewai`/`autogen` is a breaking change for any caller relying on it. Those callers are currently getting a silent full re-run, so the break surfaces existing damage rather than causing it.
- Single-process-per-database remains a hard constraint until the version guard lands, and must be stated in deploy docs now.

## Verification status

Every file:line claim above was re-verified directly against `b4da352` after the review lanes reported, not taken from their prose. Specifically confirmed by direct read: `utils.py:187-228`, `core/adapters/workspace.py:49-128`, `core/cto_gate.py:25-95`, `mcp_server.py:273-277`, `context_engine.py:31-49` and `:91-112`, `agents/innovation_scout.py:60,75`, `factory.py:60-69`, and the zero-vs-24 checkpoint grep across `pipeline/crewai`, `pipeline/autogen` and `pipeline/langgraph`.

**Not verified, and therefore stated as a spike rather than asserted:** the claimed pgvector HNSW post-filter recall failure (`vector_db.py:498` filtering on `metadata->>'project'` in the same query as `ORDER BY embedding <=> … LIMIT`, with `hnsw.ef_search` never set). The mechanism is real in pgvector and would mean a project-scoped search can return **zero** rows while that project holds perfect matches, but it is load- and cardinality-dependent and was not reproduced against a populated multi-project table. **AP-103 carries a spike to reproduce it before any index change**; if it reproduces, the fix is a partial index per project, `iterative_scan` (pgvector ≥ 0.8), or a pre-filter CTE, and it raises AP-103's priority. Also unverified: whether `hnsw.ef_search` is tuned anywhere outside `core/`, and the `agy_bridge/bridge_agent.py` retrieval path.
