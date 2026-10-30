# Security

The agents act on text from MCP callers, web pages and a vector store, and some of their tools read files, write files and run processes. Treat changes to tools, the MCP server and model routing as security-relevant.

## Untrusted input

- MCP tool arguments, fetched web pages and search results (`tools/web_fetch.py`, and the DuckDuckGo search in `tools/travel_finance.py` and `tools/langchain_bridge.py`), retrieved vector-store snippets and generated artifacts are all model-visible text that can carry prompt injection. Nothing in this repo tags or neutralises it yet; do not add a path that treats such text as instructions to the pipeline itself.
- Retrieved context is prepended to every MCP agent request (`mcp_server.run_agent_by_name`), so whatever `seed_project_context`, `save_adr` or `scripts/seed_projects.py` writes into the store reaches every later call.

## Filesystem and processes

- Paths chosen by a model or a caller go through `safe_path_join` / `safe_write` / `resolve_sandboxed` (`core/utils.py`). `core/writer.py`, `tools/fs_tools.py` (`read_file`, `write_file`, `list_dir`, split into `fs_read` and `fs_write` categories), `tools/docs.py` and `tools/memory.py` are all confined to `sandbox_root()`. `mcp_server.modify_existing_code` and `seed_project_context` are confined to `SEED_ROOTS` (default `OUTPUT_DIR`, plus `ORCHESTRATION_CODEBASE_ROOT` for `modify_existing_code`) and skip symlinks that leave the project; `scripts/seed_projects.py` calls `seed_project_files` on its own validated clones. Do not widen this; new file tools must be confined to a base directory.
- `safe_path_join` checks containment with `Path.is_relative_to` on resolved paths.
- Processes are started with argument lists (`asyncio.create_subprocess_exec`, `subprocess.run([...])`), never through a shell, with a timeout and `sanitized_subprocess_env()` for untrusted code/tests so credentials are not inherited.
- `fetch_url` (`tools/web_fetch.py`) and `ingest_url` validate URLs with `validate_public_http_url` (blocking loopback, RFC1918, link-local/cloud metadata, and internal hostnames on both the initial request and any redirect hop) and cap response size. Do not give them credentials or headers from the environment.

## MCP server

- DNS rebinding protection is on unless `MCP_DNS_REBINDING_PROTECTION` is set to something other than `1`, with allowed `Host`/`Origin` patterns configured via `MCP_ALLOWED_HOSTS` and `MCP_ALLOWED_ORIGINS`.
- Every agent run is capped by `usage_limits(agent_name)` (`request_limit=40`, `60` for `backend`, `frontend` and `service_engineer`, `total_tokens_limit=2_000_000`).

## Model routing and keys

- Keys come from settings or the environment: `LITELLM_VIRTUAL_KEY` / `LITELLM_MASTER_KEY` for the proxy, or direct provider keys. `.env` is gitignored; `.env.example` holds empty placeholders only.
- With `MODEL_PROVIDER=litellm` and no proxy key, `create_model` raises `ConfigurationError` on first model use instead of falling back to direct provider calls.

## Workflows

- `docker.yml` holds the registry push credential and runs only on push to `main`. PR workflows (`ci.yml`, `changelog.yml`) use no repository secrets; keep secrets out of `pull_request` jobs.
- Existing workflows pin actions to major-version tags. New workflows should pin to a full commit SHA with the tag in a comment, as `ai-recent-context.yml` does.
- Pass `github.event` values to scripts through `env:`, not by interpolating them into `run:`.

## Agent context files

- `.ai/recent_context.md` is written by `.github/workflows/ai-recent-context.yml` from merged PR text, so it is a channel from PR authors to every coding agent. The workflow only takes merged PRs from this repo (never forks), reads the text through the API instead of `run:`, quotes each entry and caps its length. Keep those guards if you change it.

## Secrets

- Never commit secrets or print key values in logs, spans or tool output.
- `promptfooconfig.yaml` contains a deliberately fake `sk-live-...` key as eval input; secret scanners will flag it.
