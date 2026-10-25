# Security

The agents act on text from MCP callers, web pages and a vector store, and some of their tools read files, write files and run processes. Treat changes to tools, the MCP server and model routing as security-relevant.

## Untrusted input

- MCP tool arguments, fetched web pages and search results (`tools/web_fetch.py`, and the DuckDuckGo search in `tools/travel_finance.py` and `tools/langchain_bridge.py`), retrieved vector-store snippets and generated artifacts are all model-visible text that can carry prompt injection. Nothing in this repo tags or neutralises it yet; do not add a path that treats such text as instructions to the pipeline itself.
- Retrieved context is prepended to every MCP agent request (`mcp_server.run_agent_by_name`), so whatever `seed_project_context`, `save_adr` or `scripts/seed_projects.py` writes into the store reaches every later call.

## Filesystem and processes

- Paths chosen by a model or a caller go through `safe_path_join` / `safe_write` (`core/utils.py`). `core/writer.py`, `tools/docs.py` and `tools/memory.py` do; `tools/fs_tools.py` (`read_file`, `write_file`, `list_dir`) takes raw paths and is attached to most SDLC agents, and the MCP tool `modify_existing_code` takes a caller-supplied path. `seed_project_context` is confined to `SEED_ROOTS` (default `OUTPUT_DIR`) and skips symlinks that leave the project; `scripts/seed_projects.py` calls the unrestricted `seed_project_files` on its own clones. Do not widen this; new file tools must be confined to a base directory.
- `safe_path_join` compares resolved paths with a string prefix check; keep that in mind when choosing base directories.
- Processes are started with argument lists (`asyncio.create_subprocess_exec`, `subprocess.run([...])`), never through a shell. Keep it that way and set a timeout.
- `fetch_url` accepts only `http://` and `https://` but follows redirects and does not restrict hosts. Do not give it credentials or headers from the environment.

## MCP server

- DNS rebinding protection is on unless `MCP_DNS_REBINDING_PROTECTION` is set to something other than `1`.
- Every agent run is capped at `UsageLimits(request_limit=500)`.

## Model routing and keys

- Keys come from settings or the environment: `LITELLM_VIRTUAL_KEY` / `LITELLM_MASTER_KEY` for the proxy, or direct provider keys. `.env` is gitignored; `.env.example` holds empty placeholders only.
- With `MODEL_PROVIDER=litellm` and no proxy key, `create_model` falls back to the raw model spec, so pydantic_ai may use whatever provider key is in the environment and bypass the proxy. Do not rely on that path, and prefer failing loudly when changing it.

## Workflows

- `docker.yml` holds the registry push credential and runs only on push to `main`. PR workflows (`ci.yml`, `changelog.yml`) use no repository secrets; keep secrets out of `pull_request` jobs.
- Existing workflows pin actions to major-version tags. New workflows should pin to a full commit SHA with the tag in a comment, as `ai-recent-context.yml` does.
- Pass `github.event` values to scripts through `env:`, not by interpolating them into `run:`.

## Agent context files

- `.ai/recent_context.md` is written by `.github/workflows/ai-recent-context.yml` from merged PR text, so it is a channel from PR authors to every coding agent. The workflow only takes merged PRs from this repo (never forks), reads the text through the API instead of `run:`, quotes each entry and caps its length. Keep those guards if you change it.

## Secrets

- Never commit secrets or print key values in logs, spans or tool output.
- `promptfooconfig.yaml` contains a deliberately fake `sk-live-...` key as eval input; secret scanners will flag it.
