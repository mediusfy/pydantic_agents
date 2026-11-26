"""Structured session recorder for pipeline agent runs.

Captures prompt, context summary, tool calls, output, and decisions per agent run — for debugging and context-audit
traces. Disabled by default (`SESSION_TRACE_ENABLED=false`) on the global singleton so raw prompts and generated source
code are not persisted on production pods unless explicitly enabled; when active, all persisted string/JSON fields are
scrubbed through `redact_secrets`.

Output: JSONL with one record per agent invocation. Each ``record_span`` owns its own record, so concurrent
spans (the advisory stage runs the CTO and UX agents side by side, the implementation stage every story's agents)
never overwrite each other.
"""

import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

import logfire

from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.settings import settings
from pydantic_agents.core.utils import rotate_log

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Secret scrubbing patterns covering API keys, cloud tokens, bearer/basic headers, PEM private keys, and DSNs.
_SECRET_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pat)
    for pat in (
        r"AKIA[0-9A-Z]{16}",
        r"ASIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z\-_]{35}",
        r"(?:ghp|gho|ghu|ghs|ghr)_[0-9a-zA-Z]{36}",
        r"github_pat_[0-9a-zA-Z_]{22,}",
        r"-----BEGIN\s?(?:RSA|DSA|EC|OPENSSH|PGP)?\s?PRIVATE KEY-----"
        r"[\s\S]*?-----END\s?(?:RSA|DSA|EC|OPENSSH|PGP)?\s?PRIVATE KEY-----",
        r"-----BEGIN\s?(?:RSA|DSA|EC|OPENSSH|PGP)?\s?PRIVATE KEY-----",
        r"\bsk-(?:proj-|ant-|live-|test-)?[0-9a-zA-Z_\-]{16,}",
        r"\btskey-[0-9a-zA-Z\-]{10,}",
        r"(?i)bearer\s+[0-9a-zA-Z\-_\.]{16,}",
        r"(?i)basic\s+[0-9a-zA-Z+/=]{16,}",
        r"eyJ[0-9a-zA-Z\-_]+\.eyJ[0-9a-zA-Z\-_]+\.[0-9a-zA-Z\-_]+",
        r"npm_[a-z0-9A-Z]{36}",
        r"xox[baprs]-[0-9a-zA-Z\-]{10,}",
        r"(?:sk|rk)_(?:test|live)_[0-9a-zA-Z]{24}",
        (
            r"(?i)(?:api[_-]?key|apikey|secret[_-]?key|app[_-]?secret|client[_-]?secret"
            r"|access[_-]?token|auth[_-]?token|password|passwd|pwd)"
            r"\s*[:=]\s*['\"]?[0-9a-zA-Z_\-\./+=]{8,}['\"]?"
        ),
    )
)
_URL_CRED_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^:/\s@]+:[^@\s]+@")


def redact_secrets(value: Any) -> Any:
    """Recursively scrub credentials, tokens, PEM blocks, and embedded DSN passwords from strings/dicts/lists."""
    if isinstance(value, str):
        scrubbed = _URL_CRED_RE.sub(r"\1[REDACTED]@", value)
        for regex in _SECRET_REGEXES:
            scrubbed = regex.sub("[REDACTED]", scrubbed)
        return scrubbed
    if isinstance(value, dict):
        return {k: redact_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value


# Context variable carrying current agent run info (run_id, agent_name, pipeline_stage)
# Set by record_span context manager, read by tool wrappers for cross-communication tracing.
# The default is an immutable empty mapping so no caller can mutate state shared across contexts; writers always
# allocate a new dict before .set().
current_run_context: ContextVar[Mapping[str, Any]] = ContextVar("current_run_context", default=MappingProxyType({}))

# The record of the innermost open span in this task, for tool wrappers that record tool calls.
_current_record: ContextVar["AgentRunRecord | None"] = ContextVar("session_recorder_record", default=None)


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_args: dict[str, Any]
    result_preview: str  # first 500 chars
    duration_sec: float


DEFAULT_TRACE_FILE = "session_trace.jsonl"


@dataclass
class AgentRunRecord:
    run_id: str
    timestamp: float
    agent_name: str
    pipeline_stage: str  # ba | advisory | implementation
    prompt: str
    context_summary: str  # any additional context passed via deps or metadata
    story_id: str | None  # CLUSTER-01, etc.
    story_gherkin: str  # full gherkin if applicable
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    output: str = ""
    output_json: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    duration_sec: float = 0.0
    error: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    written: bool = field(default=False, repr=False)  # a record is written once: the first finish wins

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("written", None)
        # Truncate prompt first so redaction runs over bounded length
        d["prompt_truncated"] = len(d["prompt"]) > 100_000
        if d["prompt_truncated"]:
            d["prompt"] = d["prompt"][:100_000]
        for key in ("prompt", "context_summary", "story_gherkin", "output", "output_json", "error", "tool_calls"):
            if d.get(key) is not None:
                d[key] = redact_secrets(d[key])
        return d

    def set_output(self, output: Any, success: bool = True) -> None:
        self.output = str(output)[:50_000]
        if hasattr(output, "model_dump"):
            with contextlib.suppress(Exception):
                self.output_json = output.model_dump()
        elif isinstance(output, dict):
            self.output_json = output
        self.success = success


class RecordingSpan:
    """What ``record_span`` yields: the span's own record plus the recorder that writes it."""

    def __init__(self, recorder: "SessionRecorder", record: AgentRunRecord):
        self._recorder = recorder
        self.record = record

    def set_output(self, output: Any, success: bool = True) -> None:
        """Set the output for this span (call inside the record_span block)."""
        self.record.set_output(output, success)

    def record_tool_call(self, tool_name: str, tool_args: dict[str, Any], result: Any, duration_sec: float) -> None:
        self._recorder._append_tool_call(self.record, tool_name, tool_args, result, duration_sec)

    def abort_run(self, error: str) -> None:
        """Abort this span's run without output."""
        self._recorder._finish(self.record, success=False, error=error)


class SessionRecorder:
    """Session recorder that writes one redacted trace record per agent run when enabled."""

    def __init__(
        self,
        filepath: str = DEFAULT_TRACE_FILE,
        max_bytes: int = 50 * 1024 * 1024,
        *,
        enabled: bool | None = None,
    ):
        self.filepath = filepath
        self.max_bytes = max_bytes
        self._enabled_override = enabled
        self._explicit_filepath = filepath != DEFAULT_TRACE_FILE

    def is_enabled(self) -> bool:
        """True when JSONL disk persistence is active for this recorder instance."""
        if self._enabled_override is not None:
            return self._enabled_override
        if self._explicit_filepath:
            return True
        env_raw = os.environ.get("SESSION_TRACE_ENABLED")
        if env_raw is not None:
            return env_raw.strip().lower() in _TRUTHY
        return bool(settings.session_trace_enabled)

    @staticmethod
    def _new_record(
        run_id: str,
        agent_name: str,
        pipeline_stage: str,
        prompt: str,
        context_summary: str = "",
        story_id: str = "",
        story_gherkin: str = "",
    ) -> AgentRunRecord:
        return AgentRunRecord(
            run_id=run_id,
            timestamp=time.time(),
            agent_name=agent_name,
            pipeline_stage=pipeline_stage,
            prompt=prompt,
            context_summary=context_summary,
            story_id=story_id,
            story_gherkin=story_gherkin,
        )

    @staticmethod
    def _append_tool_call(
        record: AgentRunRecord, tool_name: str, tool_args: dict[str, Any], result: Any, duration_sec: float
    ) -> None:
        record.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                tool_args=redact_secrets(dict(tool_args)),
                result_preview=redact_secrets(str(result)[:500]),
                duration_sec=duration_sec,
            )
        )

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        duration_sec: float,
    ) -> None:
        """Record a tool call made during the innermost open span of this task; a no-op outside a span."""
        record = _current_record.get()
        if record is None:
            return
        self._append_tool_call(record, tool_name, tool_args, result, duration_sec)

    def _finish(
        self,
        record: AgentRunRecord,
        success: bool,
        output: Any = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Finalize a record and write it to JSONL when enabled. Only the first call for a record writes."""
        if record.written:
            return record.to_dict()
        record.written = True
        record.success = success
        record.duration_sec = time.time() - record.timestamp
        record.error = error
        if token_usage:
            record.token_usage = token_usage
        if output is not None:
            record.set_output(output, success)
        data = record.to_dict()
        if self.is_enabled():
            self._write(data)
        return data

    @contextlib.asynccontextmanager
    async def record_span(
        self,
        run_id: str,
        agent_name: str,
        pipeline_stage: str,
        prompt: str,
        context_summary: str = "",
        story_id: str = "",
        story_gherkin: str = "",
        **logfire_attrs,
    ) -> AsyncIterator[RecordingSpan]:
        """Async context manager wrapping a full agent run span."""
        record = self._new_record(
            run_id=run_id,
            agent_name=agent_name,
            pipeline_stage=pipeline_stage,
            prompt=prompt,
            context_summary=context_summary,
            story_id=story_id,
            story_gherkin=story_gherkin,
        )
        span = RecordingSpan(self, record)
        span_attrs: dict[str, Any] = {
            "agent_name": agent_name,
            "story_id": story_id or "",
            "prompt_length": len(prompt),
            "langfuse.session.id": run_id_var.get() or run_id,
            **logfire_attrs,
        }
        with logfire.span(f"pipeline.{pipeline_stage}", _level="info", **span_attrs):
            token = current_run_context.set(
                {
                    "run_id": run_id,
                    "agent_name": agent_name,
                    "pipeline_stage": pipeline_stage,
                    "story_id": story_id,
                }
            )
            record_token = _current_record.set(record)
            finished = False
            try:
                yield span
            except BaseException as e:  # incl. CancelledError: a cancelled run is not a success
                finished = True
                self._finish(record, success=False, error=str(e) or type(e).__name__)
                raise
            finally:
                if not finished:
                    self._finish(record, success=record.success or not record.error, output=None)
                _current_record.reset(record_token)
                current_run_context.reset(token)

    def _write(self, data: dict[str, Any]) -> None:
        try:
            line = json.dumps(data, default=str) + "\n"
            self._rotate_if_needed()
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.warning(f"SessionRecorder write failed: {e}")

    def _rotate_if_needed(self) -> None:
        if rotate_log(self.filepath, self.max_bytes):
            logger.info("Rotated session trace %s", self.filepath)


# Global singleton (opt-in via SESSION_TRACE_ENABLED=true)
recorder = SessionRecorder()


def extract_traces(filepath: str = DEFAULT_TRACE_FILE, last_n: int = 0) -> list[dict]:
    """Extract session trace records as a list of dicts."""
    records = []
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return []

    if last_n > 0:
        records = records[-last_n:]
    return records


def summarize_traces(filepath: str = DEFAULT_TRACE_FILE) -> str:
    """Text summary of all trace records for real-time monitoring."""
    records = extract_traces(filepath, last_n=50)
    output = []
    for r in records:
        agent = r.get("agent_name", "?")
        stage = r.get("pipeline_stage", "?")
        sid = r.get("story_id") or "global"
        ok = "✓" if r.get("success") else "✗"
        duration = r.get("duration_sec", 0)
        tools = len(r.get("tool_calls", []))
        prompt_len = len(r.get("prompt", ""))
        error = r.get("error", "")
        line = (
            f"{ok} {agent:20s} [{sid:15s}] stage={stage:15s} {duration:5.0f}s tools={tools:2d} prompt={prompt_len:6d}ch"
        )
        if error:
            line += f" ERROR: {error[:80]}"
        output.append(line)
    return "\n".join(output)
