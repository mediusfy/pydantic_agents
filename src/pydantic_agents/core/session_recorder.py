"""Structured session recorder for pipeline agent runs.

Captures full prompt, context summary, tool calls, output, and decisions
per agent run — for fine-tuning data, debugging, and context-audit traces.

Output: JSONL with one record per agent invocation.
"""

import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

import logfire

logger = logging.getLogger(__name__)

# Context variable carrying current agent run info (run_id, agent_name, pipeline_stage)
# Set by record_span context manager, read by tool wrappers for cross-communication tracing.
current_run_context: ContextVar[dict[str, Any]] = ContextVar(
    "current_run_context",
    default={},  # noqa: B039 — safe: all writers allocate a new dict before .set()
)


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_args: dict[str, Any]
    result_preview: str  # first 500 chars
    duration_sec: float


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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Truncate prompt to keep files manageable
        d["prompt_truncated"] = len(d["prompt"]) > 100_000
        return d


class SessionRecorder:
    """Thread-safe session recorder that writes structured trace records."""

    def __init__(self, filepath: str = "session_trace.jsonl", max_bytes: int = 50 * 1024 * 1024):
        self.filepath = filepath
        self.max_bytes = max_bytes
        self._current: AgentRunRecord | None = None
        self._write_lock = False

    def start_run(
        self,
        run_id: str,
        agent_name: str,
        pipeline_stage: str,
        prompt: str,
        context_summary: str = "",
        story_id: str = "",
        story_gherkin: str = "",
    ) -> None:
        """Begin recording an agent invocation."""
        self._current = AgentRunRecord(
            run_id=run_id,
            timestamp=time.time(),
            agent_name=agent_name,
            pipeline_stage=pipeline_stage,
            prompt=prompt,
            context_summary=context_summary,
            story_id=story_id,
            story_gherkin=story_gherkin,
        )

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        duration_sec: float,
    ) -> None:
        """Record a tool call made during the current agent run."""
        if self._current is None:
            return
        result_str = str(result)[:500]
        self._current.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                tool_args=tool_args,
                result_preview=result_str,
                duration_sec=duration_sec,
            )
        )

    def finish_run(
        self,
        success: bool,
        output: Any = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        """Finalize the current agent run and write to JSONL."""
        if self._current is None:
            return None

        record = self._current
        record.success = success
        record.duration_sec = time.time() - record.timestamp
        record.error = error
        if token_usage:
            record.token_usage = token_usage

        if output is not None:
            record.output = str(output)[:50_000]
            if hasattr(output, "model_dump"):
                with contextlib.suppress(Exception):
                    record.output_json = output.model_dump()
            elif isinstance(output, dict):
                record.output_json = output

        data = record.to_dict()
        self._write(data)
        self._current = None
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
    ) -> AsyncIterator["SessionRecorder"]:
        """Async context manager wrapping a full agent run span.

        Usage:
            async with recorder.record_span(...) as rec:
                result = await run_agent(...)
                rec.set_output(result)

        Automatically calls finish_run on success or abort on exception
        — no dangling spans, no manual try/finally needed.
        """
        # Also open a logfire span for this agent run so the JSONL trace
        # correlates with Logfire's pydantic-ai spans
        with logfire.span(
            f"pipeline.{pipeline_stage}",
            agent_name=agent_name,
            story_id=story_id or "",
            prompt_length=len(prompt),
            **logfire_attrs,
            _level="info",
        ):
            self.start_run(
                run_id=run_id,
                agent_name=agent_name,
                pipeline_stage=pipeline_stage,
                prompt=prompt,
                context_summary=context_summary,
                story_id=story_id,
                story_gherkin=story_gherkin,
            )
            # Set contextvar so tool wrappers can read run_id, agent_name, pipeline_stage
            token = current_run_context.set(
                {
                    "run_id": run_id,
                    "agent_name": agent_name,
                    "pipeline_stage": pipeline_stage,
                    "story_id": story_id,
                }
            )
            try:
                yield self
            except Exception as e:
                self.finish_run(success=False, error=str(e))
                raise
            else:
                self.finish_run(success=True)
            finally:
                current_run_context.reset(token)

    def set_output(self, output: Any, success: bool = True) -> None:
        """Set the output for the current span (call inside record_span block).

        This replaces the default success=True with the actual output content.
        Must be called before the record_span block exits.
        """
        if self._current is None:
            return
        self._current.output = str(output)[:50_000]
        if hasattr(output, "model_dump"):
            with contextlib.suppress(Exception):
                self._current.output_json = output.model_dump()
        elif isinstance(output, dict):
            self._current.output_json = output
        self._current.success = success

    def abort_run(self, error: str) -> None:
        """Abort the current run without output."""
        if self._current:
            self.finish_run(success=False, error=error)

    def _write(self, data: dict[str, Any]) -> None:
        try:
            line = json.dumps(data, default=str) + "\n"
            self._rotate_if_needed()
            with open(self.filepath, "a") as f:
                f.write(line)
        except Exception as e:
            logger.warning(f"SessionRecorder write failed: {e}")

    def _rotate_if_needed(self) -> None:
        try:
            if os.path.getsize(self.filepath) > self.max_bytes:
                ts = int(time.time())
                rotated = f"{self.filepath}.{ts}.archive"
                os.rename(self.filepath, rotated)
                logger.info("Rotated session trace to %s", rotated)
        except OSError:
            pass


# Global singleton
recorder = SessionRecorder()


def extract_traces(filepath: str = "session_trace.jsonl", last_n: int = 0) -> list[dict]:
    """Extract session trace records as a list of dicts."""
    records = []
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return []

    if last_n > 0:
        records = records[-last_n:]
    return records


def summarize_traces(filepath: str = "session_trace.jsonl") -> str:
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
