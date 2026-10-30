"""Persistent disk-backed LangGraph checkpointer.

Subclasses LangGraph's ``InMemorySaver`` (``MemorySaver``) and mirrors each thread's serialized
``storage``, ``writes`` and ``blobs`` entries to ``<checkpoint_dir>/<project>/<thread_id>__langgraph.ckpt``
as JSON + base64 (never ``pickle``), using atomic temp-file replacement. That lets ``resume=True``
continue a LangGraph pipeline across process restarts when ``CHECKPOINT_DIR`` is on persistent storage,
without requiring an external Postgres checkpointer service.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import MemorySaver

from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

_FILE_SUFFIX = "__langgraph.ckpt"


def _enc_typed(pair: tuple[str, bytes]) -> list[str]:
    return [pair[0], base64.b64encode(pair[1]).decode("ascii")]


def _dec_typed(raw: Sequence[Any]) -> tuple[str, bytes]:
    return (str(raw[0]), base64.b64decode(str(raw[1])))


class DiskCheckpointSaver(MemorySaver):
    """``MemorySaver`` backed by atomic per-thread files under ``settings.checkpoint_dir/<project>``."""

    def __init__(self, project_name: str = "default", base_dir: str | Path | None = None) -> None:
        super().__init__()
        self.project_name = project_name
        self._base_dir = Path(base_dir) if base_dir is not None else None
        self._load_all()

    @property
    def checkpoint_dir(self) -> str:
        root = self._base_dir if self._base_dir is not None else settings.checkpoint_dir
        path = os.path.join(str(root), self.project_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _filepath(self, thread_id: str) -> str:
        safe = thread_id.replace("/", "_").replace(" ", "_").replace("..", "_")
        return os.path.join(self.checkpoint_dir, f"{safe}{_FILE_SUFFIX}")

    def _serialize_thread(self, thread_id: str) -> dict[str, Any]:
        storage_out: dict[str, dict[str, Any]] = {}
        for ns, ns_map in self.storage.get(thread_id, {}).items():
            storage_out[ns] = {
                cid: [_enc_typed(ckpt), _enc_typed(meta), parent_id] for cid, (ckpt, meta, parent_id) in ns_map.items()
            }
        writes_out = [
            {
                "ns": ns,
                "cid": cid,
                "items": [
                    [task_id, idx, w_task_id, channel, _enc_typed(val), task_path]
                    for (task_id, idx), (w_task_id, channel, val, task_path) in inner.items()
                ],
            }
            for (tid, ns, cid), inner in self.writes.items()
            if tid == thread_id
        ]
        blobs_out = [
            [ns, channel, version, _enc_typed(val)]
            for (tid, ns, channel, version), val in self.blobs.items()
            if tid == thread_id
        ]
        return {"thread_id": thread_id, "storage": storage_out, "writes": writes_out, "blobs": blobs_out}

    def _persist_thread(self, thread_id: str) -> None:
        filepath = self._filepath(thread_id)
        tmp_path = f"{filepath}.tmp"
        payload = json.dumps(self._serialize_thread(thread_id))
        with open(tmp_path, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)

    def _restore_payload(self, data: dict[str, Any]) -> None:
        thread_id = str(data.get("thread_id") or "")
        if not thread_id:
            return
        for ns, ns_map in (data.get("storage") or {}).items():
            for cid, triple in ns_map.items():
                self.storage[thread_id][ns][cid] = (
                    _dec_typed(triple[0]),
                    _dec_typed(triple[1]),
                    triple[2],
                )
        for entry in data.get("writes") or []:
            outer_key = (thread_id, str(entry["ns"]), str(entry["cid"]))
            for item in entry.get("items") or []:
                task_id, idx, w_task_id, channel, val, task_path = item
                self.writes[outer_key][(str(task_id), int(idx))] = (
                    str(w_task_id),
                    str(channel),
                    _dec_typed(val),
                    str(task_path),
                )
        for ns, channel, version, val in data.get("blobs") or []:
            self.blobs[(thread_id, str(ns), str(channel), version)] = _dec_typed(val)

    def _load_all(self) -> None:
        d = self.checkpoint_dir
        if not os.path.isdir(d):
            return
        for fname in os.listdir(d):
            if not fname.endswith(_FILE_SUFFIX):
                continue
            fpath = os.path.join(d, fname)
            try:
                with open(fpath) as f:
                    self._restore_payload(json.load(f))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Corrupted LangGraph checkpoint %s: %s", fpath, exc)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config.get("configurable", {}).get("thread_id") or "")
        if thread_id and thread_id not in self.storage:
            fpath = self._filepath(thread_id)
            if os.path.exists(fpath):
                try:
                    with open(fpath) as f:
                        self._restore_payload(json.load(f))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.warning("Corrupted LangGraph checkpoint %s: %s", fpath, exc)
        return super().get_tuple(config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        saved = super().put(config, checkpoint, metadata, new_versions)
        thread_id = str(config.get("configurable", {}).get("thread_id") or "")
        if thread_id:
            self._persist_thread(thread_id)
        return saved

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)
        thread_id = str(config.get("configurable", {}).get("thread_id") or "")
        if thread_id:
            self._persist_thread(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        fpath = self._filepath(thread_id)
        if os.path.exists(fpath):
            with contextlib.suppress(OSError):
                os.remove(fpath)

    def find_latest_run_id(self) -> str | None:
        """Most recently modified thread id in ``checkpoint_dir``, or None."""
        d = self.checkpoint_dir
        if not os.path.isdir(d):
            return None
        files = [f for f in os.listdir(d) if f.endswith(_FILE_SUFFIX)]
        if not files:
            return None
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        return files[0][: -len(_FILE_SUFFIX)]
