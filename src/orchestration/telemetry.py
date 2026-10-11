import asyncio
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 50 * 1024 * 1024


class Telemetry:
    def __init__(self, log_file: str = "telemetry.jsonl"):
        self.log_file = log_file
        self._rotate_if_needed()

    def _rotate_if_needed(self):
        try:
            if os.path.getsize(self.log_file) > _MAX_FILE_BYTES:
                ts = int(time.time())
                rotated = f"{self.log_file}.{ts}.archive"
                os.rename(self.log_file, rotated)
                logger.info("Rotated telemetry file to %s", rotated)
        except OSError:
            pass

    async def log_call(self, agent_name: str, duration: float, success: bool, metadata: dict[str, Any] | None = None):
        entry = {
            "timestamp": time.time(),
            "agent": agent_name,
            "duration_sec": duration,
            "success": success,
            "metadata": metadata or {},
        }
        line = json.dumps(entry) + "\n"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_line, line)

    def _write_line(self, line: str):
        self._rotate_if_needed()
        with open(self.log_file, "a") as f:
            f.write(line)

    async def track(self, agent_name: str, func, *args, **kwargs):
        start = time.time()
        success = False
        try:
            result = await func(*args, **kwargs)
            success = True
            return result
        except Exception:
            raise
        finally:
            duration = time.time() - start
            try:
                await self.log_call(agent_name, duration, success)
            except Exception as e:
                logger.warning("Telemetry logging failed for %s: %s", agent_name, e)


telemetry = Telemetry()
