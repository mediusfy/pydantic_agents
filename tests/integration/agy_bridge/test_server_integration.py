import asyncio
import json
import sys

import pytest

# Cold start of the bridge server imports pydantic_ai, langchain and all 27
# registered agents.  That takes ~4s on a warm laptop and well over 5s on a
# 2-vCPU GitHub runner with a cold bytecode cache, so the budget is generous.
SERVER_STARTUP_TIMEOUT_S = 60


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_ping_integration():
    """
    Tests that the agy_bridge server can be started as a subprocess
    and responds to a 'ping' command.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pydantic_agents.core.agy_bridge.server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    command = json.dumps({"command": "ping"}) + "\n"

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=command.encode()), timeout=SERVER_STARTUP_TIMEOUT_S
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        pytest.fail(f"Server process timed out after {SERVER_STARTUP_TIMEOUT_S}s.")

    assert proc.returncode == 0, f"server exited with {proc.returncode}; stderr:\n{stderr.decode(errors='replace')}"

    response = json.loads(stdout)
    assert "response" in response
    assert response["response"] == "pong"
