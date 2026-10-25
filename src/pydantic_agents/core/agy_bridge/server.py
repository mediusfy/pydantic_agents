import asyncio
import json
import logging
import sys
from datetime import UTC, datetime

from pydantic_agents.core.agy_bridge.agent_sync import sync_agent_definitions
from pydantic_agents.core.agy_bridge.bridge_agent import BridgeAgent
from pydantic_agents.core.exceptions import AgentNotFoundError
from pydantic_agents.core.registry import registry

logger = logging.getLogger(__name__)


async def _handle_command(cmd: dict) -> dict:
    """Handle an incoming command dictionary and return a response dictionary."""
    command = cmd.get("command")

    if command == "ping":
        return {"response": "pong", "timestamp": datetime.now(UTC).isoformat()}

    elif command == "list-agents":
        agent_names = registry.list_agents()
        agents_meta = []
        for name in agent_names:
            agent = BridgeAgent.from_registry(name)
            agents_meta.append(
                {"name": name, "description": agent.get_system_prompt(), "tools": agent.get_tool_names()}
            )
        return {"response": {"count": len(agents_meta), "agents": agents_meta}}

    elif command == "get-agent":
        agent_name = cmd.get("agent_name")
        if not isinstance(agent_name, str):
            return {"error": "agent_name must be a string"}
        try:
            agent = BridgeAgent.from_registry(agent_name)
            return {
                "response": {
                    "name": agent_name,
                    "description": agent.get_system_prompt(),
                    "tools": agent.get_tool_names(),
                }
            }
        except AgentNotFoundError:
            return {"error": f"Agent '{agent_name}' not registered"}

    elif command == "sync-definitions":
        count = await sync_agent_definitions()
        return {"response": {"synced": count}}

    else:
        return {"error": f"unknown command: {command}"}


# ── Input Processing ─────────────────────────────────────────────────


async def _read_command_async() -> str | None:
    """Asynchronously read one command from stdin (line-based NDJSON or first-arg)."""
    # Check for a command JSON as the first argument
    if len(sys.argv) > 1:
        return sys.argv[1]

    # Support piped stdin asynchronously
    if not sys.stdin.isatty():
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if line:
            return line.strip()

    return None


# ── Main Loop ────────────────────────────────────────────────────────


def _emit(payload: dict) -> None:
    """Write one NDJSON response line to stdout."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


async def _respond_to(command: str) -> None:
    """Run one JSON command and print its response; invalid JSON is reported as an error line."""
    try:
        cmd_data = json.loads(command)
        response = await _handle_command(cmd_data)
        if "id" in cmd_data:
            response["id"] = cmd_data["id"]
        _emit(response)
    except json.JSONDecodeError as e:
        _emit({"error": f"invalid JSON: {e}"})


async def run_server_loop() -> None:
    """Run the agentapi protocol server in a loop, reading from stdin asynchronously."""
    from pydantic_agents.agents.registration import register_all_agents

    register_all_agents()
    logger.info("Agy bridge server started with %d agents", len(registry.list_agents()))

    sys.stderr.write(json.dumps({"event": "server_started", "agents": len(registry.list_agents())}) + "\n")

    # Read initial command asynchronously
    command = await _read_command_async()

    if command:
        # Single command mode (pipe or argument)
        await _respond_to(command)
        return

    # Interactive/daemon mode: continuous asynchronous read loop
    logger.info("Starting interactive command loop")
    sys.stderr.write(json.dumps({"event": "ready", "agents": len(registry.list_agents())}) + "\n")

    loop = asyncio.get_running_loop()

    while True:
        # Await readline in a separate thread so the event loop remains unblocked
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break  # EOF

        line = line.strip()
        if not line:
            continue

        try:
            # You can wrap this in asyncio.create_task() if you want commands
            # to process entirely concurrently without waiting for the previous to finish.
            await _respond_to(line)
        except Exception as e:
            logger.exception("Unhandled error processing command")
            _emit({"error": f"internal error: {e}"})


def main():
    """Console-script entrypoint. Calls asyncio.run(run_server_loop())."""
    asyncio.run(run_server_loop())


if __name__ == "__main__":
    main()
