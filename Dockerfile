# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pydantic-agents

# Dependencies come from the hash-pinned lockfile (requirements.lock, linux/arm64, cp312), so the
# build is reproducible and tamper-evident. --only-binary :all: refuses sdists, so no third-party
# build scripts run. Regenerate with `make lock` after changing dependencies in pyproject.toml.
# Extras baked in: vector (ADR/architecture retrieval) and turso (STATE_BACKEND=turso, inert by default).
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes --only-binary :all: -r requirements.lock

# Install the project itself without resolving anything: deps are already locked, and hatchling
# (its build backend) is in the lock too, hence --no-build-isolation.
COPY pyproject.toml README* ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --no-build-isolation .

EXPOSE 8080

ENV LOGFIRE_SEND_TO_LOGFIRE=false

CMD ["uvicorn", "pydantic_agents.mcp_server:mcp.sse_app", "--host", "0.0.0.0", "--port", "8080"]
