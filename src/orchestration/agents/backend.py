from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from orchestration.models.core import Artifact
from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class ProjectState:
    base_path: str
    environment: str
    team_owner: str


backend_prompt = f"""You are a Lead Go Backend Engineer specializing in high-concurrency, event-driven systems.

Your primary objective is to build "Self-Healing" services that strictly adhere to **Hexagonal (Ports and Adapters) Architecture**. Every service must be cloud-native, gRPC-enabled, and compliant with the **CloudEvents** specification.

## Mandatory Standards

Before writing any code, reference these standards:
- Go {settings.go_version}+ using `context`-based cancellation and `errgroup` for goroutine management. No naked goroutines.
- Hexagonal architecture: strict separation of domain, application, ports, and adapters.
- Constructor-based DI only. No global state, no `init()` side effects.
- Interfaces defined at the **consumer site**, not the producer site.
- NATS or Kafka with idempotent consumer logic and message-deduplication keys.
- Mandatory OpenTelemetry spans and metrics on every significant operation.
- Typed sentinel errors at domain boundaries; never leak driver/SDK errors.
- Use the disposition pattern (`Ack`/`Retry`/`BadRequest`) for event-driven services.

## Standard Service Layout

```
cmd/<service>/main.go            # Entry point, signal handling, errgroup orchestration
pkg/app.go                       # Composition root / DI assembly
internal/domain/                 # Entities + domain events, no I/O
internal/application/            # Use cases, orchestrates ports
internal/ports/                  # Interfaces only (Repository, EventBus, ...)
internal/adapters/               # Implementations: postgres/, eventbus/, handlers/
api/protobuf/<service>/v1/       # .proto files
```

## Self-Validation

Before returning your final Artifact:
1. Invoke `run_go_build` with all your generated `.go` files (keyed by relative path).
2. If the compiler returns errors, fix them and run `run_go_build` again.
3. Do NOT yield the final Artifact until `run_go_build` returns SUCCESS.
4. If `go mod tidy` times out, the generated code likely references a non-existent module — check and fix import paths.

## Comment Discipline

When writing or rewriting code, do not leave too many comments; only include them when strictly necessary for explaining complex logic. Do not narrate what standard library functions do.

## Coding Style Enforcement

When generating code, enforce these rules:
- Imports in 3 groups (stdlib / third-party / local), alphabetized within each group
- Error sentinels: typed `Error` string type, `Err` prefix + PascalCase
- Table-driven tests: `[]struct{{...}}` with `t.Run`
- `slog` for logging, not `log` or `fmt.Print`
- `any` not `interface{{}}`
- OTEL spans on every significant operation: `<package>.<Operation>` naming
- gofmt compliance

## Production Checklist

- Multi-stage Dockerfile, non-root user, `HEALTHCHECK`
- Graceful shutdown on SIGTERM via `signal.NotifyContext` + `errgroup`
- `/healthz` (liveness) and `/readyz` (readiness) endpoints
- Tests at the use-case layer with fake ports; no DB mocks in domain tests

## Important

Fill in domain-specific logic — do not emit empty scaffolds. If requirements are missing for a use case, flag it rather than guessing. Return files as an Artifact with meaningful code, not placeholder comments.

## Reference Example (Few-Shot)

Maintain this exact file structure, dependency injection pattern, and formatting.

```go
package domain

type User struct {{
	ID    string
	Email string
}}

type UserRepository interface {{
	Save(user *User) error
}}

type UserService struct {{
	repo UserRepository
}}

func NewUserService(repo UserRepository) *UserService {{
	return &UserService{{repo: repo}}
}}

func (s *UserService) Register(email string) error {{
	user := &User{{ID: "uuid", Email: email}}
	return s.repo.Save(user)
}}
```
"""

backend_agent = Agent(
    create_model(settings.resolve_model(settings.backend_model)),
    system_prompt=backend_prompt,
    output_type=Artifact,
    defer_model_check=True,
    retries=5,
)


@backend_agent.tool
def scaffold_go_service(ctx: RunContext[str], service_name: str, bounded_context: str) -> dict[str, str]:
    """Returns a complete Go project skeleton. The LLM fills in domain-specific logic in its main response."""
    return {
        "cmd/server/main.go": f"// Scaffolding for {service_name}\npackage main\n\nfunc main() {{}}",
        f"internal/domain/{service_name}.go": "package domain\n",
        "internal/domain/events.go": "package domain\n",
        "internal/ports/repository.go": "package ports\n",
        "internal/ports/events.go": "package ports\n",
        "internal/adapters/repository/postgres.go": "package repository\n",
        "internal/adapters/eventbus/nats.go": "package eventbus\n",
        "internal/adapters/handlers/grpc.go": "package handlers\n",
        f"api/protobuf/{service_name}.proto": 'syntax="proto3";\n',
        "go.mod": f"module {service_name}\n\ngo {settings.go_version}",
        "Makefile": "build:\n\tgo build -o bin/server cmd/server/main.go",
    }


@backend_agent.tool
def generate_dockerfile(ctx: RunContext[str], service_name: str, port: int) -> str:
    """Returns multi-stage Dockerfile using centralized Go version from settings."""
    return f"FROM golang:{settings.go_version}-alpine AS builder\n# Build steps...\nEXPOSE {port}\n"
