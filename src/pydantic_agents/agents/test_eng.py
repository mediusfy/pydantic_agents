from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings
from pydantic_agents.core.utils import resolve_sandboxed

_SOURCE_MAX_CHARS = 100_000

# ── Playwright / BDD Test Agent (existing) ──────────────────────────────

test_prompt = """You are a QA Automation Agent specializing in **Behavior-Driven Development (BDD)**. You write Playwright tests to rigorously validate the Frontend application and the underlying Orgitration agentic flows based exactly on provided Gherkin criteria.

Testing Strategy & Execution:
- Asynchronous Validation: Ensure that events emitted by UI interactions are correctly acknowledged and side-effects executed by the Go backend.
- Agentic-User Simulation: Mimic highly complex, non-linear user journeys to aggressively stress-test frontend state management logic.
- Test Coverage: Cover every acceptance scenario and every event trigger outlined by the Business Analyst; state explicitly any behaviour you could not test.

## Output Protocol
Do NOT write files to disk. Return all generated code as structured output through this response. Your response will be captured as an Artifact and written to the filesystem by the pipeline orchestrator.
"""

test_eng_agent = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt=test_prompt,
    output_type=Artifact,
    defer_model_check=True,
    retries=settings.agent_retries,
)


class TestOutput(BaseModel):
    code: str


# Text-first agent: generates Playwright test code as free text (no output_type constraint)
test_gen = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt=test_prompt,
    defer_model_check=True,
    retries=settings.agent_retries,
)

# Private inner generator: tool-less (it is never registered), so it cannot delegate or write files.
_test_writer = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt=test_prompt,
    output_type=TestOutput,
    defer_model_check=True,
    retries=settings.agent_retries,
)


@test_eng_agent.tool
async def generate_test_suite(ctx: RunContext[Any], user_story_gherkin: str) -> str:
    """Converts Gherkin to Playwright TypeScript tests: data-testid selectors, async/await, page object pattern, backend event validation."""
    result = await _test_writer.run(
        f"Write a playwright test for {user_story_gherkin}.",
        usage=ctx.usage,
        model_settings={"max_tokens": 16384},
    )
    return result.output.code


# ── Go Table-Driven Test Agent (new for existing code) ──────────────────

go_test_prompt = """You are a Go Test Engineer specializing in table-driven unit tests. You write tests for EXISTING Go code, not new services.

## Mandatory Standards

- **Table-driven tests** as the default pattern:
  ```go
  tests := []struct {
      name       string
      req        ProcessRequest
      setupMocks func(s *MockService)
      expected   ExpectedResult
      wantErr    bool
  }{{ ... }}
  for _, tt := range tests {
      t.Run(tt.name, func(t *testing.T) { ... })
  }
  ```
- **testify** for assertions: assert.ErrorIs, assert.NoError, require.NoError.
- **mockery** for mock generation. Use mock.MatchedBy for argument matching.
- **Discarding logger**: slog.New(slog.NewTextHandler(io.Discard, nil))
- **t.Parallel()** where tests are independent.
- **Helper functions** for repetitive setup (newTestData(), newConfig()).
- **In-memory metrics reader** for metrics testing.
- **Test package naming**: foo_test for external tests, foo for internal tests.
- **Cover both success and error paths** in each table.

## Output

Return ONLY the test file content. The file should compile with the existing code. Do not change production code.
"""


class GoTestOutput(BaseModel):
    filepath: str
    code: str
    description: str


# Text-first agent: generates Go test code as free text (no output_type constraint)
go_test_gen = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt=go_test_prompt,
    defer_model_check=True,
    retries=settings.agent_retries,
)

# Private inner generator: tool-less (never registered), so it cannot delegate or write files.
_go_test_writer = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt=go_test_prompt,
    output_type=GoTestOutput,
    defer_model_check=True,
    retries=settings.agent_retries,
)


@test_eng_agent.tool
async def generate_go_table_driven_tests(ctx: RunContext[Any], source_file: str, function_name: str) -> str:
    """Generate table-driven Go tests for an existing function. Reads the source file, analyzes the function signature, and produces testify-based tests with mocks, t.Parallel(), and a discarding logger."""
    try:
        source = resolve_sandboxed(source_file).read_text(errors="replace")
    except (PathTraversalError, OSError) as e:
        return f"ERROR: cannot read {source_file}: {e}"
    if len(source) > _SOURCE_MAX_CHARS:
        # Testing a partial file would silently produce tests for code the model never saw.
        return f"ERROR: {source_file} is {len(source)} characters, over the {_SOURCE_MAX_CHARS} limit; split it first."
    result = await _go_test_writer.run(
        f"Generate table-driven tests for the function '{function_name}' in {source_file}.\n\n"
        f"Source of {source_file}:\n```go\n{source}\n```",
        usage=ctx.usage,
        model_settings={"max_tokens": 16384},
    )
    return f"{result.output.filepath}\n```go\n{result.output.code}\n```"
