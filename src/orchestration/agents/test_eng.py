from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, UsageLimits

from orchestration.models.core import Artifact
from orchestration.models.factory import create_model
from orchestration.settings import settings

# ── Playwright / BDD Test Agent (existing) ──────────────────────────────

test_prompt = """You are a QA Automation Agent specializing in **Behavior-Driven Development (BDD)**. You write Playwright tests to rigorously validate the Frontend application and the underlying Orgitration agentic flows based exactly on provided Gherkin criteria.

Testing Strategy & Execution:
- Asynchronous Validation: Ensure that events emitted by UI interactions are correctly acknowledged and side-effects executed by the Go backend.
- Agentic-User Simulation: Mimic highly complex, non-linear user journeys to aggressively stress-test frontend state management logic.
- Test Coverage: Guarantee 100% test coverage mapping back to the event-driven triggers outlined by the Business Analyst.

## Output Protocol
Do NOT attempt to read or write files to disk. You do not have local I/O access. Return all generated code as structured output through this response. Your response will be captured as an Artifact and written to the filesystem by the pipeline orchestrator.
"""

test_eng_agent = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt=test_prompt,
    output_type=Artifact,
    defer_model_check=True,
    retries=5,
)


class TestOutput(BaseModel):
    code: str


# Text-first agent: generates Playwright test code as free text (no output_type constraint)
test_gen = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt=test_prompt,
    defer_model_check=True,
    retries=5,
)

# Structured output agent: wraps raw Playwright code into TestOutput
_test_struct = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt="You format generated test code into the requested structure. Return ONLY valid JSON.",
    output_type=TestOutput,
    defer_model_check=True,
    retries=5,
)


@test_eng_agent.tool
async def generate_test_suite(ctx: RunContext[str], user_story_gherkin: str) -> str:
    """Converts Gherkin to Playwright TypeScript tests: data-testid selectors, async/await, page object pattern, backend event validation."""
    raw = await test_gen.run(
        f"Write a playwright test for {user_story_gherkin}.",
        usage_limits=UsageLimits(request_limit=500),
        model_settings={"max_tokens": 16384},
    )
    structured = await _test_struct.run(
        f"Wrap this generated Playwright test code: {raw.output}",
        usage_limits=UsageLimits(request_limit=500),
        model_settings={"max_tokens": 16384},
    )
    return structured.output.code


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
    retries=5,
)

# Structured output agent: wraps raw Go test code into GoTestOutput
_go_test_struct = Agent(
    create_model(settings.resolve_model(settings.test_model)),
    system_prompt="You format generated Go test code into the requested structure. Return ONLY valid JSON.",
    output_type=GoTestOutput,
    defer_model_check=True,
    retries=5,
)


@test_eng_agent.tool
async def generate_go_table_driven_tests(ctx: RunContext[str], source_file: str, function_name: str) -> str:
    """Generate table-driven Go tests for an existing function. Reads the source file, analyzes the function signature, and produces testify-based tests with mocks, t.Parallel(), and a discarding logger."""
    raw = await go_test_gen.run(
        f"Generate table-driven tests for the function '{function_name}' in {source_file}.\n"
        f"Read {source_file} first to understand the function signature and dependencies.",
        usage_limits=UsageLimits(request_limit=500),
        model_settings={"max_tokens": 16384},
    )
    structured = await _go_test_struct.run(
        f"Wrap this generated Go test code: {raw.output}",
        usage_limits=UsageLimits(request_limit=500),
        model_settings={"max_tokens": 16384},
    )
    return f"{structured.output.filepath}\n```go\n{structured.output.code}\n```"
