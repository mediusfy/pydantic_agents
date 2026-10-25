"""ImplementationStage.run: what is asked of which agent, how artifacts are collected, and the per-story test phase.

The model-calling provider, registry, recorder, workspace and context retrieval are fakes, so these tests pin the
orchestration (which agents run, with what prompt, how results merge, when the run reports a failure) and not model
output.
"""

import contextlib
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.core.models.core import Artifact, EventTrigger, FailureReport, PipelineState, UserStory
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.settings import settings
from pydantic_agents.core.stages import implementation_stage as impl

EXT = {"backend": "go", "frontend": "vue", "database": "sql", "devops": "yml", "docs": "md", "test_eng": "go"}


class FakeProvider:
    """Answers every agent call with a one-file Artifact named after the agent and the story in the prompt."""

    def __init__(self, failures: dict[str, BaseException] | None = None):
        self.calls: list[tuple[str, str, object]] = []
        self.failures = failures or {}

    async def run_agent(self, _sem, agent_name, _agent, prompt, deps=None):
        self.calls.append((agent_name, prompt, deps))
        if agent_name in self.failures:
            raise self.failures[agent_name]
        story = (re.search(r"US-\d+", prompt) or SimpleNamespace(group=lambda: "global")).group()
        suffix = "_test" if agent_name == "test_eng" else ""
        files = {f"{agent_name}_{story}{suffix}.{EXT[agent_name]}": f"{agent_name} code for {story}"}
        return SimpleNamespace(output=Artifact(agent_name=agent_name, files=files, metadata={}))

    def agents(self) -> list[str]:
        return [c[0] for c in self.calls]

    def prompt(self, agent: str, story: str | None = None) -> str:
        return next(p for a, p, _ in self.calls if a == agent and (story is None or story in p))


class FakeRecorder:
    @contextlib.asynccontextmanager
    async def record_span(self, **kwargs):
        yield SimpleNamespace(set_output=lambda _res: None, span=kwargs)


class FakeWorkspace:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_repo_context(self):
        return "REPO-CONTEXT"

    def get_markdown_context(self):
        return "MARKDOWN-CONTEXT"


def _story(n: int, events: tuple[tuple[str, str], ...] = ()) -> UserStory:
    return UserStory(
        id=f"US-{n}",
        title=f"Title {n}",
        gherkin=f"Feature for US-{n}",
        event_triggers=[
            EventTrigger(id=f"EV-{i}", name=name, description=desc) for i, (name, desc) in enumerate(events)
        ],
    )


def _state(*stories: UserStory, **kwargs) -> PipelineState:
    state = PipelineState(run_id="run-1", stage="advisory", requirement="x", user_stories=list(stories), **kwargs)
    state.advisory_results = {"cto": "CTO-ADVICE", "ux": "UX-ADVICE"}
    return state


def _config(**kwargs) -> PipelineConfig:
    kwargs.setdefault("dry_run", False)
    config = PipelineConfig(requirement="build the thing", **kwargs)
    config.project = "proj"
    return config


@pytest.fixture
def env(monkeypatch):
    provider = FakeProvider()
    run_tests = AsyncMock(return_value="PASSED")
    monkeypatch.setattr(impl, "registry", MagicMock(get=lambda name: f"agent:{name}"))
    monkeypatch.setattr(impl, "recorder", FakeRecorder())
    monkeypatch.setattr("pydantic_agents.core.adapters.workspace.WorkspaceContext", FakeWorkspace)
    monkeypatch.setattr(
        "pydantic_agents.core.context_engine.retrieve_context", lambda agent, _gherkin: f"[{agent} guidance]"
    )
    monkeypatch.setattr("pydantic_agents.core.tools.test_tools.execute_test_suite", run_tests)
    return SimpleNamespace(provider=provider, run_tests=run_tests, monkeypatch=monkeypatch)


async def _run(env, state, config=None, provider=None):
    stage = impl.ImplementationStage(agent_provider=provider or env.provider)
    return await stage.run(state, config or _config())


def _by_agent(state: PipelineState) -> dict[str, list[Artifact]]:
    grouped: dict[str, list[Artifact]] = {}
    for art in state.artifacts:
        grouped.setdefault(art.agent_name, []).append(art)
    return grouped


# --- dry run ----------------------------------------------------------------------------------------------------


async def test_a_dry_run_returns_the_mock_artifacts_without_calling_any_agent(env):
    state = await _run(env, _state(_story(1)), _config(dry_run=True))

    assert [a.agent_name for a in state.artifacts] == ["backend", "frontend", "test"]
    assert state.stage == "implementation"
    assert env.provider.calls == []


# --- a first run ------------------------------------------------------------------------------------------------


async def test_a_first_run_asks_each_agent_once_per_story_plus_the_global_ones(env):
    state = await _run(env, _state(_story(1), _story(2)))

    assert sorted(env.provider.agents()) == sorted(
        ["backend"] * 2 + ["frontend"] * 2 + ["database"] * 2 + ["devops", "docs"] + ["test_eng"] * 2
    )
    assert state.stage == "implementation"


async def test_artifacts_come_back_in_task_order_then_tests(env):
    state = await _run(env, _state(_story(1), _story(2)))

    assert [(a.agent_name, a.metadata["user_story_id"]) for a in state.artifacts] == [
        ("backend", "US-1"),
        ("frontend", "US-1"),
        ("database", "US-1"),
        ("backend", "US-2"),
        ("frontend", "US-2"),
        ("database", "US-2"),
        ("devops", "global"),
        ("docs", "global"),
        ("test_eng", "US-1"),
        ("test_eng", "US-2"),
    ]


async def test_backend_and_test_artifacts_record_their_story_index(env):
    state = await _run(env, _state(_story(1), _story(2)))

    grouped = _by_agent(state)
    assert [a.metadata["story_index"] for a in grouped["backend"]] == [0, 1]
    assert [a.metadata["story_index"] for a in grouped["test_eng"]] == [0, 1]
    assert "story_index" not in grouped["frontend"][0].metadata


async def test_the_backend_prompt_carries_context_and_the_hexagonal_brief(env):
    await _run(env, _state(_story(1)))

    prompt = env.provider.prompt("backend")
    assert prompt.startswith("REPO-CONTEXT\n\nCTO ADVICE:\nCTO-ADVICE\n\nFULL GHERKIN:\nFeature for US-1\n\n")
    assert "Hexagonal Architecture" in prompt
    assert settings.go_entrypoint in prompt
    assert "Run `run_go_build` to self-validate" in prompt
    assert prompt.endswith("\n\nRetrieved Backend Guidelines:\n[backend guidance]")
    assert "PREVIOUS FAILURE" not in prompt


async def test_the_backend_receives_the_project_state_as_deps(env):
    await _run(env, _state(_story(1)))

    [(_, _, deps)] = [c for c in env.provider.calls if c[0] == "backend"]
    assert deps.base_path == "/app/proj"
    assert (deps.environment, deps.team_owner) == ("dev", "platform-engineering")


async def test_the_frontend_prompt_adds_markdown_ux_and_the_story_title(env):
    await _run(env, _state(_story(1)))

    prompt = env.provider.prompt("frontend")
    assert prompt.startswith(
        "REPO-CONTEXT\n\nMARKDOWN-CONTEXT\n\nCTO ADVICE:\nCTO-ADVICE\n\nUX GUIDELINES:\nUX-ADVICE\n\n"
    )
    assert "TASK:\nImplement frontend for story US-1: Title 1" in prompt
    assert prompt.endswith("Retrieved Frontend Guidelines:\n[frontend guidance]")


async def test_the_database_prompt_lists_the_story_events(env):
    story = _story(1, events=(("OrderPlaced", "an order is placed"), ("OrderPaid", "payment received")))

    await _run(env, _state(story))

    prompt = env.provider.prompt("database")
    assert "EVENTS:\n - OrderPlaced: an order is placed\n - OrderPaid: payment received\n\n" in prompt
    assert "Generate database schema/migrations for story US-1: Title 1" in prompt
    assert prompt.endswith("Retrieved Database Guidelines:\n[database guidance]")


async def test_the_global_prompts_list_the_stories_and_the_requirement(env):
    await _run(env, _state(_story(1), _story(2)))

    assert env.provider.prompt("devops") == (
        "Generate CI/CD pipelines and Docker Compose for these stories:\n- US-1: Title 1\n- US-2: Title 2"
    )
    assert env.provider.prompt("docs") == "Generate API docs and README for: build the thing"


async def test_missing_retrieved_context_leaves_the_prompts_without_a_guidelines_section(env):
    env.monkeypatch.setattr("pydantic_agents.core.context_engine.retrieve_context", lambda _agent, _gherkin: "")

    await _run(env, _state(_story(1)))

    for agent in ("backend", "frontend", "database", "test_eng"):
        assert "Retrieved" not in env.provider.prompt(agent)


async def test_the_test_agent_prompt_uses_the_gherkin_and_test_guidelines(env):
    await _run(env, _state(_story(1)))

    assert env.provider.prompt("test_eng") == (
        "Write tests for gherkin: Feature for US-1\n\nRetrieved Test Guidelines:\n[test_eng guidance]"
    )


# --- failures during generation ---------------------------------------------------------------------------------


async def test_an_agent_that_raises_is_logged_and_the_others_still_land(env, caplog):
    provider = FakeProvider(failures={"frontend": RuntimeError("model down")})

    with caplog.at_level(logging.ERROR, logger=impl.logger.name):
        state = await _run(env, _state(_story(1)), provider=provider)

    assert "frontend" not in _by_agent(state)
    assert {"backend", "database", "devops", "docs", "test_eng"} <= set(_by_agent(state))
    assert any("Implementation task (frontend) raised: model down" in r.getMessage() for r in caplog.records)


# --- a retry after a failed test suite --------------------------------------------------------------------------


def _retry_state(*stories: UserStory, existing: list[Artifact]) -> PipelineState:
    state = _state(*stories)
    state.failure_report = FailureReport(source="test_suite", error_type="test_failure", trace="US-1 broke")
    state.artifacts = existing
    return state


async def test_a_retry_only_reruns_the_backend_and_tests(env):
    state = _retry_state(_story(1), _story(2), existing=[])

    await _run(env, state)

    assert sorted(env.provider.agents()) == ["backend", "backend", "test_eng", "test_eng"]


async def test_the_retry_backend_prompt_carries_the_failure_and_asks_for_a_patch(env):
    await _run(env, _retry_state(_story(1), existing=[]))

    prompt = env.provider.prompt("backend")
    assert prompt.startswith("REPO-CONTEXT\n\nCTO ADVICE:\nCTO-ADVICE\n\nFULL GHERKIN:\nFeature for US-1\n\n")
    assert (
        "PREVIOUS FAILURE DETECTED:\nSource: test_suite\nError Type: test_failure\nTrace/Feedback:\nUS-1 broke\n\n"
        in prompt
    )
    assert "generate a hyper-specific patch" in prompt
    assert "Hexagonal Architecture" not in prompt
    assert prompt.endswith("Retrieved Backend Guidelines:\n[backend guidance]")


async def test_a_retry_merges_new_files_into_the_existing_artifact_for_the_story(env):
    old = Artifact(
        agent_name="backend", files={"keep.go": "old", "backend_US-1.go": "stale"}, metadata={"user_story_id": "US-1"}
    )
    other = Artifact(agent_name="frontend", files={"app.vue": "ui"}, metadata={"user_story_id": "US-1"})

    state = await _run(env, _retry_state(_story(1), existing=[old, other]))

    backend = _by_agent(state)["backend"]
    assert len(backend) == 1
    assert backend[0].files == {"keep.go": "old", "backend_US-1.go": "backend code for US-1"}
    assert _by_agent(state)["frontend"] == [other]


async def test_a_retry_appends_an_artifact_for_a_story_with_no_existing_one(env):
    existing = Artifact(agent_name="backend", files={"a.go": "x"}, metadata={"user_story_id": "US-1"})

    state = await _run(env, _retry_state(_story(2), existing=[existing]))

    assert [a.metadata["user_story_id"] for a in _by_agent(state)["backend"]] == ["US-1", "US-2"]


# --- the per-story test phase -----------------------------------------------------------------------------------


async def test_the_test_suite_runs_with_the_story_s_test_and_backend_source(env):
    await _run(env, _state(_story(1)))

    env.run_tests.assert_awaited_once_with(
        "test_eng_US-1_test.go", "test_eng code for US-1", "backend_US-1.go", "backend code for US-1"
    )


async def test_a_passing_suite_clears_any_earlier_failure(env):
    state = _retry_state(_story(1), existing=[])
    state.feedback = "old feedback"

    state = await _run(env, state)

    assert (state.feedback, state.failure_report, state.approved) == (None, None, False)


async def test_a_failing_suite_records_the_failed_stories(env, caplog):
    env.run_tests.side_effect = ["FAILED: assertion", "PASSED", "FAILED: compile error"]

    with caplog.at_level(logging.WARNING, logger=impl.logger.name):
        state = await _run(env, _state(_story(1), _story(2), _story(3)))

    assert state.feedback == (
        "Test suite failed for stories: US-1, US-3. Rewrite the implementations to resolve these errors."
    )
    report = state.failure_report
    assert (report.source, report.error_type, report.trace) == (
        "test_suite",
        "test_failure",
        "Failed stories: US-1, US-3",
    )
    assert report.relevant_files == ["US-1", "US-3"]
    assert state.approved is False
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Test suite failed for US-1: FAILED: assertion") for m in messages)
    assert any("Test failures in 2/3 stories: US-1, US-3" in m for m in messages)


async def test_a_story_with_no_backend_source_is_not_run(env):
    provider = FakeProvider(failures={"backend": RuntimeError("no backend")})

    state = await _run(env, _state(_story(1)), provider=provider)

    env.run_tests.assert_not_awaited()
    assert state.failure_report is None


async def test_the_first_matching_source_file_is_the_test_target(env):
    class TwoFileProvider(FakeProvider):
        async def run_agent(self, sem, agent_name, agent, prompt, deps=None):
            res = await super().run_agent(sem, agent_name, agent, prompt, deps)
            if agent_name == "backend":
                res.output.files = {"README.md": "docs", "api.go": "package api", "other.go": "package other"}
            return res

    await _run(env, _state(_story(1)), provider=TwoFileProvider())

    assert env.run_tests.await_args.args[2:] == ("api.go", "package api")


async def test_a_test_agent_that_raises_is_skipped_and_not_counted_as_a_failure(env, caplog):
    provider = FakeProvider(failures={"test_eng": RuntimeError("no tests today")})

    with caplog.at_level(logging.ERROR, logger=impl.logger.name):
        state = await _run(env, _state(_story(1)), provider=provider)

    assert "test_eng" not in _by_agent(state)
    env.run_tests.assert_not_awaited()
    assert state.failure_report is None
    assert any("Test agent for US-1 failed: no tests today" in r.getMessage() for r in caplog.records)


async def test_a_new_test_artifact_replaces_the_story_s_old_one(env):
    old_test = Artifact(agent_name="test_eng", files={"old_test.go": "stale"}, metadata={"user_story_id": "US-1"})
    other_test = Artifact(agent_name="test_eng", files={"other_test.go": "keep"}, metadata={"user_story_id": "US-2"})

    state = await _run(env, _retry_state(_story(1), existing=[old_test, other_test]))

    tests = _by_agent(state)["test_eng"]
    assert [list(a.files) for a in tests] == [["other_test.go"], ["test_eng_US-1_test.go"]]


async def test_with_no_user_stories_only_the_global_agents_run(env):
    state = await _run(env, _state())

    assert sorted(env.provider.agents()) == ["devops", "docs"]
    assert state.failure_report is None
    assert state.stage == "implementation"
