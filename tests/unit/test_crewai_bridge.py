"""Pins the CrewAI bridge against a fake ``crewai`` package (the real one is not installed in CI)."""

import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import Agent, RunContext

from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.models.core import Artifact, PipelineState
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings

_OUR_CREWAI_MODULES = (
    "pydantic_agents.core.pipeline.crewai.crews",
    "pydantic_agents.core.pipeline.crewai.bridge",
    "pydantic_agents.core.pipeline.crewai.orchestrator",
)


class FakeLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **kwargs: Any):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.kwargs = kwargs


class _KwargsRecord:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeAgent(_KwargsRecord):
    pass


class FakeTask(_KwargsRecord):
    pass


class FakeCrew(_KwargsRecord):
    kickoff_result: Any = None

    def kickoff(self) -> Any:
        return self.kickoff_result


class FakeProcess:
    sequential = "sequential"
    hierarchical = "hierarchical"


class FakeBaseTool:
    name: str = ""
    description: str = ""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


@pytest.fixture
def fake_crewai(monkeypatch):
    """Install a fake ``crewai`` package and make the bridge modules import against it."""
    crewai = types.ModuleType("crewai")
    tools = types.ModuleType("crewai.tools")
    crewai.Agent = FakeAgent  # type: ignore[attr-defined]
    crewai.Crew = FakeCrew  # type: ignore[attr-defined]
    crewai.Process = FakeProcess  # type: ignore[attr-defined]
    crewai.Task = FakeTask  # type: ignore[attr-defined]
    crewai.LLM = FakeLLM  # type: ignore[attr-defined]
    tools.BaseTool = FakeBaseTool  # type: ignore[attr-defined]
    crewai.tools = tools  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crewai", crewai)
    monkeypatch.setitem(sys.modules, "crewai.tools", tools)
    for name in _OUR_CREWAI_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        yield crewai
    finally:
        # Modules imported during the test are bound to the fake; never leak them into other tests.
        for name in _OUR_CREWAI_MODULES:
            sys.modules.pop(name, None)


@pytest.fixture
def litellm_env(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "litellm")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.test:4000")
    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "sk-test-virtual")
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setattr(settings, "model_provider", "litellm")


@pytest.fixture
def registered_agents():
    register_all_agents()


def _agent_with_tools() -> Agent:
    agent = Agent("test", system_prompt="You are a tool tester.", defer_model_check=True)

    def plain(x: int) -> int:
        """Return x unchanged."""
        return x

    async def async_plain(x: int) -> int:
        """Return x after awaiting."""
        await asyncio.sleep(0)
        return x

    def needs_ctx(ctx: RunContext[None], x: int) -> int:
        return x

    agent.tool_plain(plain)
    agent.tool_plain(async_plain)
    agent.tool(needs_ctx)
    return agent


class TestToolBridging:
    def test_tools_are_bridged_by_value_and_runcontext_tools_are_skipped(self, fake_crewai):
        from pydantic_agents.core.adapters import crewai_bridge

        tools = crewai_bridge._extract_tool_defs(_agent_with_tools())

        by_name = {tool.name: tool for tool in tools}
        assert set(by_name) == {"plain", "async_plain"}
        assert all(isinstance(tool, FakeBaseTool) for tool in tools)
        assert by_name["plain"].description == "Return x unchanged."
        assert by_name["plain"].run(x=2) == 2
        assert by_name["async_plain"].run(x=3) == 3

    async def test_async_tool_runs_inside_a_running_loop(self, fake_crewai):
        from pydantic_agents.core.adapters import crewai_bridge

        tools = {tool.name: tool for tool in crewai_bridge._extract_tool_defs(_agent_with_tools())}
        assert tools["async_plain"].run(x=5) == 5

    async def test_bridged_calls_share_one_bounded_executor(self, fake_crewai, monkeypatch):
        """Every call inside a running loop goes to the module-level pool, not a pool per call."""
        from pydantic_agents.core.adapters import crewai_bridge

        submitted: list[object] = []
        real_submit = crewai_bridge._TOOL_EXECUTOR.submit

        def spy(fn, *args, **kwargs):
            submitted.append(fn)
            return real_submit(fn, *args, **kwargs)

        monkeypatch.setattr(crewai_bridge._TOOL_EXECUTOR, "submit", spy)
        tools = {tool.name: tool for tool in crewai_bridge._extract_tool_defs(_agent_with_tools())}
        assert [tools["async_plain"].run(x=i) for i in range(3)] == [0, 1, 2]
        assert len(submitted) == 3  # every call went through the one module-level pool

    def test_agent_without_toolset_yields_no_tools(self, fake_crewai):
        from pydantic_agents.core.adapters import crewai_bridge

        assert crewai_bridge._extract_tool_defs(SimpleNamespace()) == []


class TestLlmResolution:
    def test_litellm_provider_builds_prefixed_llm_from_environment(self, fake_crewai, litellm_env):
        from pydantic_agents.core.adapters import crewai_bridge

        agent = Agent(create_model("openai:deepseek-v4-pro"), system_prompt="hi", defer_model_check=True)
        llm = crewai_bridge.build_crewai_llm(agent)

        assert isinstance(llm, FakeLLM)
        assert llm.model == "openai/deepseek-v4-pro"
        assert llm.base_url == "http://litellm.test:4000/v1"
        assert llm.api_key == "sk-test-virtual"

    def test_as_crewai_agent_carries_llm_backstory_and_tools(self, fake_crewai, litellm_env):
        from pydantic_agents.core.adapters import crewai_bridge

        crew_agent = crewai_bridge.as_crewai_agent(_agent_with_tools(), role="Tester", goal="Test things")

        assert isinstance(crew_agent, FakeAgent)
        assert crew_agent.role == "Tester"
        assert crew_agent.backstory == "You are a tool tester."
        assert isinstance(crew_agent.llm, FakeLLM)
        assert crew_agent.llm.model == "openai/test"
        assert {tool.name for tool in crew_agent.tools} == {"plain", "async_plain"}

    def test_non_litellm_provider_passes_model_name_through(self, fake_crewai, monkeypatch):
        from pydantic_agents.core.adapters import crewai_bridge

        monkeypatch.setattr(settings, "model_provider", "deepseek")
        agent = Agent(create_model("deepseek:deepseek-reasoner"), defer_model_check=True)

        assert crewai_bridge.build_crewai_llm(agent) == "deepseek-reasoner"

    def test_agent_without_model_is_rejected(self, fake_crewai):
        from pydantic_agents.core.adapters import crewai_bridge

        agent = SimpleNamespace(_model=None)
        with pytest.raises(ValueError, match="no model"):
            crewai_bridge.build_crewai_llm(agent)


def _state(**overrides: Any) -> PipelineState:
    fields: dict[str, Any] = {"run_id": "run-crew", "stage": "init", "requirement": "Build a billing service"}
    fields.update(overrides)
    return PipelineState(**fields)


CREW_CASES = [
    pytest.param(
        "SdlcCrew",
        "sequential",
        6,
        ["ba/note", "cto/note", "backend/artifact", "frontend/artifact", "test_eng/artifact", "review/verdict"],
        id="sdlc",
    ),
    pytest.param(
        "ResearchCrew",
        "hierarchical",
        3,
        ["weather/note", "crypto_oracle/note", "market_shark/note"],
        id="research",
    ),
    pytest.param(
        "SecurityReviewCrew",
        "sequential",
        3,
        ["security/note", "sre/note", "review/verdict"],
        id="security_review",
    ),
    pytest.param(
        "ArchitectureReviewCrew",
        "hierarchical",
        5,
        ["ba/note", "security/note", "database/note", "sre/note", "cto/advice"],
        id="architecture_review",
    ),
    pytest.param(
        "AuditCrew",
        "sequential",
        4,
        ["security/note", "devops/note", "sre/note", "review/findings"],
        id="audit",
    ),
]


class TestCrews:
    @pytest.mark.parametrize(("crew_name", "process", "agent_count", "task_names"), CREW_CASES)
    def test_build_accepts_pipeline_state(
        self, fake_crewai, litellm_env, registered_agents, crew_name, process, agent_count, task_names
    ):
        crews = importlib.import_module("pydantic_agents.core.pipeline.crewai.crews")
        state = _state(artifacts=[Artifact(agent_name="backend", files={"main.go": "package main"})])

        crew = getattr(crews, crew_name)().build(state)

        assert isinstance(crew, FakeCrew)
        assert crew.process == process
        assert len(crew.agents) == agent_count
        assert all(isinstance(agent, FakeAgent) for agent in crew.agents)
        assert all(isinstance(agent.llm, FakeLLM) and agent.llm.model.startswith("openai/") for agent in crew.agents)
        assert [task.name for task in crew.tasks] == task_names
        assert all(task.agent in crew.agents for task in crew.tasks)
        if process == "hierarchical":
            assert isinstance(crew.manager_llm, FakeLLM)
            assert crew.manager_llm.model.startswith("openai/")
        else:
            assert "manager_llm" not in crew.kwargs

    def test_requirement_and_artifacts_reach_the_task_descriptions(self, fake_crewai, litellm_env, registered_agents):
        crews = importlib.import_module("pydantic_agents.core.pipeline.crewai.crews")
        state = _state(artifacts=[Artifact(agent_name="backend", files={"main.go": "package main"})])

        research = crews.ResearchCrew().build(state)
        audit = crews.AuditCrew().build(state)

        assert all("Build a billing service" in task.description for task in research.tasks)
        assert "package main" in audit.tasks[0].description

    def test_qa_tasks_require_criterion_ids(self, fake_crewai, litellm_env, registered_agents):
        crews = importlib.import_module("pydantic_agents.core.pipeline.crewai.crews")
        from pydantic_agents.core.models.core import TestRunReport

        qa = crews.QaCrew()
        report = TestRunReport()
        tests_crew = qa.build_tests(_state())
        execution_crew = qa.build_execution("tests go here")
        fix_crew = qa.build_fix("tests go here", report)
        signoff_crew = qa.build_signoff("tests go here", report)

        assert all("criterion id" in task.description for task in tests_crew.tasks)
        assert "criterion id" in execution_crew.tasks[0].description
        assert "criterion id" in fix_crew.tasks[0].description
        assert "criterion id" in signoff_crew.tasks[0].description
        assert "approved" in signoff_crew.tasks[0].expected_output

    def test_qa_fix_names_only_the_failing_criteria(self, fake_crewai, litellm_env, registered_agents):
        crews = importlib.import_module("pydantic_agents.core.pipeline.crewai.crews")
        from pydantic_agents.core.models.core import CriterionResult, TestRunReport

        report = TestRunReport(
            criterion_results=[
                CriterionResult(criterion_id="ORDERS-01-AC-1", passed=True),
                CriterionResult(criterion_id="ORDERS-01-AC-2", passed=False),
            ],
            unexecuted_criterion_ids=["ORDERS-01-AC-3"],
        )

        fix_crew = crews.QaCrew().build_fix("tests go here", report)

        description = fix_crew.tasks[0].description
        assert "ORDERS-01-AC-2" in description
        assert "ORDERS-01-AC-3" in description
        assert "ORDERS-01-AC-1" not in description

    def test_json_contract_tasks_name_the_schema(self, fake_crewai, litellm_env, registered_agents):
        crews = importlib.import_module("pydantic_agents.core.pipeline.crewai.crews")

        decision = crews.ArchitectureReviewCrew().build(_state()).tasks[-1]

        assert "approved_to_proceed" in decision.expected_output
        assert "CRITICAL" in decision.description

    def test_all_crews_are_registered_for_their_blueprints(self, fake_crewai, litellm_env):
        from pydantic_agents.core.pipeline.config import BLUEPRINTS

        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")

        crew_blueprints = {name: bp["crew_class"] for name, bp in BLUEPRINTS.items() if bp["framework"] == "crewai"}
        assert set(crew_blueprints.values()) <= set(bridge._CREW_MAP)
        assert crew_blueprints["crew_qa"] == "QaCrew"
        assert crew_blueprints["crew_architecture_review"] == "ArchitectureReviewCrew"
        assert crew_blueprints["crew_audit"] == "AuditCrew"


def _task_output(name: str | None, output: str, agent: str = "Someone") -> SimpleNamespace:
    return SimpleNamespace(name=name, agent=agent, raw=output, output=output)


_VALID_VERDICT = json.dumps(
    {"approved": True, "feedback": ["ORDERS-01-AC-1 covered"], "security_score": 90, "risk_level": "LOW"}
)
_VALID_ADVICE = json.dumps(
    {"bottlenecks_identified": [], "recommendations": ["ship"], "adr_id": "0002", "approved_to_proceed": True}
)
_BLOCKING_ADVICE = json.dumps({"bottlenecks_identified": ["CRITICAL: no auth"], "approved_to_proceed": False})
_GO_TEST = "tests for ORDERS-01-AC-1\n\n`billing_test.go`\n```go\npackage billing\n```\n"


class TestApplyResult:
    @pytest.mark.parametrize(
        ("output", "approved", "error_fragment"),
        [
            pytest.param("The change is not approved; please fix the handler.", False, "no JSON object", id="prose"),
            pytest.param("Verdict: APPROVED", False, "no JSON object", id="approved_prose"),
            pytest.param(_VALID_VERDICT, True, None, id="valid_json"),
            pytest.param(f"Here is my verdict:\n```json\n{_VALID_VERDICT}\n```", True, None, id="fenced_json"),
            pytest.param('{"approved": true}', False, "failed validation", id="incomplete_json"),
        ],
    )
    def test_review_verdict_is_parsed_as_review_result(self, fake_crewai, output, approved, error_fragment):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()

        bridge._apply_result(state, SimpleNamespace(tasks_output=[_task_output("review/verdict", output)]))

        assert state.approved is approved
        assert state.feedback
        if error_fragment is None:
            assert state.errors == []
        else:
            assert len(state.errors) == 1
            assert error_fragment in state.errors[0]

    @pytest.mark.parametrize(
        ("output", "cto_approved", "error_count"),
        [
            pytest.param(_VALID_ADVICE, True, 0, id="approved"),
            pytest.param(_BLOCKING_ADVICE, False, 0, id="blocked"),
            pytest.param("Looks fine to me.", False, 1, id="prose"),
        ],
    )
    def test_cto_advice_sets_the_cto_gate(self, fake_crewai, output, cto_approved, error_count):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()

        bridge._apply_result(state, SimpleNamespace(tasks_output=[_task_output("cto/advice", output)]))

        assert state.advisory_results["cto"] == output
        assert state.cto_approved is cto_approved
        assert len(state.errors) == error_count

    def test_qa_test_outputs_merge_into_one_test_eng_artifact(self, fake_crewai):
        """The fix task re-emits the files it changed: per path the later task wins, the rest is kept."""
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()
        fixed = "fixed ORDERS-01-AC-1\n\n`billing_test.go`\n```go\npackage billing // fixed\n```\n"
        result = SimpleNamespace(
            tasks_output=[
                _task_output("ba/note", "ORDERS-01-AC-1: Given ... When ... Then ..."),
                _task_output("test_eng/artifact", _GO_TEST + "`e2e.spec.ts`\n```ts\ntest('ORDERS-01-AC-1')\n```\n"),
                _task_output("test_eng/artifact", fixed),
            ]
        )

        bridge._apply_result(state, result)

        assert state.advisory_results["ba"].startswith("ORDERS-01-AC-1")
        assert [art.agent_name for art in state.artifacts] == ["test_eng"]
        assert state.artifacts[0].files == {
            "billing_test.go": "package billing // fixed",
            "e2e.spec.ts": "test('ORDERS-01-AC-1')",
        }

    def test_audit_findings_become_feedback(self, fake_crewai):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()
        table = (
            "| id | severity | location | finding | owner | due date |\n"
            "| 1 | HIGH | main.go:12 | no limits | sre | 2026-11-01 |"
        )
        result = SimpleNamespace(
            tasks_output=[
                _task_output("security/note", "secret at config.go:3"),
                _task_output("review/feedback", table),
            ]
        )

        bridge._apply_result(state, result)

        assert state.feedback == table
        assert state.advisory_results["security"] == "secret at config.go:3"
        assert state.approved is False

    def test_findings_drops_entries_missing_a_location(self, fake_crewai):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()
        findings = json.dumps(
            [
                {
                    "id": "f1",
                    "source_agent": "security",
                    "severity": "HIGH",
                    "description": "no limits",
                    "location": "main.go:12",
                    "owner": "sre",
                    "due_date": "2026-11-01",
                },
                {
                    "id": "f2",
                    "source_agent": "security",
                    "severity": "LOW",
                    "description": "vague concern",
                    "owner": "sre",
                    "due_date": "2026-11-01",
                },
            ]
        )

        bridge._apply_result(state, SimpleNamespace(tasks_output=[_task_output("review/findings", findings)]))

        assert "main.go:12" in state.feedback
        assert "1 finding(s) dropped" in state.feedback
        assert state.errors == []

    def test_findings_rejected_when_owner_or_due_date_missing(self, fake_crewai):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()
        findings = json.dumps(
            [
                {
                    "id": "f1",
                    "source_agent": "security",
                    "severity": "HIGH",
                    "description": "no limits",
                    "location": "main.go:12",
                    "owner": "",
                    "due_date": "",
                },
            ]
        )

        bridge._apply_result(state, SimpleNamespace(tasks_output=[_task_output("review/findings", findings)]))

        assert state.feedback is None
        assert len(state.errors) == 1
        assert "missing owner/due_date" in state.errors[0]

    @pytest.mark.parametrize(
        "item",
        [
            {"id": "f1", "source_agent": "security", "severity": "SEVERE", "description": "x", "location": "a.go:1"},
            {"id": "f1", "severity": "HIGH", "description": "x", "location": "a.go:1"},
            "a.go:1 is broken",
        ],
    )
    def test_findings_with_a_location_but_otherwise_invalid_reject_the_batch(self, fake_crewai, item):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()

        bridge._apply_result(state, SimpleNamespace(tasks_output=[_task_output("review/findings", json.dumps([item]))]))

        assert state.feedback is None
        assert len(state.errors) == 1
        assert "invalid finding" in state.errors[0]

    def test_findings_accept_lowercase_severity(self, fake_crewai):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()
        item = {
            "id": "f1",
            "source_agent": "security",
            "severity": "critical",
            "description": "leaked key",
            "location": "config.go:3",
            "owner": "security",
            "due_date": "2026-10-30",
        }

        bridge._apply_result(state, SimpleNamespace(tasks_output=[_task_output("review/findings", json.dumps([item]))]))

        assert state.errors == []
        assert "[CRITICAL] config.go:3" in state.feedback

    def test_unnamed_tasks_fall_back_to_the_agent_role(self, fake_crewai):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        state = _state()
        result = SimpleNamespace(
            tasks_output=[
                _task_output(None, "stories", agent="Senior Business Analyst"),
                _task_output(None, "`a.go`\n```go\npackage a\n```", agent="Lead Go Backend Engineer"),
                _task_output(None, "not approved", agent="Code Review Engineer"),
            ]
        )

        bridge._apply_result(state, result)

        assert state.advisory_results["ba"] == "stories"
        assert state.artifacts[0].agent_name == "backend"
        assert state.approved is False


class TestRunCrewaiPipeline:
    async def test_crew_is_built_from_state_and_result_applied(self, fake_crewai, litellm_env, monkeypatch):
        from pydantic_agents.core.pipeline.config import PipelineConfig

        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        recorder = MagicMock()
        recorder.record_span.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        recorder.record_span.return_value.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(bridge, "recorder", recorder)

        built_with: list[PipelineState] = []

        class StubCrew:
            def build(self, state: PipelineState) -> FakeCrew:
                built_with.append(state)
                crew = FakeCrew(agents=[], tasks=[], process="sequential")
                crew.kickoff_result = SimpleNamespace(
                    tasks_output=[
                        _task_output("test_eng/artifact", _GO_TEST),
                        _task_output("review/verdict", _VALID_VERDICT),
                    ]
                )
                return crew

        monkeypatch.setitem(bridge._CREW_MAP, "SdlcCrew", StubCrew)
        config = PipelineConfig(blueprint="crew_sdlc", framework="crewai", requirement="Build a billing service")
        state = _state()

        final = await bridge.run_crewai_pipeline(state, config)

        assert built_with == [state]
        assert final.approved is True
        assert final.errors == []
        assert [art.agent_name for art in final.artifacts] == ["test_eng"]
        assert final.active_context is not None
        assert final.active_context.modified_files == ["billing_test.go"]
        assert final.stage == "implementation"

    @pytest.mark.parametrize("crew_class_name", ["QaCrew", "RenamedQaCrew"])
    async def test_staged_crews_dispatch_by_class_not_by_name(
        self, fake_crewai, litellm_env, monkeypatch, crew_class_name
    ):
        """The QaCrew runner is keyed by class: renaming the blueprint's crew_class entry keeps the staged loop."""
        from pydantic_agents.core.pipeline.config import BLUEPRINTS, BlueprintSpec, Framework, PipelineConfig

        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        recorder = MagicMock()
        recorder.record_span.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        recorder.record_span.return_value.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(bridge, "recorder", recorder)
        qa_cls = bridge._CREW_MAP["QaCrew"]
        monkeypatch.setitem(bridge._CREW_MAP, crew_class_name, qa_cls)
        monkeypatch.setitem(
            BLUEPRINTS,
            "crew_qa_test",
            BlueprintSpec(
                name="crew_qa_test", description="qa", crew_class=crew_class_name, framework=Framework.CREWAI
            ).to_dict(),
        )
        ran_with: list[PipelineState] = []

        async def fake_runner(state: PipelineState) -> str:
            ran_with.append(state)
            state.approved = True
            return "QaCrew finished: report=parsed"

        monkeypatch.setitem(bridge._STAGED_RUNNERS, qa_cls, fake_runner)
        config = PipelineConfig(blueprint="crew_qa_test", framework="crewai", requirement="Build a billing service")
        state = _state()

        final = await bridge.run_crewai_pipeline(state, config)

        assert ran_with == [state]
        assert final.approved is True
        assert final.stage == "implementation"
        assert final.errors == []


_PASSING_REPORT = json.dumps(
    {
        "criterion_results": [{"criterion_id": "ORDERS-01-AC-1", "passed": True}],
        "coverage_percent": 90,
    }
)
_FAILING_REPORT = json.dumps(
    {
        "criterion_results": [{"criterion_id": "ORDERS-01-AC-1", "passed": False}],
        "coverage_percent": 50,
    }
)


class TestQaCrewStagedGate:
    @staticmethod
    def _output_for(crew: FakeCrew, exec_output: str) -> SimpleNamespace:
        names = [task.name for task in crew.tasks]
        if names == ["ba/note", "test_eng/artifact"]:
            return SimpleNamespace(
                tasks_output=[_task_output("ba/note", "stories"), _task_output("test_eng/artifact", _GO_TEST)]
            )
        if names == ["qa_executor/note"]:
            return SimpleNamespace(tasks_output=[_task_output("qa_executor/note", exec_output)])
        if names == ["test_eng/artifact"]:
            return SimpleNamespace(tasks_output=[_task_output("test_eng/artifact", _GO_TEST)])
        if names == ["review/verdict"]:
            return SimpleNamespace(tasks_output=[_task_output("review/verdict", _VALID_VERDICT)])
        raise AssertionError(f"unexpected crew tasks: {names}")

    async def test_gate_passes_without_a_fix_round_when_the_first_run_is_clean(
        self, fake_crewai, litellm_env, registered_agents, monkeypatch
    ):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        calls: list[list[str]] = []

        async def fake_run(crew: FakeCrew) -> SimpleNamespace:
            calls.append([task.name for task in crew.tasks])
            return self._output_for(crew, _PASSING_REPORT)

        monkeypatch.setattr(bridge, "_run_crew_async", fake_run)
        state = _state()

        summary = await bridge._run_qa_staged_gate(state)

        assert calls == [["ba/note", "test_eng/artifact"], ["qa_executor/note"], ["review/verdict"]]
        assert state.approved is True
        assert state.test_run_report is not None
        assert state.test_run_report.all_passed is True
        assert state.errors == []
        assert "all_passed=True" in summary

    async def test_one_fix_round_then_the_gate_passes(self, fake_crewai, litellm_env, registered_agents, monkeypatch):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        reports = iter([_FAILING_REPORT, _PASSING_REPORT])

        async def fake_run(crew: FakeCrew) -> SimpleNamespace:
            names = [task.name for task in crew.tasks]
            if names == ["qa_executor/note"]:
                return self._output_for(crew, next(reports))
            return self._output_for(crew, "")

        monkeypatch.setattr(bridge, "_run_crew_async", fake_run)
        state = _state()

        await bridge._run_qa_staged_gate(state)

        assert state.approved is True
        assert state.test_run_report is not None
        assert state.test_run_report.all_passed is True

    async def test_signoff_crew_sees_the_latest_report_after_a_fix_round(
        self, fake_crewai, litellm_env, registered_agents, monkeypatch
    ):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        reports = iter([_FAILING_REPORT, _PASSING_REPORT])
        signoff_crews: list[FakeCrew] = []

        async def fake_run(crew: FakeCrew) -> SimpleNamespace:
            names = [task.name for task in crew.tasks]
            if names == ["qa_executor/note"]:
                return self._output_for(crew, next(reports))
            if names == ["review/verdict"]:
                signoff_crews.append(crew)
            return self._output_for(crew, "")

        monkeypatch.setattr(bridge, "_run_crew_async", fake_run)
        state = _state()

        await bridge._run_qa_staged_gate(state)

        assert len(signoff_crews) == 1
        description = signoff_crews[0].tasks[0].description
        assert '"all_passed":true' in description.replace(" ", "")
        assert '"coverage_percent":90' in description.replace(" ", "")

    async def test_gate_blocks_approval_after_the_fix_loop_is_exhausted(
        self, fake_crewai, litellm_env, registered_agents, monkeypatch
    ):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")

        async def fake_run(crew: FakeCrew) -> SimpleNamespace:
            return self._output_for(crew, _FAILING_REPORT)

        monkeypatch.setattr(bridge, "_run_crew_async", fake_run)
        state = _state()

        await bridge._run_qa_staged_gate(state)

        assert state.approved is False
        assert state.test_run_report is not None
        assert state.test_run_report.all_passed is False
        assert any("fix loop exhausted" in err for err in state.errors)

    async def test_unparseable_execution_report_skips_the_fix_round(
        self, fake_crewai, litellm_env, registered_agents, monkeypatch
    ):
        """A qa_executor output that doesn't parse as TestRunReport is not a test failure: don't burn a fix
        round feeding go_test a meaningless empty failing-id list, just fail the gate closed."""
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        calls: list[list[str]] = []

        async def fake_run(crew: FakeCrew) -> SimpleNamespace:
            names = [task.name for task in crew.tasks]
            calls.append(names)
            if names == ["qa_executor/note"]:
                return self._output_for(crew, "not json at all")
            return self._output_for(crew, "")

        monkeypatch.setattr(bridge, "_run_crew_async", fake_run)
        state = _state()

        await bridge._run_qa_staged_gate(state)

        assert calls == [["ba/note", "test_eng/artifact"], ["qa_executor/note"], ["review/verdict"]]
        assert state.approved is False
        assert state.test_run_report is None
        assert any("did not parse" in err for err in state.errors)
        assert not any("fix loop exhausted" in err for err in state.errors)

    async def test_signoff_is_told_when_no_report_parsed(
        self, fake_crewai, litellm_env, registered_agents, monkeypatch
    ):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        signoff_crews: list[FakeCrew] = []

        async def fake_run(crew: FakeCrew) -> SimpleNamespace:
            names = [task.name for task in crew.tasks]
            if names == ["qa_executor/note"]:
                return self._output_for(crew, "not json at all")
            if names == ["review/verdict"]:
                signoff_crews.append(crew)
            return self._output_for(crew, "")

        monkeypatch.setattr(bridge, "_run_crew_async", fake_run)

        summary = await bridge._run_qa_staged_gate(_state())

        assert "did not parse" in signoff_crews[0].tasks[0].description
        assert summary.startswith("QaCrew finished: report=unparsed")

    async def test_zero_fix_rounds_runs_the_executor_once_then_gates(
        self, fake_crewai, litellm_env, registered_agents, monkeypatch
    ):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        monkeypatch.setattr(settings, "qa_max_fix_rounds", 0)
        calls: list[list[str]] = []

        async def fake_run(crew: FakeCrew) -> SimpleNamespace:
            calls.append([task.name for task in crew.tasks])
            return self._output_for(crew, _FAILING_REPORT)

        monkeypatch.setattr(bridge, "_run_crew_async", fake_run)
        state = _state()

        summary = await bridge._run_qa_staged_gate(state)

        assert calls == [["ba/note", "test_eng/artifact"], ["qa_executor/note"], ["review/verdict"]]
        assert state.approved is False
        assert any("exhausted after 0 round(s)" in err for err in state.errors)
        assert "fix_rounds=0" in summary

    @pytest.mark.parametrize(
        ("coverage", "approved"),
        [pytest.param(80.0, True, id="at_floor"), pytest.param(79.9, False, id="just_below_floor")],
    )
    async def test_coverage_floor_is_inclusive(
        self, fake_crewai, litellm_env, registered_agents, monkeypatch, coverage, approved
    ):
        bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
        monkeypatch.setattr(settings, "qa_min_coverage_percent", 80.0)
        report = json.dumps(
            {"criterion_results": [{"criterion_id": "ORDERS-01-AC-1", "passed": True}], "coverage_percent": coverage}
        )

        async def fake_run(crew: FakeCrew) -> SimpleNamespace:
            return self._output_for(crew, report)

        monkeypatch.setattr(bridge, "_run_crew_async", fake_run)
        state = _state()

        await bridge._run_qa_staged_gate(state)

        assert state.approved is approved
        assert any("below floor" in err for err in state.errors) is not approved


def test_implementation_crew_builds_tasks_from_state(fake_crewai, litellm_env, registered_agents):
    from pydantic_agents.core.models.core import UserStory

    crews = importlib.import_module("pydantic_agents.core.pipeline.crewai.crews")
    bridge = importlib.import_module("pydantic_agents.core.pipeline.crewai.bridge")
    assert bridge._CREW_MAP["ImplementationCrew"] is crews.ImplementationCrew

    state = PipelineState(
        run_id="r1",
        stage="implementation",
        requirement="Build health endpoint",
        user_stories=[UserStory(id="US-001", title="Health", gherkin="Given /healthz")],
        advisory_results={"cto": "use stdlib net/http"},
        feedback="Add timeout on server",
    )
    crew = crews.ImplementationCrew().build(state)
    assert [t.name for t in crew.tasks] == ["backend/artifact", "frontend/artifact", "test_eng/artifact"]
    assert "US-001" in crew.tasks[0].description
    assert "use stdlib net/http" in crew.tasks[0].description
    assert "Add timeout on server" in crew.tasks[0].description
