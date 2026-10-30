"""CrewAI crews built from the registry's pydantic_ai agents.

Every crew has the same entry point, ``build(state: PipelineState) -> Crew``.
Each task carries a name of the form ``<owner>/<kind>`` so that
:func:`pydantic_agents.core.pipeline.crewai.bridge._apply_result` can map its
output back onto the pipeline state without guessing from agent roles:

- ``artifact``: the output is code; it becomes an ``Artifact`` for ``owner``.
- ``verdict``: the output is a ``ReviewResult`` JSON object (approval gate).
- ``advice``: the output is an ``AgentAdvice`` JSON object (CTO gate).
- ``feedback``: the output is the pipeline feedback text (audit findings).
- ``note``: the output is stored under ``advisory_results[owner]``.

This module imports ``crewai`` at import time, so it must only be imported
lazily, once the extra is known to be installed.
"""

import json
import logging
from typing import Any

from crewai import Crew, Process, Task

from pydantic_agents.core.adapters.crewai_bridge import as_crewai_agent, build_crewai_llm
from pydantic_agents.core.models.core import AgentAdvice, Finding, PipelineState, ReviewResult, TestRunReport
from pydantic_agents.core.registry import registry
from pydantic_agents.core.utils import format_artifacts

logger = logging.getLogger(__name__)

TASK_KINDS = ("artifact", "verdict", "advice", "feedback", "findings", "note")


def _task_name(owner: str, kind: str) -> str:
    if kind not in TASK_KINDS:
        raise ValueError(f"Unknown task kind '{kind}'; expected one of {TASK_KINDS}")
    return f"{owner}/{kind}"


def _json_contract(model: type[Any], preamble: str) -> str:
    schema = json.dumps(model.model_json_schema(), separators=(",", ":"))
    return (
        f"{preamble} Reply with exactly one JSON object and nothing else (no prose, no code fence) "
        f"that validates against this JSON schema: {schema}"
    )


_ROLE_BA = "Senior Business Analyst"
_ROLE_SECURITY = "Principal Security Engineer"
_ROLE_SRE = "Site Reliability Engineer"

REVIEW_VERDICT_OUTPUT = _json_contract(
    ReviewResult,
    "A review verdict. Set approved to false whenever anything must be fixed before merge.",
)
CTO_ADVICE_OUTPUT = _json_contract(
    AgentAdvice,
    "Architecture advice. approved_to_proceed must be false while any CRITICAL finding is open, or while adr_id "
    "is empty.",
)
QA_TEST_REPORT_OUTPUT = _json_contract(
    TestRunReport,
    "A test-execution report, one CriterionResult per acceptance-criterion id you ran.",
)


def _json_array_contract(item_model: type[Any], preamble: str) -> str:
    item_schema = json.dumps(item_model.model_json_schema(), separators=(",", ":"))
    return (
        f"{preamble} Reply with exactly one JSON array and nothing else (no prose, no code fence), each element "
        f"validating against this JSON schema: {item_schema}"
    )


AUDIT_FINDINGS_OUTPUT = _json_array_contract(
    Finding,
    "The consolidated findings, one array element per finding, with id, source_agent, severity, description, "
    "location set and owner/due_date filled in now that you are consolidating.",
)


def _make_task(
    description: str,
    agent: Any,
    expected_output: str,
    context: list[Any] | None = None,
    *,
    name: str,
) -> Task:
    return Task(
        name=name,
        description=description,
        agent=agent,
        expected_output=expected_output,
        context=context or [],
    )


def _agent(name: str, role: str, goal: str, *, allow_delegation: bool = False) -> Any:
    return as_crewai_agent(registry.get(name), role=role, goal=goal, allow_delegation=allow_delegation)


def _manager_llm() -> Any:
    """The LLM that runs a hierarchical crew: the CTO's model."""
    return build_crewai_llm(registry.get("cto"))


def _code_context(state: PipelineState) -> str:
    """The requirement plus every artifact produced so far, for review-style crews."""
    parts = [f"Requirement:\n{state.requirement}"] if state.requirement else []
    if state.artifacts:
        parts.append(f"Code under review:\n{format_artifacts(state.artifacts)}")
    return "\n\n".join(parts) or "(no requirement or code was supplied)"


class SdlcCrew:
    def build(self, state: PipelineState) -> Crew:
        ba = _agent(
            "ba",
            _ROLE_BA,
            "Refine raw requirements into structured user stories with Gherkin acceptance criteria",
        )
        cto = _agent("cto", "CTO", "Review architecture for scalability, security, and compliance")
        backend = _agent("backend", "Lead Go Backend Engineer", "Implement Go services using hexagonal architecture")
        frontend = _agent(
            "frontend", "Senior Vue 3 Frontend Engineer", "Build Vue 3 components with WCAG 2.1 AA compliance"
        )
        qa = _agent("test_eng", "QA Automation Engineer", "Write comprehensive tests for all generated code")
        reviewer = _agent(
            "review", "Code Review Engineer", "Review all code for quality, security, and standards compliance"
        )

        refine_task = _make_task(
            f"Refine these requirements: {state.requirement}",
            ba,
            "List of structured user stories with Gherkin scenarios",
            name=_task_name("ba", "note"),
        )
        review_arch_task = _make_task(
            "Review the refined stories for architectural soundness",
            cto,
            "Architecture review with recommendations and bottlenecks",
            [refine_task],
            name=_task_name("cto", "note"),
        )
        impl_task = _make_task(
            "Implement Go backend for the user stories",
            backend,
            "Complete Go source files organized in hexagonal architecture, each in a fenced code block "
            "preceded by its file path",
            [refine_task, review_arch_task],
            name=_task_name("backend", "artifact"),
        )
        frontend_task = _make_task(
            "Build Vue 3 frontend components matching the backend API",
            frontend,
            "Vue 3 component files with TypeScript, each in a fenced code block preceded by its file path",
            [refine_task, impl_task],
            name=_task_name("frontend", "artifact"),
        )
        test_task = _make_task(
            "Write unit and integration tests for the backend code",
            qa,
            "Test files with comprehensive coverage, each in a fenced code block preceded by its file path",
            [impl_task],
            name=_task_name("test_eng", "artifact"),
        )
        review_task = _make_task(
            "Review all generated code for quality, security, and compliance",
            reviewer,
            REVIEW_VERDICT_OUTPUT,
            [impl_task, frontend_task, test_task],
            name=_task_name("review", "verdict"),
        )

        return Crew(
            agents=[ba, cto, backend, frontend, qa, reviewer],
            tasks=[refine_task, review_arch_task, impl_task, frontend_task, test_task, review_task],
            process=Process.sequential,
            verbose=True,
        )


class ImplementationCrew:
    """Implementation-only crew for the ``composite`` blueprint.

    LangGraph runs BA, the CTO advisory gate, and the post-implementation quality gate; this crew executes the
    code- and test-generation step in between using the stories, advisory notes, and any retry feedback already on
    ``PipelineState``.
    """

    def build(self, state: PipelineState) -> Crew:
        backend = _agent("backend", "Lead Go Backend Engineer", "Implement Go services using hexagonal architecture")
        frontend = _agent(
            "frontend", "Senior Vue 3 Frontend Engineer", "Build Vue 3 components with WCAG 2.1 AA compliance"
        )
        qa = _agent("test_eng", "QA Automation Engineer", "Write comprehensive tests for all generated code")

        stories_text = "\n\n".join(f"### {s.id}: {s.title}\n{s.gherkin}" for s in (state.user_stories or []))
        if not stories_text:
            stories_text = state.requirement or "(no stories supplied)"
        advisory_text = "\n".join(f"- {k}: {v}" for k, v in (state.advisory_results or {}).items())
        context_block = f"User stories:\n{stories_text}"
        if advisory_text:
            context_block += f"\n\nArchitecture guidance:\n{advisory_text}"
        if state.feedback:
            context_block += f"\n\nPrevious attempt feedback to fix:\n{state.feedback}"

        impl_task = _make_task(
            f"Implement Go backend for the user stories.\n\n{context_block}",
            backend,
            "Complete Go source files organized in hexagonal architecture, each in a fenced code block "
            "preceded by its file path",
            name=_task_name("backend", "artifact"),
        )
        frontend_task = _make_task(
            f"Build Vue 3 frontend components matching the backend API.\n\n{context_block}",
            frontend,
            "Vue 3 component files with TypeScript, each in a fenced code block preceded by its file path",
            [impl_task],
            name=_task_name("frontend", "artifact"),
        )
        test_task = _make_task(
            f"Write unit and integration tests for the backend code.\n\n{context_block}",
            qa,
            "Test files with comprehensive coverage, each in a fenced code block preceded by its file path",
            [impl_task],
            name=_task_name("test_eng", "artifact"),
        )

        return Crew(
            agents=[backend, frontend, qa],
            tasks=[impl_task, frontend_task, test_task],
            process=Process.sequential,
            verbose=True,
        )


class ResearchCrew:
    def build(self, state: PipelineState) -> Crew:
        topic = state.requirement
        weather = _agent("weather", "Weather Data Analyst", "Research weather patterns and conditions")
        crypto = _agent("crypto_oracle", "Cryptocurrency Analyst", "Research cryptocurrency market trends")
        market = _agent("market_shark", "Market Research Analyst", "Analyze market conditions and trends")

        tasks = [
            _make_task(
                f"Research weather conditions related to: {topic}",
                weather,
                "Weather research report",
                name=_task_name("weather", "note"),
            ),
            _make_task(
                f"Research cryptocurrency market for: {topic}",
                crypto,
                "Crypto market report",
                name=_task_name("crypto_oracle", "note"),
            ),
            _make_task(
                f"Research general market conditions for: {topic}",
                market,
                "Market analysis report",
                name=_task_name("market_shark", "note"),
            ),
        ]

        return Crew(
            agents=[weather, crypto, market],
            tasks=tasks,
            process=Process.hierarchical,
            manager_llm=_manager_llm(),
            verbose=True,
        )


class SecurityReviewCrew:
    def build(self, state: PipelineState) -> Crew:
        code_context = _code_context(state)
        security = _agent("security", _ROLE_SECURITY, "Audit code for OWASP Top 10 vulnerabilities")
        reviewer = _agent("review", "Code Review Engineer", "Review code for quality and security compliance")
        sre = _agent("sre", _ROLE_SRE, "Review infrastructure and deployment configuration")

        sec_task = _make_task(
            f"Audit this code for security vulnerabilities:\n{code_context}",
            security,
            "Security audit report with findings and remediation",
            name=_task_name("security", "note"),
        )
        infra_task = _make_task(
            "Review infrastructure requirements for the code",
            sre,
            "Infrastructure review report",
            [sec_task],
            name=_task_name("sre", "note"),
        )
        review_task = _make_task(
            f"Review this code for quality, taking the security audit and infrastructure review into account:\n"
            f"{code_context}",
            reviewer,
            REVIEW_VERDICT_OUTPUT,
            [sec_task, infra_task],
            name=_task_name("review", "verdict"),
        )

        return Crew(
            agents=[security, reviewer, sre],
            tasks=[sec_task, infra_task, review_task],
            process=Process.sequential,
            verbose=True,
        )


_CRITERION_ID_RULE = (
    "Acceptance-criterion ids have the form <story id>-AC-<n> (for example ORDERS-01-AC-2). "
    "Every item in your output must name the criterion id(s) it covers; an item without a criterion id is invalid."
)


class QaCrew:
    """Acceptance criteria → tests → execute → bounded fix loop → sign-off, keyed by criterion id.

    CrewAI's ``Process.sequential`` runs each task exactly once, so the bounded retry (KAN-487) can't live
    inside one ``Crew``: each stage below is its own single- or two-task crew, chained by the orchestrator in
    ``bridge._run_qa_staged_gate``, which re-runs ``build_fix``/``build_execution`` up to
    ``settings.qa_max_fix_rounds`` times before ``build_signoff`` runs regardless.
    """

    def build_tests(self, state: PipelineState) -> Crew:
        """Stage 1: user stories with acceptance-criterion ids, then the tests that prove each one."""
        ba = _agent(
            "ba",
            _ROLE_BA,
            "Turn requirements into user stories with ids and numbered Gherkin acceptance criteria",
        )
        test_gen = _agent(
            "test_gen",
            "Test Generation Engineer",
            "Write Go table-driven tests and Playwright specs for every acceptance criterion",
        )
        criteria_task = _make_task(
            f"Write user stories for this requirement:\n{state.requirement}\n\n"
            "Give every story an id of the form BC-NN and number its Gherkin acceptance criteria. "
            f"{_CRITERION_ID_RULE}",
            ba,
            "User stories, each with its story id and a numbered list of acceptance criteria whose ids follow "
            "<story id>-AC-<n>",
            name=_task_name("ba", "note"),
        )
        tests_task = _make_task(
            "For every acceptance criterion from the previous task write the tests that prove it: Go table-driven "
            "tests (one table row per criterion, the row name is the criterion id) and Playwright specs (one "
            "`test()` per criterion, titled with the criterion id). Emit each file in a fenced code block preceded "
            f"by its file path. {_CRITERION_ID_RULE}",
            test_gen,
            "Go `_test.go` files and Playwright `.spec.ts` files, each in a fenced code block preceded by its "
            "file path, where every test case names the criterion id it covers",
            [criteria_task],
            name=_task_name("test_eng", "artifact"),
        )
        return Crew(
            agents=[ba, test_gen],
            tasks=[criteria_task, tests_task],
            process=Process.sequential,
            verbose=True,
        )

    def build_execution(self, tests_output: str) -> Crew:
        """Stage 2 (and re-run after each fix round): qa_executor runs the current tests and reports results."""
        qa_executor = _agent(
            "qa_executor",
            "QA Executor",
            "Run the generated test suites and report pass/fail per acceptance-criterion id",
        )
        execution_task = _make_task(
            f"Run every test suite below and report the result. {_CRITERION_ID_RULE}\n\n{tests_output}",
            qa_executor,
            QA_TEST_REPORT_OUTPUT,
            name=_task_name("qa_executor", "note"),
        )
        return Crew(
            agents=[qa_executor],
            tasks=[execution_task],
            process=Process.sequential,
            verbose=True,
        )

    def build_fix(self, tests_output: str, report: TestRunReport) -> Crew:
        """One bounded fix round: go_test only touches the criteria the last report marked failing.

        ``bridge._run_qa_staged_gate`` only calls this once a parsed ``TestRunReport`` says the gate isn't
        met yet, so the all-unnamed fallback below fires only for a parsed report with zero criterion results
        (an executor that ran nothing, reported no failures and no unexecuted ids) — not for a report that
        failed to parse at all, which short-circuits straight to the gate instead of reaching this method.
        """
        go_test = _agent("go_test", "Go Test Engineer", "Make failing Go tests pass without weakening them")
        failing_ids = [r.criterion_id for r in report.criterion_results if not r.passed]
        failing_ids.extend(report.unexecuted_criterion_ids)
        fix_task = _make_task(
            "Run the generated Go tests and fix only the tests covering these failing/unexecuted criterion ids: "
            f"{', '.join(failing_ids) or '(none named; re-check all)'}. Do not add, remove or rename test cases, "
            f"and keep every criterion id. {_CRITERION_ID_RULE}\n\nCurrent tests:\n{tests_output}",
            go_test,
            "The corrected test files in fenced code blocks preceded by their file paths, plus the list of "
            "criterion ids whose tests were fixed",
            name=_task_name("test_eng", "artifact"),
        )
        return Crew(
            agents=[go_test],
            tasks=[fix_task],
            process=Process.sequential,
            verbose=True,
        )

    def build_signoff(self, tests_output: str, report: TestRunReport | None) -> Crew:
        """Final stage: review signs off, but the fix-loop's own TestRunReport is the fail-closed gate.

        ``report`` is None when qa_executor's output never parsed; the reviewer is told so rather than shown an
        empty report, and the bridge blocks approval in code either way.
        """
        reviewer = _agent("review", "QA Reviewer", "Sign off test coverage against the acceptance criteria")
        report_text = report.model_dump_json() if report is not None else "(none: qa_executor's output did not parse)"
        signoff_task = _make_task(
            "Check that every acceptance criterion has at least one passing test that references its id. "
            "Approve only when no criterion is left uncovered. Every feedback entry must start with the "
            f"criterion id it concerns. {_CRITERION_ID_RULE}\n\n"
            f"Tests:\n{tests_output}\n\nLatest test-execution report: {report_text}",
            reviewer,
            REVIEW_VERDICT_OUTPUT,
            name=_task_name("review", "verdict"),
        )
        return Crew(
            agents=[reviewer],
            tasks=[signoff_task],
            process=Process.sequential,
            verbose=True,
        )


class ArchitectureReviewCrew:
    """Domain, threat, data and operations reviews merged by the CTO into an ``AgentAdvice`` verdict."""

    def build(self, state: PipelineState) -> Crew:
        context = _code_context(state)
        ba = _agent("ba", _ROLE_BA, "Map bounded contexts and the domain events between them")
        security = _agent("security", _ROLE_SECURITY, "Produce a threat model for the design")
        database = _agent("database", "Database Engineer", "Assess schema design and migration risk")
        sre = _agent("sre", _ROLE_SRE, "Assess deployment topology and capacity")
        cto = _agent("cto", "CTO", "Merge the specialist reviews into one architecture decision")

        contexts_task = _make_task(
            f"Identify the bounded contexts and the domain events exchanged between them for:\n{context}\n\n"
            "Name every event past-tense and namespaced (orders.OrderPlaced) with its owner and consumers.",
            ba,
            "Bounded contexts with their responsibilities and a table of domain events (name, owner, consumers, "
            "payload sketch)",
            name=_task_name("ba", "note"),
        )
        threat_task = _make_task(
            f"Build a threat model for:\n{context}\n\n"
            "Cover trust boundaries, data flows, authentication/authorization, secrets and the OWASP Top 10. "
            "Rate every finding LOW, MEDIUM, HIGH or CRITICAL.",
            security,
            "Threat model: trust boundaries, assets, threats with STRIDE category, severity and mitigation",
            name=_task_name("security", "note"),
        )
        schema_task = _make_task(
            f"Assess the schema and migration risk for:\n{context}\n\n"
            "Cover data ownership per bounded context, migrations that lock or rewrite large tables, "
            "backwards compatibility and rollback. Rate every risk LOW, MEDIUM, HIGH or CRITICAL.",
            database,
            "Schema review: entities per context, migration plan risks with severity and mitigation",
            name=_task_name("database", "note"),
        )
        deploy_task = _make_task(
            f"Assess deployment and capacity for:\n{context}\n\n"
            "Cover topology, resource requests and limits, scaling, failure domains, observability and cost. "
            "Rate every risk LOW, MEDIUM, HIGH or CRITICAL.",
            sre,
            "Deployment review: topology, capacity estimate, risks with severity and mitigation",
            name=_task_name("sre", "note"),
        )
        decision_task = _make_task(
            "Merge the bounded-context map, threat model, schema review and deployment review into one "
            "architecture decision. List bottlenecks, recommendations and the constraints that bind each "
            "downstream agent. Record this decision under an adr_id (e.g. the next free ADR number, such as "
            "'0002'); approved_to_proceed must be false while any CRITICAL finding from any review is still open, "
            "or while adr_id is empty.",
            cto,
            CTO_ADVICE_OUTPUT,
            [contexts_task, threat_task, schema_task, deploy_task],
            name=_task_name("cto", "advice"),
        )

        return Crew(
            agents=[ba, security, database, sre, cto],
            tasks=[contexts_task, threat_task, schema_task, deploy_task, decision_task],
            process=Process.hierarchical,
            manager_llm=_manager_llm(),
            verbose=True,
        )


_FINDING_LOCATION_RULE = (
    "Every finding must point at a location: a `file:line` reference or a resource id (Kubernetes "
    "kind/namespace/name, image digest, workflow job id). Findings without a location are dropped."
)


class AuditCrew:
    """Independent security, CI/container and runtime audits consolidated into one findings table."""

    def build(self, state: PipelineState) -> Crew:
        context = _code_context(state)
        security = _agent("security", _ROLE_SECURITY, "Find leaked secrets and OWASP Top 10 issues")
        devops = _agent("devops", "DevOps Engineer", "Harden CI pipelines and container images")
        sre = _agent("sre", _ROLE_SRE, "Audit network policies, resource limits and alerting")
        reviewer = _agent("review", "Audit Lead", "Consolidate audit findings into an owned, dated action table")

        secrets_task = _make_task(
            f"Scan for committed secrets and OWASP Top 10 vulnerabilities in:\n{context}\n\n{_FINDING_LOCATION_RULE}",
            security,
            "Security findings, each with severity, location (file:line), description and remediation",
            name=_task_name("security", "note"),
        )
        ci_task = _make_task(
            f"Audit the CI pipelines and container images in:\n{context}\n\n"
            "Cover unpinned actions and base images, missing SBOM/signing, root containers, writable root "
            f"filesystems and secrets in build args or layers. {_FINDING_LOCATION_RULE}",
            devops,
            "CI and container hardening findings, each with severity, location (file:line or image digest), "
            "description and remediation",
            name=_task_name("devops", "note"),
        )
        runtime_task = _make_task(
            f"Audit the runtime configuration in:\n{context}\n\n"
            "Cover network policies (default deny, egress), resource requests and limits, pod security, "
            f"probes and alert coverage. {_FINDING_LOCATION_RULE}",
            sre,
            "Runtime findings, each with severity, resource id (kind/namespace/name), description and remediation",
            name=_task_name("sre", "note"),
        )
        consolidate_task = _make_task(
            "Consolidate the security, CI/container and runtime findings into one list. Drop every finding that "
            "has neither a file:line reference nor a resource id. Owners are the responsible agent roles "
            "(security, devops, sre, backend, frontend); due dates are ISO dates relative to today, sooner for "
            "higher severity. Every finding you keep must have both an owner and a due date set.",
            reviewer,
            AUDIT_FINDINGS_OUTPUT,
            [secrets_task, ci_task, runtime_task],
            name=_task_name("review", "findings"),
        )

        return Crew(
            agents=[security, devops, sre, reviewer],
            tasks=[secrets_task, ci_task, runtime_task, consolidate_task],
            process=Process.sequential,
            verbose=True,
        )
