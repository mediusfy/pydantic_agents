import logging

from crewai import Crew, Process, Task

from pydantic_agents.core.adapters.crewai_bridge import as_crewai_agent
from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.registry import registry

logger = logging.getLogger(__name__)


def _make_task(description: str, agent, expected_output: str, context: list | None = None) -> Task:
    return Task(
        description=description,
        agent=agent,
        expected_output=expected_output,
        context=context or [],
    )


class SdlcCrew:
    def build(self, state: PipelineState) -> Crew:
        ba = as_crewai_agent(
            registry.get("ba"),
            role="Senior Business Analyst",
            goal="Refine raw requirements into structured user stories with Gherkin acceptance criteria",
        )
        cto = as_crewai_agent(
            registry.get("cto"),
            role="CTO",
            goal="Review architecture for scalability, security, and compliance",
        )
        backend = as_crewai_agent(
            registry.get("backend"),
            role="Lead Go Backend Engineer",
            goal="Implement Go services using hexagonal architecture",
        )
        frontend = as_crewai_agent(
            registry.get("frontend"),
            role="Senior Vue 3 Frontend Engineer",
            goal="Build Vue 3 components with WCAG 2.1 AA compliance",
        )
        qa = as_crewai_agent(
            registry.get("test_eng"),
            role="QA Automation Engineer",
            goal="Write comprehensive tests for all generated code",
        )
        reviewer = as_crewai_agent(
            registry.get("review"),
            role="Code Review Engineer",
            goal="Review all code for quality, security, and standards compliance",
        )

        refine_task = _make_task(
            f"Refine these requirements: {state.requirement}",
            ba,
            "List of structured user stories with Gherkin scenarios",
        )
        review_arch_task = _make_task(
            "Review the refined stories for architectural soundness",
            cto,
            "Architecture review with recommendations and bottlenecks",
            [refine_task],
        )
        impl_task = _make_task(
            "Implement Go backend for the user stories",
            backend,
            "Complete Go source files organized in hexagonal architecture",
            [refine_task, review_arch_task],
        )
        frontend_task = _make_task(
            "Build Vue 3 frontend components matching the backend API",
            frontend,
            "Vue 3 component files with TypeScript",
            [refine_task, impl_task],
        )
        test_task = _make_task(
            "Write unit and integration tests for the backend code",
            qa,
            "Test files with comprehensive coverage",
            [impl_task],
        )
        review_task = _make_task(
            "Review all generated code for quality, security, and compliance",
            reviewer,
            "Code review report with approval status and feedback",
            [impl_task, frontend_task, test_task],
        )

        return Crew(
            agents=[ba, cto, backend, frontend, qa, reviewer],
            tasks=[refine_task, review_arch_task, impl_task, frontend_task, test_task, review_task],
            process=Process.sequential,
            verbose=True,
        )


class ResearchCrew:
    def build(self, topic: str) -> Crew:
        weather = as_crewai_agent(
            registry.get("weather"),
            role="Weather Data Analyst",
            goal="Research weather patterns and conditions",
        )
        crypto = as_crewai_agent(
            registry.get("crypto_oracle"),
            role="Cryptocurrency Analyst",
            goal="Research cryptocurrency market trends",
        )
        market = as_crewai_agent(
            registry.get("market_shark"),
            role="Market Research Analyst",
            goal="Analyze market conditions and trends",
        )

        tasks = [
            _make_task(f"Research weather conditions related to: {topic}", weather, "Weather research report"),
            _make_task(f"Research cryptocurrency market for: {topic}", crypto, "Crypto market report"),
            _make_task(f"Research general market conditions for: {topic}", market, "Market analysis report"),
        ]

        return Crew(
            agents=[weather, crypto, market],
            tasks=tasks,
            process=Process.hierarchical,
            verbose=True,
        )


class SecurityReviewCrew:
    def build(self, code_context: str) -> Crew:
        security = as_crewai_agent(
            registry.get("security"),
            role="Principal Security Engineer",
            goal="Audit code for OWASP Top 10 vulnerabilities",
        )
        reviewer = as_crewai_agent(
            registry.get("review"),
            role="Code Review Engineer",
            goal="Review code for quality and security compliance",
        )
        sre = as_crewai_agent(
            registry.get("sre"),
            role="Site Reliability Engineer",
            goal="Review infrastructure and deployment configuration",
        )

        sec_task = _make_task(
            f"Audit this code for security vulnerabilities:\n{code_context}",
            security,
            "Security audit report with findings and remediation",
        )
        review_task = _make_task(
            f"Review this code for quality:\n{code_context}",
            reviewer,
            "Code review report",
            [sec_task],
        )
        infra_task = _make_task(
            "Review infrastructure requirements for the code",
            sre,
            "Infrastructure review report",
            [sec_task],
        )

        return Crew(
            agents=[security, reviewer, sre],
            tasks=[sec_task, review_task, infra_task],
            process=Process.sequential,
            verbose=True,
        )
