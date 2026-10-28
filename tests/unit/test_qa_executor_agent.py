"""qa_executor (KAN-483): registered, can run suites, and reports through the code-derived TestRunReport verdict."""

from pydantic_agents.agents.qa_executor import qa_executor_agent
from pydantic_agents.agents.registration import register_all_agents
from pydantic_agents.core.models.core import TestRunReport
from pydantic_agents.core.registry import registry
from pydantic_agents.core.tool_assignments import AGENT_TOOL_CATEGORIES
from pydantic_agents.core.tools import test_tools  # noqa: F401  (registers execute_test_suite)
from pydantic_agents.core.tools.registry import tool_registry


def test_qa_executor_is_registered():
    register_all_agents()
    assert registry.get("qa_executor") is qa_executor_agent


def test_qa_executor_outputs_a_test_run_report():
    assert qa_executor_agent.output_type is TestRunReport


def test_qa_executor_gets_the_tool_its_prompt_names():
    categories = set(AGENT_TOOL_CATEGORIES["qa_executor"])
    assert set(tool_registry["execute_test_suite"].categories) & categories
