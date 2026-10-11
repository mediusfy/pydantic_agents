from dataclasses import dataclass

from pydantic_ai import Agent

from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class AsymmetricReviewerDeps:
    agent_name: str = "asymmetric_reviewer"


reviewer_prompt = """You are an Asymmetric Reviewer Agent.
    You are an expert in law and corporate tricks. Your goal is to help save money when traveling, when done correctly,
    by finding loopholes and exploiting asymmetric information.
"""

asymmetric_reviewer_agent = Agent(
    create_model(settings.resolve_model(settings.asymmetric_reviewer_model)),
    deps_type=AsymmetricReviewerDeps,
    system_prompt=reviewer_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
