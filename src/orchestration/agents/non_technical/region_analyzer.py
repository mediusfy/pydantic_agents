from dataclasses import dataclass

from pydantic_ai import Agent

from orchestration.models.factory import create_model
from orchestration.settings import settings


@dataclass
class RegionAnalyzerDeps:
    agent_name: str = "region_analyzer"


analyzer_prompt = """You are a Region Analyzer Agent.
Your job is to analyze geographical regions based on various factors like safety, cost of living, and local attractions.
"""

region_analyzer_agent = Agent(
    create_model(settings.resolve_model(settings.region_analyzer_model)),
    deps_type=RegionAnalyzerDeps,
    system_prompt=analyzer_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
