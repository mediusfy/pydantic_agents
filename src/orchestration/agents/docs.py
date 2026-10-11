from pydantic_ai import Agent

from orchestration.models.core import Artifact
from orchestration.models.factory import create_model
from orchestration.settings import settings

docs_agent = Agent(
    create_model(settings.resolve_model(settings.docs_model)),
    system_prompt="""You are a Technical Writer and API Documentation Specialist.
Your objective is to generate accurate OpenAPI/Swagger specifications, comprehensive README files, and detailed API reference docs from the provided technical specifications and code artifacts.
Return Artifact with files dict mapping relative paths to file contents.
""",
    output_type=Artifact,
    defer_model_check=True,
    retries=5,
)
