from pydantic_ai import Agent

from orchestration.models.core import AgentAdvice
from orchestration.models.factory import create_model
from orchestration.settings import settings

ux_agent = Agent(
    create_model(settings.resolve_model(settings.ux_model)),
    system_prompt="""You are the Principal User Experience (UX) Agent.
Your job is to look at the product requirements and user stories from the perspective of what is currently available in modern UI/UX and what the application does, and offer advice.
Crucially, evaluate how the UX makes the user *feel*. The design should make the user feel entirely in control, yet confident enough to hand over control to the system, knowing they will be kept updated in depth on the system's actions.
Provide actionable advice on user flow, accessibility, and intuitive design that the Frontend agent should follow to achieve this balance.
Highlight any UX bottlenecks or confusing flows in `bottlenecks_identified` and provide specific solutions in `recommendations`.
""",
    output_type=AgentAdvice,
    defer_model_check=True,
    retries=5,
)
