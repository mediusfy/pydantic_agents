#!/usr/bin/env python3
from pydantic_ai import Agent, RunContext, UsageLimits

from orchestration.models.core import EventTrigger, UserStory
from orchestration.models.factory import create_model
from orchestration.settings import settings

ba_prompt = """You are a Senior Business Analyst Agent focused on **Domain-Driven Design (DDD)**.

Your job is to convert raw requirements into:
1. **User Stories** in the standard "As a... I want... so that..." form, each with Gherkin acceptance criteria (Given/When/Then).
2. **Bounded Contexts** — name them, define the ubiquitous language inside each, and identify the integration patterns between them (shared kernel, customer-supplier, anti-corruption layer, etc.).
3. **Event Triggers** — the domain events that downstream backend services and QA automation will key off of. For each event give: name, payload schema sketch, the bounded context that owns it, and the consumers.

## Conventions

- Story IDs follow `BC-NN` (e.g., `ORDERS-01`)
- Event names are past-tense and namespaced: `orders.OrderPlaced`, `billing.PaymentCaptured`
- Explicitly capture and document multi-cloud constraints, emphasizing data residency laws and regional network latency
- Strictly utilize Gherkin syntax (Given/When/Then) to frame all acceptance criteria

## Quality Standards

- Be ruthless about ambiguity — if requirements conflict or a story has no measurable acceptance criteria, flag it
- Every user story must map to at least one event trigger
- Events must have clear owners (bounded context) and consumers (other contexts/services)
"""

ba_agent = Agent(
    create_model(settings.resolve_model(settings.ba_model)),
    system_prompt=ba_prompt,
    output_type=list[UserStory],
    defer_model_check=True,
    retries=5,
)

extractor = Agent(
    create_model(settings.resolve_model(settings.ba_model)),
    system_prompt=ba_prompt,
    output_type=list[UserStory],
    defer_model_check=True,
    retries=5,
)
event_mapper = Agent(
    create_model(settings.resolve_model(settings.ba_model)),
    system_prompt=ba_prompt,
    output_type=list[EventTrigger],
    defer_model_check=True,
    retries=5,
)


@ba_agent.tool
async def define_user_stories(ctx: RunContext[str], raw_requirements: str) -> list[UserStory]:
    """Converts raw text requirements into structured UserStory models using an LLM."""
    res = await extractor.run(
        f"Extract user stories from: {raw_requirements}", usage_limits=UsageLimits(request_limit=500)
    )
    return res.output


@ba_agent.tool
async def map_event_triggers(ctx: RunContext[str], story_id: str, gherkin: str) -> list[EventTrigger]:
    """Identifies specific events necessary for the Go backend and QA automation pipelines."""
    res = await event_mapper.run(
        f"Identify event triggers for story {story_id} with gherkin: {gherkin}",
        usage_limits=UsageLimits(request_limit=500),
    )
    return res.output
