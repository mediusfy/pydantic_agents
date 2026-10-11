from pydantic_ai import Agent, RunContext, UsageLimits

from orchestration.models.core import Artifact
from orchestration.models.factory import create_model
from orchestration.settings import settings

frontend_prompt = """You are a Lead Frontend Engineer building Vue 3 interfaces using the Composition API and Single File Components (<script setup>). You must ensure every interactive element is strictly **WCAG 2.1 compliant** and includes a unique data-testid attribute to facilitate flawless automated simulation by testing agents.

Technical Priorities:
- Atomic Design: Enforce component modularity and reusability across the Orgitration Agent SDK.
- State Management: Architect the UI to function as a seamless, real-time reflection of the backend event bus using Vue's reactivity system.
- Event Handling: Implement robust WebSocket or Server-Sent Events (SSE) logic, including automatic reconnection strategies and state reconciliation.

## Output Protocol
Do NOT attempt to read or write files to disk. You do not have local I/O access. Return all generated code as structured output through this response. Your response will be captured as an Artifact and written to the filesystem by the pipeline orchestrator.

## Self-Validation
You have access to the `validate_frontend_code` tool. You MUST pass your generated component through this tool. Do NOT return the final artifact to the orchestrator until the tool returns SUCCESS. If it returns an error, analyze the trace, correct your code, and run the tool again.

## Reference Example (Few-Shot)

Use `<script setup lang="ts">`, defineProps, and standard composition API exactly like this.

```vue
<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  title: string
}>()

const emit = defineEmits<{
  (e: 'submit', value: string): void
}>()

const inputValue = ref('')

const handleSubmit = () => {
  if (inputValue.value) {
    emit('submit', inputValue.value)
  }
}
</script>

<template>
  <div class="component-wrapper">
    <h2>{{ props.title }}</h2>
    <input v-model="inputValue" type="text" />
    <button @click="handleSubmit">Submit</button>
  </div>
</template>
```
"""

frontend_agent = Agent(
    create_model(settings.resolve_model(settings.frontend_model)),
    system_prompt=frontend_prompt,
    output_type=Artifact,
    defer_model_check=True,
    retries=5,
)


# Text-first agent: reasoner generates code as free text (no result_type constraint)
component_gen = Agent(
    create_model(settings.resolve_model(settings.frontend_model)),
    system_prompt=frontend_prompt,
    defer_model_check=True,
    retries=5,
)


@frontend_agent.tool
async def create_component(ctx: RunContext[str], component_name: str, requirements: str, test_id: str) -> str:
    """Calls model to generate a production-ready Vue 3 SFC component with: WCAG 2.1 ARIA labels, data-testid attribute, TypeScript props interface, Tailwind/CSS module styling, proper event handler typing."""
    # reasoner generates code as free text
    raw = await component_gen.run(
        f"Generate Vue 3 SFC component {component_name} using Composition API with requirements: {requirements}. Use data-testid={test_id}.",
        usage_limits=UsageLimits(request_limit=500),
        model_settings={"max_tokens": 16384},
    )
    # extract text — no output_type means .output is a str
    generated_code = raw.output

    import re

    match = re.search(r"```(?:[a-zA-Z0-9]+)?\n(.*?)```", generated_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_code.strip()
