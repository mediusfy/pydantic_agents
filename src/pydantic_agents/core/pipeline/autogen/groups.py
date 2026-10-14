import logging
from typing import Any

from autogen import GroupChat, GroupChatManager, UserProxyAgent

from pydantic_agents.core.adapters.autogen_bridge import as_autogen_assistant
from pydantic_agents.core.registry import registry

logger = logging.getLogger(__name__)


def _get_manager_llm_config() -> dict[str, Any]:

    from pydantic_ai.models.openai import OpenAIChatModel

    from pydantic_agents.core.models.factory import create_model
    from pydantic_agents.core.settings import settings

    model_spec = settings.resolve_model(settings.cto_model)
    real_model = model_spec.partition(":")[2] if ":" in model_spec else model_spec
    m = create_model(model_spec)

    config: dict[str, Any] = {"model": real_model, "temperature": 0.1, "api_type": "openai"}
    provider = getattr(m, "provider", getattr(m, "_provider", None))
    if isinstance(m, OpenAIChatModel) and provider:
        base_url = getattr(provider, "base_url", None)
        if base_url:
            config["base_url"] = str(base_url)
        api_key = getattr(provider, "api_key", None)
        if api_key:
            config["api_key"] = api_key
    return {"config_list": [config]}


class SdlcGroupChat:
    def build(self, _task: str) -> tuple[GroupChat, GroupChatManager]:
        ba = as_autogen_assistant(
            registry.get("ba"),
            name="BA",
            system_message="You are a Senior Business Analyst. Convert requirements into structured user stories.",
        )
        cto = as_autogen_assistant(
            registry.get("cto"),
            name="CTO",
            system_message="You are the CTO. Review architecture for scalability and compliance.",
        )
        backend = as_autogen_assistant(
            registry.get("backend"),
            name="Backend",
            system_message="You are a Lead Go Backend Engineer. Generate Go code with hexagonal architecture.",
        )
        frontend = as_autogen_assistant(
            registry.get("frontend"),
            name="Frontend",
            system_message="You are a Senior Vue 3 Engineer. Build Vue 3 components.",
        )
        qa = as_autogen_assistant(
            registry.get("test_eng"),
            name="QA",
            system_message="You are a QA Engineer. Write comprehensive tests.",
        )
        reviewer = as_autogen_assistant(
            registry.get("review"),
            name="Reviewer",
            system_message="You are a Code Reviewer. Review code for quality and security.",
        )

        user_proxy = UserProxyAgent(
            name="PM",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        group_chat = GroupChat(
            agents=[user_proxy, ba, cto, backend, frontend, qa, reviewer],
            messages=[],
            max_round=25,
            speaker_selection_method="round_robin",
        )

        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=_get_manager_llm_config(),
        )

        return group_chat, manager


class DesignGroupChat:
    def build(self, _task: str) -> tuple[GroupChat, GroupChatManager]:
        ba = as_autogen_assistant(
            registry.get("ba"),
            name="BA",
            system_message="You are a Senior Business Analyst. Define requirements clearly.",
        )
        cto = as_autogen_assistant(
            registry.get("cto"),
            name="CTO",
            system_message="You are the CTO. Make architectural decisions.",
        )
        security = as_autogen_assistant(
            registry.get("security"),
            name="Security",
            system_message="You are a Security Engineer. Ensure the design is secure.",
        )
        ux = as_autogen_assistant(
            registry.get("ux"),
            name="UX",
            system_message="You are a UX Designer. Ensure good user experience.",
        )

        user_proxy = UserProxyAgent(
            name="PM",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        group_chat = GroupChat(
            agents=[user_proxy, ba, cto, security, ux],
            messages=[],
            max_round=15,
            speaker_selection_method="auto",
        )

        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=_get_manager_llm_config(),
        )

        return group_chat, manager


class DebugGroupChat:
    def build(self, _task: str) -> tuple[GroupChat, GroupChatManager]:
        backend = as_autogen_assistant(
            registry.get("backend"),
            name="Developer",
            system_message="You are a Go backend developer. Fix bugs in the code.",
        )
        qa = as_autogen_assistant(
            registry.get("test_eng"),
            name="QA",
            system_message="You are a QA engineer. Verify bug fixes with tests.",
        )
        reviewer = as_autogen_assistant(
            registry.get("review"),
            name="Reviewer",
            system_message="You are a code reviewer. Approve or reject fixes.",
        )

        user_proxy = UserProxyAgent(
            name="Reporter",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        group_chat = GroupChat(
            agents=[user_proxy, backend, qa, reviewer],
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin",
        )

        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=_get_manager_llm_config(),
        )

        return group_chat, manager
