from unittest.mock import AsyncMock

import pytest

from pydantic_agents.core.agy_bridge.bridge_agent import HookResult, Hooks


@pytest.mark.asyncio
async def test_pre_turn_hook():
    hooks = Hooks()
    mock_hook = AsyncMock(return_value=HookResult(allow=True, modified_input="Modified"))

    @hooks.on_pre_turn
    async def pre_turn_hook(prompt: str):
        return await mock_hook(prompt)

    results = [await hook("Original") for hook in hooks.pre_turn]

    assert len(results) == 1
    assert results[0].modified_input == "Modified"
    mock_hook.assert_called_once_with("Original")


@pytest.mark.asyncio
async def test_post_turn_hook():
    hooks = Hooks()
    mock_hook = AsyncMock()

    @hooks.on_post_turn
    async def post_turn_hook(output: str):
        await mock_hook(output)

    for hook in hooks.post_turn:
        await hook("Test output")

    mock_hook.assert_called_once_with("Test output")
