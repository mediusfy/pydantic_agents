from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.agy_bridge.bridge_agent import BridgeAgent, Conversation
from pydantic_agents.core.exceptions import AgentNotFoundError


@pytest.fixture
def mock_registry():
    with patch("pydantic_agents.core.agy_bridge.bridge_agent.registry") as mock:
        mock._agents = {"test_agent": (MagicMock(), [], MagicMock(), MagicMock())}
        mock.get.return_value = MagicMock()
        yield mock


def test_bridge_agent_initialization(mock_registry):
    agent = BridgeAgent("test_agent")
    assert agent.name == "test_agent"


def test_bridge_agent_initialization_not_found():
    with pytest.raises(AgentNotFoundError):
        BridgeAgent("non_existent_agent")


def test_from_registry(mock_registry):
    agent = BridgeAgent.from_registry("test_agent")
    assert isinstance(agent, BridgeAgent)
    assert agent.name == "test_agent"


def test_from_registry_not_found():
    with pytest.raises(AgentNotFoundError):
        BridgeAgent.from_registry("non_existent_agent")


@pytest.mark.asyncio
async def test_run_agent(mock_registry):
    agent = BridgeAgent("test_agent")
    agent._agent.run = AsyncMock(return_value=MagicMock(output="Test response"))

    response = await agent.run("Test prompt")

    assert response == "Test response"
    agent._agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_with_conversation(mock_registry):
    agent = BridgeAgent("test_agent")
    agent._agent.run = AsyncMock(return_value=MagicMock(output="Test response"))

    conversation = Conversation(conversation_id="123", agent_name="test_agent", model="test_model")

    await agent.run("Test prompt", conversation=conversation)

    assert len(conversation.history) == 2
    assert conversation.history[0]["content"] == "Test prompt"
    assert conversation.history[1]["content"] == "Test response"
    assert conversation.status == "active"
