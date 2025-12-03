import pytest
from agentforge import Agent, tools

def test_agent_init():
    agent = Agent()
    assert agent is not None

def test_tool_registration():
    t = tools.http()
    agent = Agent(tools=[t])
    assert "http_fetch" in agent.tools



    
