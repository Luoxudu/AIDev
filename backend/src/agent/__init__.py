"""Agent package — re-exports for backward compatibility."""

from src.agent.agent import create_agent, get_agent, reset_agent

__all__ = ["get_agent", "create_agent", "reset_agent", "agent_executor"]


def __getattr__(name):
    if name == "agent_executor":
        return get_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
