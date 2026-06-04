"""pytest fixtures for AIDev test suite.

All fixtures use public interfaces only — no internal mocking.
Tests describe behavior, not implementation.
"""

import sys
from pathlib import Path

import pytest

# 确保项目根目录在 import 路径中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="session")
def rag_chain():
    """RAG 检索链 — 会话级复用（构建一次，所有 RAG 测试共享）。"""
    from src.retrieval.rag_chain import create_rag_chain

    return create_rag_chain()


@pytest.fixture(scope="session")
def agent():
    """Agent 执行器 — 会话级复用。"""
    from src.agent import get_agent

    return get_agent()


@pytest.fixture(scope="session")
def calculator():
    """计算器工具。"""
    from src.agent.tools import calculate

    return calculate


@pytest.fixture(scope="session")
def weather_tool():
    """天气工具。"""
    from src.agent.tools import get_weather

    return get_weather
