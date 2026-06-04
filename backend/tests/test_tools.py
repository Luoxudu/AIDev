"""基础设施 & 工具测试 — 模块导入、LLM 连接、计算器安全。

这些是快速冒烟测试，验证项目基础设施就绪。
"""

import pytest


class TestInfrastructure:
    """基础设施 — 导入和连接。"""

    def test_all_modules_import(self):
        """所有核心模块可正常导入。"""
        from src.core.settings import create_llm, RETRIEVER_K, HYDE_ENABLED
        from src.agent.tools import calculate, get_weather
        from src.agent import get_agent
        from src.retrieval.rag_chain import create_rag_chain

        assert create_llm is not None
        assert get_agent is not None
        assert create_rag_chain is not None
        assert calculate is not None
        assert get_weather is not None
        assert isinstance(RETRIEVER_K, int)
        assert isinstance(HYDE_ENABLED, bool)

    def test_llm_connection(self):
        """LLM API 可连通并返回有效响应。"""
        from src.core.settings import create_llm

        llm = create_llm()
        resp = llm.invoke("回复 OK")
        assert len(resp.content) > 0, "LLM 应返回非空响应"


class TestCalculatorSafety:
    """计算器安全 — AST 沙箱拒绝危险表达式。"""

    @pytest.mark.parametrize("expr", [
        "__import__('os')",
        "2 ** 100",
        "open('/etc/passwd')",
        "exec('print(1)')",
        "[i for i in range(1000000)]",
        "eval('1+1')",
    ])
    def test_rejects_dangerous_expression(self, expr, calculator):
        """危险表达式应被 AST 沙箱拒绝。"""
        result = calculator.invoke({"expression": expr})
        assert "失败" in result, (
            f"表达式 '{expr}' 应被拒绝，实际返回: {result}"
        )

    @pytest.mark.parametrize("expr, expected", [
        ("2 + 3", "5"),
        ("(100 + 200) * 3 / 5", "180.0"),
        ("10 - 3 * 2", "4"),
        ("-5 + 10", "5"),
        ("3.14 * 2", "6.28"),
    ])
    def test_evaluates_valid_expression(self, expr, expected, calculator):
        """合法数学表达式应正确计算。"""
        result = calculator.invoke({"expression": expr})
        assert expected in result, f"{expr} 应得到 {expected}, 实际: {result}"
