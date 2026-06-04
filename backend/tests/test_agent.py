"""Agent 工具路由测试 — 验证 Agent 将查询正确路由到对应工具。

测试粒度：每个测试验证一个类型的问题得到正确的答案。
公共接口：agent.invoke({"input": query}) → {"output": "..."}

按 TDD 哲学，测试关注可观察的输出行为，不检查内部工具调用链。
工具是否正确路由通过输出内容来验证。
"""

import pytest


class TestAgentRouting:
    """Agent 工具路由 — 验证三类查询得到正确答案。"""

    def test_knowledge_query_returns_leave_policy(self, agent):
        """知识类问题 → 输出包含年假政策相关信息。

        Tracer bullet for Agent routing.
        """
        result = agent.invoke({"input": "公司年假怎么算？"})
        output = result["output"]

        assert len(output) > 20, f"输出过短: {output}"
        assert "年假" in output, f"应提及'年假': {output}"
        assert "5" in output, f"应包含起步天数 5: {output}"
        assert "15" in output, f"应包含上限 15: {output}"

    def test_weather_query_returns_city_weather(self, agent):
        """天气类问题 → 输出包含城市名和天气信息。"""
        result = agent.invoke({"input": "上海今天天气怎么样？"})
        output = result["output"]

        assert len(output) > 10, f"输出过短: {output}"
        assert "上海" in output, f"应包含城市名: {output}"

    def test_calculation_query_returns_correct_result(self, agent):
        """计算类问题 → 输出包含正确的计算结果。"""
        result = agent.invoke({"input": "帮我算一下 123 * 456 等于多少"})
        output = result["output"]

        assert len(output) > 5, f"输出过短: {output}"
        # 结果中应包含正确数字（允许千分位格式如 56,088）
        assert ("56088" in output or "56,088" in output or "56 088" in output), (
            f"应包含计算结果 56088: {output}"
        )
