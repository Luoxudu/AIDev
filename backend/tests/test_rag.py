"""RAG 检索链路测试 — 验证知识库查询行为的正确性。

测试粒度：每个测试验证一个可观察的行为，通过公共接口 `rag_chain.invoke()`。
"""

import pytest


class TestRAGBasicRetrieval:
    """RAG 基础检索 — 知识库能返回相关答案。"""

    def test_retrieves_answer_for_work_hours_query(self, rag_chain):
        """查询考勤时间 → 返回包含时间信息的答案。

        这是 tracer bullet — 验证 RAG 链路端到端能跑通。
        """
        result = rag_chain.invoke({"input": "公司考勤时间是几点？"})
        answer = result["answer"]

        # 行为断言：答案应包含时间相关信息
        assert len(answer) > 20, f"答案过短: {answer}"
        # 弹性打卡 8:30-9:30 或固定时间 9:00 都是正确答案
        has_morning = any(t in answer for t in ["8:30", "8：30", "9:00", "9：00"])
        assert has_morning, f"应包含上班时间: {answer}"
        has_evening = any(t in answer for t in ["6:00", "6：00", "18:00"])
        assert has_evening, f"应包含下班时间: {answer}"

    def test_retrieves_correct_framework_for_platform_query(self, rag_chain):
        """查询 StarBrain 平台后端框架 → 返回 FastAPI。"""
        result = rag_chain.invoke({"input": "StarBrain 平台用了什么后端框架？"})
        answer = result["answer"]

        assert len(answer) > 10, f"答案过短: {answer}"
        assert "FastAPI" in answer, f"应包含框架名 FastAPI: {answer}"

    def test_retrieves_correct_annual_leave_policy(self, rag_chain):
        """查询年假政策 → 返回 5天起、上限15天。"""
        result = rag_chain.invoke({"input": "员工有多少天年假？"})
        answer = result["answer"]

        assert len(answer) > 10, f"答案过短: {answer}"
        assert "5" in answer, f"应包含年假起步天数 5: {answer}"
        assert "15" in answer, f"应包含年假上限 15: {answer}"

    def test_retrieves_departments_list(self, rag_chain):
        """查询公司部门 → 返回部门名称列表。"""
        result = rag_chain.invoke({"input": "公司有哪些部门？"})
        answer = result["answer"]

        assert len(answer) > 10, f"答案过短: {answer}"
        # 至少应包含一个已知部门
        departments = ["产品研发", "数据科学", "解决方案", "市场", "运营管理"]
        found = any(dept in answer for dept in departments)
        assert found, f"应包含至少一个部门名称: {answer}"
