"""AIDev 系统测试套件。"""

import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_imports():
    """测试 1: 模块导入"""
    print("\n[Test 1] 模块导入...")
    try:
        from src.core.settings import create_llm, RETRIEVER_K, MULTI_QUERY_COUNT, HYDE_ENABLED
        from src.agent.tools import calculate, get_weather
        from src.agent import get_agent
        from src.retrieval.rag_chain import create_rag_chain
        print("  [PASS] 所有模块导入成功")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_llm_connection():
    """测试 2: LLM 连接"""
    print("\n[Test 2] LLM 连接...")
    try:
        from src.core.settings import create_llm
        llm = create_llm()
        resp = llm.invoke("回复 OK")
        print(f"  [PASS] LLM 响应: {resp.content[:80]}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_calculator():
    """测试 3: 计算器工具"""
    print("\n[Test 3] 计算器工具...")
    from src.agent.tools import calculate
    tests = [
        ("2 + 3", "5"),
        ("(100 + 200) * 3 / 5", "180.0"),
        ("10 - 3 * 2", "4"),
        ("-5 + 10", "5"),
        ("3.14 * 2", "6.28"),
    ]
    all_pass = True
    for expr, expected in tests:
        result = calculate.invoke({"expression": expr})
        ok = expected in result
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {expr} = {result}")
    return all_pass


def test_calculator_safety():
    """测试 4: 计算器安全（拒绝危险表达式）"""
    print("\n[Test 4] 计算器安全验证...")
    from src.agent.tools import calculate
    dangerous = [
        "__import__('os')",
        "2 ** 100",
        "open('/etc/passwd')",
        "exec('print(1)')",
        "[i for i in range(1000000)]",
    ]
    all_pass = True
    for expr in dangerous:
        result = calculate.invoke({"expression": expr})
        ok = "失败" in result
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] 拒绝: {expr} -> {result[:60]}")
    return all_pass


def test_weather_tool_unit():
    """测试 5: 天气工具（单元）"""
    print("\n[Test 5] 天气工具（单元测试）...")
    from src.agent.tools import get_weather

    # 测试 URL 编码
    result = get_weather.invoke({"city": "北京"})
    ok = "北京" in result and ("天气" in result or "°" in result or "fail" not in result.lower())
    status = "PASS" if ok else "WARN"
    print(f"  [{status}] 北京天气: {result[:80]}")
    return ok


def test_rag_chain_direct():
    """测试 6: RAG 链直接检索"""
    print("\n[Test 6] RAG 检索链（直接调用）...")
    from src.retrieval.rag_chain import create_rag_chain

    rag = create_rag_chain()
    queries = [
        "公司考勤时间是几点？",
        "StarBrain 平台用了什么后端框架？",
        "员工有多少天年假？",
        "公司有哪些部门？",
    ]
    all_pass = True
    for q in queries:
        result = rag.invoke({"input": q})
        answer = result.get("answer", "")
        ok = len(answer) > 10 and "无法" not in answer
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] Q: {q}")
        print(f"         A: {answer[:120]}...")
    return all_pass


def test_agent_knowledge():
    """测试 7: Agent 知识库路由"""
    print("\n[Test 7] Agent 知识库路由...")
    from src.agent import get_agent

    agent = get_agent()
    queries = [
        ("公司年假怎么算？", "年假"),
        ("技术架构中用了什么数据库？", "数据库"),
    ]
    all_pass = True
    for q, keyword in queries:
        result = agent.invoke({"input": q})
        output = result.get("output", "")
        ok = keyword in output and len(output) > 20
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] Q: {q}")
        print(f"         A: {output[:150]}...")
    return all_pass


def test_agent_weather():
    """测试 8: Agent 天气路由"""
    print("\n[Test 8] Agent 天气工具路由...")
    from src.agent import get_agent

    agent = get_agent()
    result = agent.invoke({"input": "上海今天天气怎么样？"})
    output = result.get("output", "")
    ok = ("上海" in output) and ("°" in output or "温度" in output or "天气" in output)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Q: 上海今天天气怎么样？")
    print(f"         A: {output[:150]}...")
    return ok


def test_agent_calculator():
    """测试 9: Agent 计算器路由"""
    print("\n[Test 9] Agent 计算器工具路由...")
    from src.agent import get_agent

    agent = get_agent()
    result = agent.invoke({"input": "帮我算一下 123 * 456 等于多少"})
    output = result.get("output", "")
    ok = "56088" in output or "56,088" in output or "56 088" in output
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Q: 帮我算一下 123 * 456 等于多少")
    print(f"         A: {output[:150]}...")
    return ok


def main():
    print("=" * 60)
    print("  AIDev 系统测试套件")
    print("=" * 60)

    results = {}
    tests = [
        ("模块导入", test_imports),
        ("LLM 连接", test_llm_connection),
        ("计算器", test_calculator),
        ("计算器安全", test_calculator_safety),
        ("天气工具（单元）", test_weather_tool_unit),
        ("RAG 检索链（直接）", test_rag_chain_direct),
        ("Agent 知识库路由", test_agent_knowledge),
        ("Agent 天气路由", test_agent_weather),
        ("Agent 计算器路由", test_agent_calculator),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            results[name] = False

    # 汇总
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'[PASS]' if ok else '[FAIL]'}  {name}")
    print(f"\n  通过: {passed}/{total}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
