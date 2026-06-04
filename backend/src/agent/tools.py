"""外部工具定义：天气查询 & 数学计算。"""

import ast
import operator
from urllib.parse import quote

import requests
from langchain_core.tools import tool

_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_SAFE_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_math(node):
    """递归求值 AST 节点，仅允许数字和四则运算。"""
    if isinstance(node, ast.Expression):
        return _safe_eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_safe_eval_math(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        return _SAFE_BIN_OPS[type(node.op)](
            _safe_eval_math(node.left), _safe_eval_math(node.right)
        )
    raise ValueError("表达式包含不支持的语法")


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气信息，包括天气描述和温度。

    Args:
        city: 城市名称，支持中英文（如 Beijing、上海、Tokyo）。

    Returns:
        包含天气描述和温度的字符串。
    """
    try:
        url = f"https://wttr.in/{quote(city)}?format=%C+%t"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return f"{city} 当前天气：{resp.text.strip()}"
    except requests.RequestException as e:
        return f"查询天气失败：{e}"


@tool
def calculate(expression: str) -> str:
    """执行四则运算数学表达式计算。支持 +、-、*、/ 和括号。

    Args:
        expression: 数学表达式字符串，如 "2 + 3 * (4 - 1)"。

    Returns:
        计算结果字符串。
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval_math(tree)
        return str(result)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as e:
        return f"计算失败：{e}"
