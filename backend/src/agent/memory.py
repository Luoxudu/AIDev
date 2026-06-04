"""长期记忆模块：JSON 文件存储用户画像，会话结束时 LLM 自动提取。"""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.core.settings import MEMORY_AUTO_EXTRACT, MEMORY_PROFILE_PATH

_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一个信息提取助手。根据对话记录和现有用户画像，输出完整的更新后画像。\n"
        "规则：\n"
        "- 保留仍正确的事实，添加新事实，删除矛盾的旧事实\n"
        "- 只提取用户个人信息、偏好、角色、工作背景\n"
        "- 不要提取临时性问题或知识库查询内容\n"
        "- 如果没有新信息，原样返回现有画像\n\n"
        "请严格按以下 JSON 格式输出，不要输出其他内容：\n"
        '[\n  {{"category": "identity|preference|work|other", "content": "具体内容"}}\n]'
    )),
    ("human", "现有画像：\n{existing}\n\n对话记录：\n{conversation}"),
])

_SIGNAL_KEYWORDS = [
    "我是", "我叫", "我的名字", "我负责", "我是做", "我的职位", "我的角色",
    "我偏好", "我喜欢", "我不喜欢", "我习惯", "我希望你",
    "我在", "我来自", "我的部门", "我的团队",
    "记住", "帮我记", "以后", "每次都",
]


def _content_str(msg) -> str:
    """安全提取消息文本内容，兼容多模态 list 格式。"""
    content = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def load_profile() -> list[dict]:
    """加载用户画像，返回事实列表。"""
    facts = load_raw_profile().get("facts", [])
    return facts if isinstance(facts, list) else []


def load_raw_profile() -> dict:
    """加载 profile.json 完整内容（含 system_prompt、facts 等）。"""
    path = Path(MEMORY_PROFILE_PATH)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"facts": data}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass
    return {}


def save_profile(facts: list[dict]) -> None:
    """保存用户画像到 JSON 文件，保留已有字段（如 system_prompt）。"""
    existing = load_raw_profile()
    existing["facts"] = facts
    save_raw_profile(existing)


def save_raw_profile(data: dict) -> None:
    """写入 profile.json 完整内容。"""
    path = Path(MEMORY_PROFILE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def format_profile_for_prompt(facts: list[dict]) -> str:
    """将画像格式化为可注入 prompt 的文本。"""
    if not facts:
        return ""
    lines = ["\n\n## 用户画像"]
    by_category: dict[str, list[str]] = {}
    for fact in facts:
        cat = fact.get("category", "other")
        by_category.setdefault(cat, []).append(fact["content"])
    labels = {"identity": "身份", "preference": "偏好", "work": "工作", "other": "其他"}
    for cat, items in by_category.items():
        label = labels.get(cat, cat)
        lines.append(f"- {label}：" + "；".join(items))
    return "\n".join(lines)


def _has_memory_signal(messages: list) -> bool:
    """检查对话中是否包含值得提取记忆的信号。"""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = _content_str(msg).lower()
            if any(kw in text for kw in _SIGNAL_KEYWORDS):
                return True
    return False


def _format_conversation(messages: list) -> str:
    """将消息列表格式化为文本。"""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"用户: {_content_str(msg)}")
        elif isinstance(msg, AIMessage):
            text = _content_str(msg)
            if len(text) > 300:
                text = text[:300] + "..."
            lines.append(f"助手: {text}")
    return "\n".join(lines)


def _format_existing(facts: list[dict]) -> str:
    """将现有画像格式化为提取提示的输入。"""
    if not facts:
        return "（无）"
    return "\n".join(f"- [{f.get('category', 'other')}] {f['content']}" for f in facts)


def _parse_extraction(raw: str) -> list[dict]:
    """解析 LLM 返回的 JSON 提取结果。"""
    text = raw.strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        items = json.loads(text[start:end])
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict) and "content" in i]
    except json.JSONDecodeError:
        pass
    return []


def extract_and_update(messages: list, llm=None) -> list[dict] | None:
    """从对话中提取记忆并更新 profile.json。

    LLM 接收现有画像 + 对话记录，直接输出完整的更新后画像列表，
    由 LLM 负责语义去重和合并，避免本地字符串匹配的局限性。

    Args:
        messages: 本轮会话的消息列表。
        llm: 可选的 LLM 实例，未提供时通过 create_llm() 创建。

    Returns:
        更新后的事实列表；如果未更新返回 None。
    """
    if not MEMORY_AUTO_EXTRACT:
        return None
    if not _has_memory_signal(messages):
        return None

    existing = load_profile()
    if llm is None:
        from src.core.settings import create_llm
        llm = create_llm()
    chain = _EXTRACT_PROMPT | llm
    conversation = _format_conversation(messages)
    existing_text = _format_existing(existing)

    try:
        response = chain.invoke({
            "conversation": conversation,
            "existing": existing_text,
        })
        updated = _parse_extraction(str(response.content))
    except Exception as e:
        print(f"  [Memory] 记忆提取失败: {e}")
        return None

    if not updated:
        return None

    save_profile(updated)
    new_count = len(updated) - len(existing)
    print(f"  [Memory] 已更新画像（{new_count:+d} 条，共 {len(updated)} 条）")
    return updated
