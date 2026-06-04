"""Agent 构建：CLI 用旧 API + Web 用 langgraph create_react_agent。"""

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from src.core.settings import MEMORY_ENABLED, create_llm
from src.agent.memory import load_profile, load_raw_profile, format_profile_for_prompt
from src.retrieval.rag_chain import create_rag_chain
from src.agent.tools import calculate, get_weather

# ── 全局 RAG 链（懒加载）──
_rag_chain = None


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = create_rag_chain()
    return _rag_chain


@tool
def search_knowledge_base(query: str) -> str:
    """从公司内部知识库中检索并回答关于文档内容的问题。

    当用户提出的问题可能与公司文档、内部知识、产品说明相关时，
    应优先使用此工具进行检索。

    Args:
        query: 要查询的问题。

    Returns:
        基于知识库文档内容的回答。如果文档中没有相关信息，会明确告知。
    """
    rag_chain = _get_rag_chain()
    result = rag_chain.invoke({"input": query})
    return result["answer"]


def _global_tool_prompt() -> str:
    return (
        "你是一个智能助手，具备以下能力：\n"
        "1. 使用 search_knowledge_base 工具检索公司内部知识库中的文档内容，"
        "回答用户关于文档的问题。\n"
        "2. 使用 get_weather 工具查询天气，仅在用户明确询问天气信息时调用。\n"
        "3. 使用 calculate 工具进行数学计算，仅在需要精确计算时调用。\n\n"
        "工具调用决策原则（非常重要）：\n"
        "- 如果用户发送的是问候语、闲聊、意义不明或并非疑问句的内容，"
        "不要调用任何工具，直接以自然对话方式回答。\n"
        "- 只有用户明确在询问知识、查资料、问事实性问题时，才调用 search_knowledge_base。\n"
        "- 如果知识库返回「无法回答」，不要编造，如实告知用户\n"
        "- 只在与天气或计算明确相关时才使用对应工具"
    )


# ══════════════════════════════════════════════════════════════
#  CLI Agent（旧 API，保留不动）
# ══════════════════════════════════════════════════════════════

def create_agent() -> AgentExecutor:
    llm = create_llm(streaming=True)
    global _agent_llm
    _agent_llm = llm

    tools = [search_knowledge_base, get_weather, calculate]
    agent_system_prompt = _global_tool_prompt()

    if MEMORY_ENABLED:
        profile = load_profile()
        agent_system_prompt += format_profile_for_prompt(profile)

    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=False,
    )


_agent_executor = None
_agent_llm = None


def get_agent() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = create_agent()
    return _agent_executor


def reset_agent():
    global _agent_executor
    _agent_executor = None


# ══════════════════════════════════════════════════════════════
#  Web Agent（langgraph create_react_agent）
# ══════════════════════════════════════════════════════════════

_web_agent = None


def build_system_prompt() -> str:
    """组装两层系统提示词（全局工具说明 + 用户自定义）。"""
    prompt = _global_tool_prompt()

    profile = load_raw_profile()
    user_prompt = profile.get("system_prompt", "")
    if user_prompt:
        prompt += f"\n\n─── 以下为用户自定义指令 ───\n{user_prompt}"

    if MEMORY_ENABLED:
        facts = load_profile()
        prompt += format_profile_for_prompt(facts)

    return prompt


def _create_web_agent():
    from langgraph.prebuilt import create_react_agent

    llm = create_llm(streaming=True)
    tools = [search_knowledge_base, get_weather, calculate]
    system_content = build_system_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_content,
    )


def get_web_agent():
    global _web_agent
    if _web_agent is None:
        _web_agent = _create_web_agent()
    return _web_agent


def reset_web_agent():
    global _web_agent
    _web_agent = None


def reset_rag_chain():
    global _rag_chain
    _rag_chain = None


def __getattr__(name):
    if name == "agent_executor":
        return get_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
