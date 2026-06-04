"""AIDev - AI Agent 命令行入口。"""

import os
import sys
from datetime import datetime

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from src.core.settings import CHAT_HISTORY_DB, CHAT_HISTORY_WINDOW, MEMORY_ENABLED
from src.agent.memory import extract_and_update

_TOOL_LABELS = {
    "search_knowledge_base": "检索知识库",
    "get_weather": "查询天气",
    "calculate": "计算中",
}


class TypewriterCallback(BaseCallbackHandler):
    """逐 token 打印 LLM 输出，工具调用时显示状态提示。"""

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "")
        label = _TOOL_LABELS.get(tool_name, tool_name)
        sys.stdout.write(f"\n  [{label}...] ")
        sys.stdout.flush()

    def on_tool_end(self, output, **kwargs):
        sys.stdout.write("\n")
        sys.stdout.flush()

    def on_llm_new_token(self, token: str, **kwargs):
        chunk = kwargs.get("chunk")
        if chunk and hasattr(chunk, "message"):
            msg = chunk.message
            if getattr(msg, "tool_call_chunks", None):
                return
        if token:
            try:
                sys.stdout.write(token)
            except UnicodeEncodeError:
                # Windows GBK 控制台不支持部分 Unicode 字符（如 emoji）
                pass
            sys.stdout.flush()


def _get_chat_history(session_id: str) -> SQLChatMessageHistory:
    db_dir = os.path.dirname(CHAT_HISTORY_DB)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=f"sqlite:///{CHAT_HISTORY_DB}",
    )


def main():
    print("=" * 50)
    print("  AIDev — AI Agent (RAG + 工具)")
    print("  输入 'exit' / 'quit' 退出")
    print("=" * 50)

    # 懒加载 Agent
    from src.agent import get_agent
    agent_executor = get_agent()

    callback = TypewriterCallback()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = _get_chat_history(session_id)

    # 本地消息缓存，避免每轮重复查询 DB
    session_msgs = list(history.messages)
    max_msgs = CHAT_HISTORY_WINDOW * 2

    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in ("exit", "quit", "q"):
            break

        if not query:
            continue

        # 从本地缓存取滑动窗口（不包含当前 query，保证成对）
        chat_history = session_msgs[-max_msgs:]

        try:
            print("\n助手: ", end="", flush=True)
            result = agent_executor.invoke(
                {
                    "input": query,
                    "chat_history": chat_history,
                },
                config={"callbacks": [callback]},
            )
            print()

            output = result.get("output", "")
            # 仅在成功时持久化，保证 human/ai 成对
            history.add_user_message(query)
            session_msgs.append(HumanMessage(content=query))
            if output:
                history.add_ai_message(output)
                session_msgs.append(AIMessage(content=output))
        except Exception as e:
            print(f"\n[错误] {e}")

    # 会话结束：提取长期记忆
    if MEMORY_ENABLED and session_msgs:
        from src.agent.agent import reset_agent, _agent_llm
        if extract_and_update(session_msgs, llm=_agent_llm):
            reset_agent()

    print("再见！")


if __name__ == "__main__":
    main()
