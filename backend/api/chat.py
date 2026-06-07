"""SSE 流式聊天端点。"""

import asyncio
import json
import sqlite3

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from api.sessions import (
    _get_conn,
    get_chat_history_for_agent,
    get_next_branch_index,
    insert_message,
    touch_session,
    update_session_title,
)

router = APIRouter()

# 心跳间隔（秒）：工具执行期间无事件时，定期检查连接并发送心跳
_HEARTBEAT_INTERVAL = 3.0


class ChatRequest(BaseModel):
    session_id: str
    query: str
    parent_message_id: str | None = None

    @field_validator("parent_message_id")
    @classmethod
    def normalize_parent_id(cls, v: str | None) -> str | None:
        if v == "":
            return None
        return v


async def _stream_sse(
    conn: sqlite3.Connection,
    session_id: str,
    parent_id: str | None,
    human_msg_id: str,
    query: str,
    req: Request,
):
    """生成 SSE 事件流。check_branch_limit/human INSERT 已在外部完成。

    使用 asyncio.Queue 桥接 Agent 事件流，支持心跳机制：
    - 工具执行期间无事件时，每隔 _HEARTBEAT_INTERVAL 秒检查连接状态
    - 发送 SSE 注释行作为心跳，防止代理/浏览器超时断开
    - 检测到客户端断开时立即中断流
    """
    accumulated_content = ""
    tool_calls_log: list[dict] = []
    interrupted = False

    # 用于取消后台任务的 Event
    _stop_checker = asyncio.Event()

    async def _disconnect_checker():
        """后台任务：定期检查客户端是否断开。"""
        while not _stop_checker.is_set():
            try:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                if _stop_checker.is_set():
                    return
                if await req.is_disconnected():
                    _stop_checker.set()
                    return
            except asyncio.CancelledError:
                return

    try:
        chat_history = get_chat_history_for_agent(conn, parent_id, session_id)

        from src.agent.agent import get_web_agent
        from langchain_core.messages import HumanMessage

        agent = get_web_agent()
        messages = [*chat_history, HumanMessage(content=query)]

        # 将 astream_events 封装为 put 到队列的生产者
        event_queue: asyncio.Queue = asyncio.Queue()

        async def _event_producer():
            try:
                async for event in agent.astream_events(
                    {"messages": messages}, version="v2",
                ):
                    await event_queue.put(("event", event))
                await event_queue.put(("done", None))
            except asyncio.CancelledError:
                pass
            except Exception as e:
                await event_queue.put(("error", e))

        producer_task = asyncio.create_task(_event_producer())
        checker_task = asyncio.create_task(_disconnect_checker())

        try:
            while True:
                # 等待下一个事件或超时
                try:
                    msg_type, msg_data = await asyncio.wait_for(
                        event_queue.get(), timeout=_HEARTBEAT_INTERVAL
                    )
                except asyncio.TimeoutError:
                    # 超时：检查是否断开，发送心跳
                    if await req.is_disconnected():
                        interrupted = True
                        break
                    # SSE 注释行作为心跳（客户端忽略以 ':' 开头的行）
                    yield ": heartbeat\n\n"
                    continue

                if msg_type == "done":
                    break
                elif msg_type == "error":
                    interrupted = True
                    yield f"event: error\ndata: {json.dumps({'message': str(msg_data)}, ensure_ascii=False)}\n\n"
                    break

                event = msg_data

                # 在处理每个事件前检查断开
                if _stop_checker.is_set() or await req.is_disconnected():
                    interrupted = True
                    break

                kind = event.get("event")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is not None:
                        if getattr(chunk, "tool_call_chunks", None):
                            continue
                        text = chunk.content if isinstance(chunk.content, str) else ""
                        if text:
                            accumulated_content += text
                            yield f"event: token\ndata: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_start":
                    name = event.get("name", "")
                    inp = event.get("data", {}).get("input", {})
                    tool_calls_log.append({"name": name, "input": str(inp), "output": ""})
                    yield f"event: tool_start\ndata: {json.dumps({'name': name, 'input': str(inp)}, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_end":
                    name = event.get("name", "")
                    output = event.get("data", {}).get("output", "")
                    for tc in reversed(tool_calls_log):
                        if tc["name"] == name and tc["output"] == "":
                            tc["output"] = str(output)
                            break
                    yield f"event: tool_end\ndata: {json.dumps({'name': name, 'output': str(output)}, ensure_ascii=False)}\n\n"

        finally:
            # 清理后台任务
            _stop_checker.set()
            producer_task.cancel()
            checker_task.cancel()
            try:
                await asyncio.gather(producer_task, checker_task)
            except (asyncio.CancelledError, Exception):
                pass

        if not interrupted:
            yield "event: done\ndata: {}\n\n"

    except asyncio.CancelledError:
        interrupted = True
    except Exception as e:
        interrupted = True
        yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"
    finally:
        insert_message(
            conn, session_id, "ai", accumulated_content,
            parent_id=human_msg_id, branch_index=0,
            tool_calls=tool_calls_log if tool_calls_log else None,
            interrupted=interrupted,
        )
        touch_session(conn, session_id)
        conn.commit()
        conn.close()


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    """Check branch limit synchronously before sending SSE headers."""
    conn = _get_conn()
    try:
        parent_id = request.parent_message_id
        branch_index = get_next_branch_index(conn, parent_id, request.session_id)
        human_msg_id = insert_message(
            conn, request.session_id, "human", request.query,
            parent_id=parent_id, branch_index=branch_index,
        )
        update_session_title(conn, request.session_id, request.query)
        touch_session(conn, request.session_id)
        conn.commit()
    except Exception:
        conn.close()
        raise

    return StreamingResponse(
        _stream_sse(conn, request.session_id, parent_id, human_msg_id, request.query, req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
