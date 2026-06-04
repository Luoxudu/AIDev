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
    """生成 SSE 事件流。check_branch_limit/human INSERT 已在外部完成。"""
    accumulated_content = ""
    tool_calls_log: list[dict] = []
    interrupted = False

    try:
        chat_history = get_chat_history_for_agent(conn, parent_id, session_id)

        from src.agent.agent import get_web_agent
        from langchain_core.messages import HumanMessage

        agent = get_web_agent()
        messages = [*chat_history, HumanMessage(content=query)]

        async for event in agent.astream_events(
            {"messages": messages}, version="v2",
        ):
            if await req.is_disconnected():
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
