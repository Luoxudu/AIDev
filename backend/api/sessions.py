"""会话 & 消息 CRUD — SQLite + 树结构。"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.settings import CHAT_HISTORY_WEB_DB, CHAT_HISTORY_WINDOW

# ── DB 路径 ──
_DB_PATH = CHAT_HISTORY_WEB_DB

router = APIRouter()


# ══════════════════════════════════════════════════════════════
#  DB 初始化
# ══════════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    import os
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = _get_conn()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id            TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            parent_id     TEXT,
            branch_index  INTEGER NOT NULL DEFAULT 0,
            role          TEXT NOT NULL CHECK(role IN ('human','ai')),
            content       TEXT NOT NULL,
            tool_calls    TEXT,
            interrupted   INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    conn.close()


# ══════════════════════════════════════════════════════════════
#  Pydantic Models
# ══════════════════════════════════════════════════════════════

class SessionCreate(BaseModel):
    pass  # title 自动生成


class SessionRename(BaseModel):
    title: str


class MessageNode(BaseModel):
    human: dict
    ai: dict | None = None
    children: list["MessageNode"] = []


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("tool_calls"):
        try:
            d["tool_calls"] = json.loads(d["tool_calls"])
        except (json.JSONDecodeError, ValueError):
            d["tool_calls"] = []
    else:
        d["tool_calls"] = []
    d["interrupted"] = bool(d.get("interrupted", 0))
    return d


# ══════════════════════════════════════════════════════════════
#  会话 CRUD
# ══════════════════════════════════════════════════════════════

@router.get("/sessions")
def list_sessions():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/sessions", status_code=201)
def create_session(_: Optional[SessionCreate] = None):
    sid = _uuid()
    now = _now()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, "新会话", now, now),
    )
    conn.commit()
    conn.close()
    return {"id": sid, "title": "新会话", "created_at": now, "updated_at": now}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "会话不存在")
    return dict(row)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@router.patch("/sessions/{session_id}")
def rename_session(session_id: str, body: SessionRename):
    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (body.title, _now(), session_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


# ══════════════════════════════════════════════════════════════
#  消息操作
# ══════════════════════════════════════════════════════════════

def insert_message(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    parent_id: str | None = None,
    branch_index: int = 0,
    tool_calls: list | None = None,
    interrupted: bool = False,
) -> str:
    msg_id = _uuid()
    now = _now()
    conn.execute(
        "INSERT INTO messages (id, session_id, parent_id, branch_index, role, content, tool_calls, interrupted, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            msg_id,
            session_id,
            parent_id,
            branch_index,
            role,
            content,
            json.dumps(tool_calls, ensure_ascii=False) if tool_calls is not None else None,
            1 if interrupted else 0,
            now,
        ),
    )
    return msg_id


def get_next_branch_index(conn: sqlite3.Connection, parent_id: str | None, session_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(branch_index), -1) AS max_bi, "
        "COUNT(DISTINCT branch_index) AS cnt "
        "FROM messages WHERE parent_id IS ? AND session_id = ?",
        (parent_id, session_id),
    ).fetchone()
    if row["cnt"] >= 3:
        raise HTTPException(400, detail={
            "error": "branch_limit_reached",
            "message": "该节点下最多支持 3 个分支",
        })
    return row["max_bi"] + 1


def update_session_title(conn: sqlite3.Connection, session_id: str, content: str):
    row = conn.execute(
        "SELECT title FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row and row["title"] == "新会话":
        title = content[:30] + ("..." if len(content) > 30 else "")
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )


def touch_session(conn: sqlite3.Connection, session_id: str):
    conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
    )


def build_branch_path(conn: sqlite3.Connection, leaf_ai_id: str | None, session_id: str) -> list[dict]:
    """从 leaf ai 消息向上回溯到根，返回路径上所有消息（按 created_at 升序）。"""
    if leaf_ai_id is None:
        return []
    ids = set()
    current_id = leaf_ai_id
    while current_id is not None:
        if current_id in ids:
            break
        ids.add(current_id)
        row = conn.execute(
            "SELECT parent_id FROM messages WHERE id = ? AND session_id = ?",
            (current_id, session_id),
        ).fetchone()
        if not row or row["parent_id"] is None:
            break
        current_id = row["parent_id"]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM messages WHERE id IN ({placeholders}) AND session_id = ? ORDER BY created_at ASC",
        [*list(ids), session_id],
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_chat_history_for_agent(conn: sqlite3.Connection, leaf_ai_id: str | None, session_id: str) -> list:
    """组装 LangChain 消息列表：branch_path + 滑动窗口。"""
    from langchain_core.messages import AIMessage, HumanMessage

    path_msgs = build_branch_path(conn, leaf_ai_id, session_id)
    max_pairs = CHAT_HISTORY_WINDOW
    pairs: list = []
    human_buf = None
    for msg in path_msgs:
        if msg["role"] == "human":
            human_buf = HumanMessage(content=msg["content"])
        elif msg["role"] == "ai" and human_buf is not None:
            pairs.append((human_buf, AIMessage(content=msg["content"])))
            human_buf = None
    pairs = pairs[-max_pairs:]
    result = []
    for h, a in pairs:
        result.append(h)
        result.append(a)
    return result


# ══════════════════════════════════════════════════════════════
#  树结构
# ══════════════════════════════════════════════════════════════

@router.get("/sessions/{session_id}/tree")
def get_tree(session_id: str):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY branch_index, created_at",
        (session_id,),
    ).fetchall()
    conn.close()

    children_of: dict[str | None, list] = {}
    for r in rows:
        d = _row_to_dict(r)
        pid = d["parent_id"]
        children_of.setdefault(pid, []).append(d)

    def build_node(human_msg: dict) -> dict | None:
        ai_candidates = children_of.get(human_msg["id"], [])
        ai_msg = ai_candidates[0] if ai_candidates else None
        child_humans = children_of.get(ai_msg["id"], []) if ai_msg else []
        children = [build_node(ch) for ch in child_humans if ch["role"] == "human"]
        return {
            "human": {
                "id": human_msg["id"],
                "content": human_msg["content"],
                "branch_index": human_msg["branch_index"],
                "created_at": human_msg["created_at"],
            },
            "ai": {
                "id": ai_msg["id"],
                "content": ai_msg["content"],
                "tool_calls": ai_msg.get("tool_calls", []),
                "interrupted": ai_msg.get("interrupted", False),
                "created_at": ai_msg["created_at"],
            } if ai_msg else None,
            "children": children,
        }

    root_humans = [m for m in children_of.get(None, []) if m["role"] == "human"]
    nodes = [build_node(h) for h in root_humans]
    return {"session_id": session_id, "nodes": nodes}


# ══════════════════════════════════════════════════════════════
#  获取当前分支的消息列表（用于前端对话区域渲染）
# ══════════════════════════════════════════════════════════════

@router.get("/sessions/{session_id}/messages")
def get_branch_messages(session_id: str, leaf_ai_id: str | None = None):
    """获取从根到 leaf_ai_id 的分支路径上的所有消息。"""
    conn = _get_conn()
    msgs = build_branch_path(conn, leaf_ai_id, session_id)
    conn.close()
    return {"messages": msgs}


# ══════════════════════════════════════════════════════════════
#  删除节点（human+ai 对）
# ══════════════════════════════════════════════════════════════

@router.delete("/messages/{human_msg_id}")
def delete_message(human_msg_id: str, session_id: str | None = None):
    conn = _get_conn()
    try:
        human_row = conn.execute(
            "SELECT session_id, parent_id FROM messages WHERE id = ? AND role = 'human'",
            (human_msg_id,),
        ).fetchone()
        if not human_row:
            raise HTTPException(404, "消息不存在")
        if session_id is not None and human_row["session_id"] != session_id:
            raise HTTPException(403, "消息不属于指定会话")
        actual_session_id = human_row["session_id"]
        grandparent_id = human_row["parent_id"]

        ai_row = conn.execute(
            "SELECT id FROM messages WHERE parent_id = ? AND role = 'ai'",
            (human_msg_id,),
        ).fetchone()
        if not ai_row:
            raise HTTPException(404, "消息不存在或无配对 AI 回复")
        ai_id = ai_row["id"]

        children = conn.execute(
            "SELECT id, branch_index FROM messages WHERE parent_id = ? AND role = 'human'",
            (ai_id,),
        ).fetchall()

        if children:
            existing_branches = conn.execute(
                "SELECT branch_index FROM messages WHERE parent_id IS ? AND role = 'human' AND id != ?",
                (grandparent_id, human_msg_id),
            ).fetchall()
            existing_indices = {r["branch_index"] for r in existing_branches}

            for child in sorted(children, key=lambda r: r["branch_index"]):
                bi = 0
                while bi in existing_indices:
                    bi += 1
                conn.execute(
                    "UPDATE messages SET parent_id = ?, branch_index = ? WHERE id = ?",
                    (grandparent_id, bi, child["id"]),
                )
                existing_indices.add(bi)

        conn.execute("DELETE FROM messages WHERE id = ?", (ai_id,))
        conn.execute("DELETE FROM messages WHERE id = ?", (human_msg_id,))

        remaining = conn.execute(
            "SELECT id FROM messages WHERE parent_id IS ? AND role = 'human' ORDER BY branch_index",
            (grandparent_id,),
        ).fetchall()
        for i, row in enumerate(remaining):
            conn.execute(
                "UPDATE messages SET branch_index = ? WHERE id = ?",
                (i, row["id"]),
            )

        touch_session(conn, actual_session_id)
        conn.commit()
    except HTTPException:
        raise
    except Exception:
        raise
    finally:
        conn.close()
    return {"ok": True}


# ══════════════════════════════════════════════════════════════
#  复制节点到其他会话
# ══════════════════════════════════════════════════════════════

class CopyNodeRequest(BaseModel):
    source_human_id: str
    target_human_id: str
    target_session_id: str
    mode: str  # "above" | "below" | "replace"


@router.post("/messages/copy")
def copy_node(body: CopyNodeRequest):
    if body.mode not in ("above", "below", "replace"):
        raise HTTPException(400, "mode 必须为 above / below / replace")

    conn = _get_conn()
    try:
        # ── 查找源节点 ──
        src_human = conn.execute(
            "SELECT * FROM messages WHERE id = ? AND role = 'human'",
            (body.source_human_id,),
        ).fetchone()
        if not src_human:
            raise HTTPException(404, "源消息不存在")
        src_session_id = src_human["session_id"]
        if src_session_id == body.target_session_id:
            raise HTTPException(400, "禁止复制到自身会话")

        src_ai = conn.execute(
            "SELECT * FROM messages WHERE parent_id = ? AND role = 'ai' AND session_id = ?",
            (body.source_human_id, src_session_id),
        ).fetchone()
        if not src_ai:
            raise HTTPException(404, "源消息缺少 AI 回复")

        # ── 查找目标节点 ──
        tgt_human = conn.execute(
            "SELECT * FROM messages WHERE id = ? AND role = 'human' AND session_id = ?",
            (body.target_human_id, body.target_session_id),
        ).fetchone()
        if not tgt_human:
            raise HTTPException(404, "目标消息不存在")

        tgt_ai = conn.execute(
            "SELECT * FROM messages WHERE parent_id = ? AND role = 'ai' AND session_id = ?",
            (body.target_human_id, body.target_session_id),
        ).fetchone()

        src_tool_calls = json.loads(src_ai["tool_calls"]) if src_ai["tool_calls"] else None
        new_human_id: str | None = None

        if body.mode == "below":
            if not tgt_ai:
                raise HTTPException(400, "目标节点缺少 AI 回复，无法插入到下方")
            new_parent_id = tgt_ai["id"]
            new_branch = get_next_branch_index(conn, new_parent_id, body.target_session_id)

            new_human_id = insert_message(
                conn, body.target_session_id, "human",
                src_human["content"], parent_id=new_parent_id, branch_index=new_branch,
            )
            insert_message(
                conn, body.target_session_id, "ai",
                src_ai["content"], parent_id=new_human_id, branch_index=0,
                tool_calls=src_tool_calls,
                interrupted=bool(src_ai["interrupted"]),
            )

        elif body.mode == "above":
            new_parent_id = tgt_human["parent_id"]
            new_branch = get_next_branch_index(conn, new_parent_id, body.target_session_id)

            new_human_id = insert_message(
                conn, body.target_session_id, "human",
                src_human["content"], parent_id=new_parent_id, branch_index=new_branch,
            )
            new_ai_id = insert_message(
                conn, body.target_session_id, "ai",
                src_ai["content"], parent_id=new_human_id, branch_index=0,
                tool_calls=src_tool_calls,
                interrupted=bool(src_ai["interrupted"]),
            )
            conn.execute(
                "UPDATE messages SET parent_id = ?, branch_index = 0 WHERE id = ?",
                (new_ai_id, tgt_human["id"]),
            )
            if new_parent_id is not None:
                _renumber_siblings(conn, new_parent_id, body.target_session_id)

        elif body.mode == "replace":
            if not tgt_ai:
                raise HTTPException(400, "目标节点缺少 AI 回复，无法替换")
            tgt_parent_id = tgt_human["parent_id"]
            tgt_branch_index = tgt_human["branch_index"]

            children = conn.execute(
                "SELECT id, branch_index FROM messages WHERE parent_id = ? AND role = 'human' AND session_id = ?",
                (tgt_ai["id"], body.target_session_id),
            ).fetchall()

            new_human_id = insert_message(
                conn, body.target_session_id, "human",
                src_human["content"], parent_id=tgt_parent_id, branch_index=tgt_branch_index,
            )
            new_ai_id = insert_message(
                conn, body.target_session_id, "ai",
                src_ai["content"], parent_id=new_human_id, branch_index=0,
                tool_calls=src_tool_calls,
                interrupted=bool(src_ai["interrupted"]),
            )

            existing = set()
            for child in sorted(children, key=lambda r: r["branch_index"]):
                bi = 0
                while bi in existing:
                    bi += 1
                existing.add(bi)
                conn.execute(
                    "UPDATE messages SET parent_id = ?, branch_index = ? WHERE id = ?",
                    (new_ai_id, bi, child["id"]),
                )

            conn.execute("DELETE FROM messages WHERE id = ?", (tgt_ai["id"],))
            conn.execute("DELETE FROM messages WHERE id = ?", (tgt_human["id"],))

        touch_session(conn, body.target_session_id)
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, f"复制失败: {e}")
    finally:
        conn.close()
    return {"ok": True, "target_session_id": body.target_session_id, "new_human_id": new_human_id}


def _renumber_siblings(conn: sqlite3.Connection, parent_id: str | None, session_id: str):
    rows = conn.execute(
        "SELECT id FROM messages WHERE parent_id IS ? AND role = 'human' AND session_id = ? ORDER BY branch_index",
        (parent_id, session_id),
    ).fetchall()
    for i, row in enumerate(rows):
        conn.execute(
            "UPDATE messages SET branch_index = ? WHERE id = ?",
            (i, row["id"]),
        )
