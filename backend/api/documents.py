"""文档管理：上传文件 + 全量重建索引 + 状态查询。"""

import asyncio
import os
import tempfile
import threading
import time

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

router = APIRouter()

DOCUMENTS_DIR = os.path.abspath("./data/documents")
LAST_BUILD_FILE = os.path.abspath("./data/db/.last_build")

_build_status = "idle"  # idle | building | done | error
_build_error: str | None = None
_build_lock = threading.Lock()


def _safe_path(filename: str) -> str | None:
    safe_name = os.path.basename(filename)
    if not safe_name:
        return None
    dest = os.path.abspath(os.path.join(DOCUMENTS_DIR, safe_name))
    if not os.path.normcase(dest).startswith(os.path.normcase(DOCUMENTS_DIR) + os.sep):
        return None
    return dest


def _write_upload(dest: str, content: bytes) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    dir_ = os.path.dirname(dest)
    fd, tmp = tempfile.mkstemp(dir=dir_)
    try:
        os.write(fd, content)
        os.close(fd)
        os.rename(tmp, dest)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class BuildStatusResponse(BaseModel):
    status: str  # idle / building / done / error / pending
    error: str | None = None


def _list_uploaded_files() -> list[str]:
    if not os.path.exists(DOCUMENTS_DIR):
        return []
    return sorted(
        f for f in os.listdir(DOCUMENTS_DIR)
        if os.path.isfile(os.path.join(DOCUMENTS_DIR, f))
    )


def _latest_file_mtime() -> float:
    """返回文档目录中最新文件的修改时间，空目录返回 0。"""
    if not os.path.exists(DOCUMENTS_DIR):
        return 0
    mtimes = [
        os.path.getmtime(os.path.join(DOCUMENTS_DIR, f))
        for f in os.listdir(DOCUMENTS_DIR)
        if os.path.isfile(os.path.join(DOCUMENTS_DIR, f))
    ]
    return max(mtimes) if mtimes else 0


def _read_last_build() -> float:
    """读取上次索引构建的时间戳，不存在返回 0。"""
    try:
        return float(open(LAST_BUILD_FILE).read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _write_last_build() -> None:
    """写入当前时间作为上次索引构建时间戳。"""
    os.makedirs(os.path.dirname(LAST_BUILD_FILE), exist_ok=True)
    with open(LAST_BUILD_FILE, "w") as f:
        f.write(str(time.time()))


def _check_needs_rebuild() -> bool:
    """对比磁盘文件修改时间与上次构建时间，判断是否需要重建。"""
    return _latest_file_mtime() >= _read_last_build()


def _do_rebuild():
    """后台线程：全量重建索引。"""
    global _build_status, _build_error
    try:
        from src.retrieval.index_builder import build_index
        build_index(DOCUMENTS_DIR)
        from src.agent.agent import reset_rag_chain, reset_web_agent
        reset_rag_chain()
        reset_web_agent()
        _write_last_build()
        with _build_lock:
            _build_status = "done"
    except Exception as e:
        with _build_lock:
            _build_status = "error"
            _build_error = str(e)


def _start_rebuild():
    global _build_status, _build_error
    with _build_lock:
        if _build_status == "building":
            return
        _build_status = "building"
        _build_error = None
    t = threading.Thread(target=_do_rebuild, daemon=True)
    t.start()


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    raw_name = file.filename or "unnamed"
    dest = _safe_path(raw_name)
    if dest is None:
        return {"error": "invalid_filename", "message": "无效的文件名"}
    safe_name = os.path.basename(dest)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in (".md", ".txt", ".pdf", ".docx"):
        return {"error": "unsupported_format", "message": f"不支持的文件格式: {ext}"}

    content = await file.read()
    try:
        await asyncio.to_thread(_write_upload, dest, content)
    except FileExistsError:
        return {"error": "duplicate", "message": f"文件 {safe_name} 已存在"}

    with _build_lock:
        if _build_status == "error":
            _build_status = "idle"

    return {
        "ok": True,
        "filename": safe_name,
        "size": len(content),
    }


@router.post("/documents/rebuild")
def rebuild_index():
    _start_rebuild()
    return {"ok": True, "message": "索引重建已启动"}


@router.get("/documents/status")
def get_build_status():
    global _build_status
    with _build_lock:
        status = _build_status
        error = _build_error

    if status == "building":
        return BuildStatusResponse(status="building")

    if status == "error":
        return BuildStatusResponse(status="error", error=error)

    with _build_lock:
        status = _build_status
        error = _build_error
        if status == "done":
            _build_status = "idle"

    if status == "done":
        return BuildStatusResponse(status="done")

    if status == "error":
        return BuildStatusResponse(status="error", error=error)

    if _check_needs_rebuild():
        return BuildStatusResponse(status="pending")

    return BuildStatusResponse(status="idle")


@router.get("/documents")
def list_documents():
    return {"files": _list_uploaded_files()}


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    filepath = _safe_path(filename)
    if filepath is None:
        return {"error": "invalid_filename", "message": "无效的文件名"}
    if not os.path.exists(filepath):
        return {"error": "not_found", "message": "文件不存在"}
    os.remove(filepath)
    if os.path.exists(LAST_BUILD_FILE):
        os.remove(LAST_BUILD_FILE)
    return {"ok": True}
