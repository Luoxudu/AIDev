"""Qwen3-Reranker 精排：基于 sentence_transformers CrossEncoder。

启动时检测本地模型，不存在则从 ModelScope 自动下载。
懒加载 + 线程安全单例。
"""

import os
import threading

from src.core.settings import RERANKER_MODEL_PATH, RERANKER_ENABLED

_MODELSCOPE_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"

# 模块级单例状态
_model = None
_load_failed = False
_lock = threading.Lock()


def find_model_path(base_path: str) -> str | None:
    """在目录中递归查找包含 config.json 的子目录（兼容 ModelScope 下载结构）。"""
    if os.path.isfile(os.path.join(base_path, "config.json")):
        return base_path
    for entry in os.listdir(base_path):
        sub = os.path.join(base_path, entry)
        if os.path.isdir(sub):
            result = find_model_path(sub)
            if result:
                return result
    return None


def check_and_download_reranker_model() -> None:
    """启动时调用：检测本地模型，不存在则从 ModelScope 下载到 RERANKER_MODEL_PATH。"""
    if not RERANKER_ENABLED:
        print("  [Reranker] 已禁用，跳过")
        return

    model_path = find_model_path(RERANKER_MODEL_PATH)
    if model_path:
        print(f"  [Reranker] 检测到本地模型：{model_path}")
        return

    print(f"  [Reranker] 本地模型不存在：{RERANKER_MODEL_PATH}，开始从 ModelScope 下载...")
    os.makedirs(RERANKER_MODEL_PATH, exist_ok=True)
    try:
        from modelscope import snapshot_download
        snapshot_download(
            model_id=_MODELSCOPE_MODEL_ID,
            cache_dir=RERANKER_MODEL_PATH,
            revision="master",
        )
        print(f"  [Reranker] 模型下载完成：{RERANKER_MODEL_PATH}")
    except ImportError:
        print("  [Reranker][WARN] modelscope 未安装，无法自动下载。请手动下载模型或安装 modelscope：uv add modelscope")
    except Exception as e:
        print(f"  [Reranker][WARN] 模型下载失败：{e}")


def _load_model():
    """加载 CrossEncoder 模型（首次调用时触发）。"""
    import torch
    from sentence_transformers import CrossEncoder

    global _model, _load_failed

    try:
        model_path = find_model_path(RERANKER_MODEL_PATH)
        if not model_path:
            raise FileNotFoundError(f"未在 {RERANKER_MODEL_PATH} 下找到模型文件")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("  [Reranker][WARN] GPU 不可用，使用 CPU 推理，速度较慢")

        _model = CrossEncoder(
            model_path,
            max_length=512,
            device=device,
        )
        print(f"  [Reranker] 模型已加载到 {device}")
    except Exception as e:
        _load_failed = True
        print(f"  [Reranker][WARN] 模型加载失败，将跳过精排: {e}")


def _ensure_loaded():
    """确保模型已加载（线程安全），加载失败时不做任何事。"""
    global _load_failed
    if not RERANKER_ENABLED:
        _load_failed = True
        return
    if _model is None and not _load_failed:
        with _lock:
            if _model is None and not _load_failed:
                _load_model()


def is_available() -> bool:
    """Reranker 是否可用。"""
    _ensure_loaded()
    return _model is not None


def rerank(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """对 query-doc pairs 做精排。

    若模型不可用，返回原始顺序（均匀递减分数）。

    Args:
        query: 原始用户查询。
        documents: 文档文本列表。

    Returns:
        按相关性降序排列的 (原始索引, 分数) 列表。
    """
    import torch

    _ensure_loaded()

    if _model is None:
        return [(i, 1.0 - i * 0.01) for i in range(len(documents))]

    pairs = [(query, doc) for doc in documents]
    with torch.no_grad():
        scores = _model.predict(pairs, batch_size=len(pairs))

    results = list(enumerate(scores.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
