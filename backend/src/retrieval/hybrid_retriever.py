"""混合检索：BM25 稀疏检索 + 向量检索融合。

融合公式：final_score = α·bm25_norm + (1-α)·vector_norm
两边均做 Min-Max 归一化到 [0, 1]，消除区间偏移。
"""

from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.retrieval.bm25 import bm25_search
from src.core.settings import RETRIEVER_HYBRID_ALPHA, RETRIEVER_HYBRID_K


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Min-Max 归一化到 [0, 1]。全相同值时返回 [0.5, ...]。"""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_search(
    query: str,
    vector_store: Chroma,
    bm25_obj,
    doc_ids: list[str],
    child_texts: list[str],
    alpha: float = RETRIEVER_HYBRID_ALPHA,
    k: int = RETRIEVER_HYBRID_K,
) -> list[tuple[Document, float]]:
    """混合检索：BM25 + 向量检索融合。

    Args:
        query: 查询字符串。
        vector_store: Chroma 子块集合。
        bm25_obj: BM25Okapi 实例。
        doc_ids: Chroma 文档 ID 列表（与 child_texts 一一对应）。
        child_texts: 子块文本列表（用于构建 Document 映射）。
        alpha: BM25 权重系数。
        k: 每路召回数量。

    Returns:
        [(Document, final_score), ...] 按 final_score 降序。
    """
    # text → [doc_ids] 映射（同一文本可能对应多个 doc_id）
    text_to_ids: dict[str, list[str]] = defaultdict(list)
    for i, t in enumerate(child_texts):
        text_to_ids[t].append(doc_ids[i])
    # 扁平化的反向映射：doc_id → text
    doc_id_to_text = {doc_ids[i]: child_texts[i] for i in range(len(doc_ids))}

    # 1. 向量检索
    vector_results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    vector_scores_raw = [max(score, 0.0) for _, score in vector_results]

    # 2. BM25 检索
    bm25_results = bm25_search(bm25_obj, doc_ids, query, k=k)
    bm25_scores_raw = [score for _, score in bm25_results]

    # 3. Min-Max 归一化
    vector_norms = _min_max_normalize(vector_scores_raw)
    bm25_norms = _min_max_normalize(bm25_scores_raw)

    # 4. 向量结果 → doc_id 映射（用文本查找所有匹配 ID）
    vec_by_id: dict[str, tuple[Document, float]] = {}
    for i, (doc, _) in enumerate(vector_results):
        for chroma_id in text_to_ids.get(doc.page_content, []):
            if chroma_id not in vec_by_id:
                vec_by_id[chroma_id] = (doc, vector_norms[i])

    # BM25 结果映射
    bm25_by_id: dict[str, float] = {}
    for i, (did, _) in enumerate(bm25_results):
        bm25_by_id[did] = bm25_norms[i]

    # 5. 融合所有出现过的 doc_id
    all_ids = set(vec_by_id.keys()) | set(bm25_by_id.keys())
    fused: list[tuple[str, float]] = []
    for did in all_ids:
        v_score = vec_by_id[did][1] if did in vec_by_id else 0.0
        b_score = bm25_by_id.get(did, 0.0)
        final = alpha * b_score + (1 - alpha) * v_score
        fused.append((did, final))

    fused.sort(key=lambda x: x[1], reverse=True)

    # 6. 映射回 Document
    results: list[tuple[Document, float]] = []
    for did, score in fused:
        if did in vec_by_id:
            doc = vec_by_id[did][0]
        else:
            doc = _fetch_doc_from_chroma(vector_store, did, doc_id_to_text.get(did, ""))
            if doc is None:
                continue
        results.append((doc, score))

    return results


def _fetch_doc_from_chroma(
    store: Chroma, doc_id: str, fallback_text: str
) -> Document | None:
    """从 Chroma 按文档 ID 获取 Document。"""
    try:
        res = store.get(ids=[doc_id], include=["documents", "metadatas"])
        if res["documents"]:
            return Document(
                page_content=res["documents"][0],
                metadata=res["metadatas"][0] if res["metadatas"] else {},
            )
    except Exception:
        pass
    if fallback_text:
        return Document(page_content=fallback_text, metadata={})
    return None
