"""BM25 稀疏检索：索引构建、持久化、加载、检索。

使用 jieba 中文分词 + rank_bm25.BM25Okapi 算法。
持久化格式：pickle 序列化 (tokenized_corpus, doc_ids) 元组。
"""

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """用 jieba 对中文文本分词，过滤空白 token。"""
    return [t for t in jieba.lcut(text) if t.strip()]


def build_bm25_index(texts: list[str], doc_ids: list[str]) -> tuple[list[list[str]], list[str]]:
    """构建 BM25 索引所需的分词语料。

    Args:
        texts: 子块文本列表。
        doc_ids: 与 texts 一一对应的 Chroma 文档 ID 列表。

    Returns:
        (tokenized_corpus, doc_ids) 元组，可直接传给 save_bm25_index。
    """
    tokenized_corpus = [tokenize(t) for t in texts]
    return tokenized_corpus, doc_ids


def save_bm25_index(
    tokenized_corpus: list[list[str]],
    doc_ids: list[str],
    path: str | Path,
) -> None:
    """将 BM25 索引持久化到磁盘。

    Args:
        tokenized_corpus: 分词后的语料。
        doc_ids: 对应的 Chroma 文档 ID 列表。
        path: 序列化目标路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((tokenized_corpus, doc_ids), f)


def load_bm25_index(path: str | Path) -> tuple[BM25Okapi, list[str]]:
    """从磁盘加载 BM25 索引。

    Args:
        path: 序列化文件路径。

    Returns:
        (BM25Okapi 实例, doc_ids 列表)。
    """
    with open(path, "rb") as f:
        tokenized_corpus, doc_ids = pickle.load(f)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, doc_ids


def bm25_search(
    bm25: BM25Okapi,
    doc_ids: list[str],
    query: str,
    k: int,
) -> list[tuple[str, float]]:
    """BM25 检索。

    Args:
        bm25: 已加载的 BM25Okapi 实例。
        doc_ids: 对应的 Chroma 文档 ID 列表。
        query: 用户查询字符串。
        k: 返回前 K 个结果。

    Returns:
        [(doc_id, score), ...] 按 score 降序排列。
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    scored = [(doc_ids[i], float(scores[i])) for i in range(len(doc_ids))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
