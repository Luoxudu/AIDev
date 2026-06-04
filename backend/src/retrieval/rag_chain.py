"""RAG 检索链：增强版 —— HyDE + 多查询 + 混合检索 + CrossEncoder 精排 + 父块层级检索。

检索流水线：
  1. [HyDE] 短查询时生成假设答案，用假设答案的 Embedding 检索
  2. [多查询] 将查询扩展为 N 个不同角度的变体，并行检索
  3. [混合检索] BM25 + 向量检索融合，每路 top-K
  4. [精排] CrossEncoder 对合并结果精排 top-K
  5. [父块] 命中子块 → 返回父块（完整上下文）→ 去重后喂给 LLM

若 BM25 索引不存在，自动退化为纯向量检索（保持向后兼容）。
"""

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings

from src.core.settings import (
    BM25_INDEX_PATH,
    EMBEDDING_MODEL,
    HYDE_ENABLED,
    HYDE_MIN_QUERY_LENGTH,
    MULTI_QUERY_COUNT,
    OLLAMA_BASE_URL,
    PERSIST_DIR,
    RETRIEVER_CROSSENCODER_TOP_K,
    RETRIEVER_HYBRID_ALPHA,
    RETRIEVER_HYBRID_K,
    RETRIEVER_K,
    RETRIEVER_MERGE_TOP_K,
    RETRIEVER_SCORE_THRESHOLD,
    create_llm,
)

# ── HyDE 提示模板 ──
_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一个知识助手。请根据以下问题，写一段可能的回答。"
        "即使你不确定，也请根据常识给出最合理的假设回答。"
        "只需要输出回答内容，不要加任何前缀。"
    )),
    ("human", "{query}"),
])

# ── 多查询扩展提示模板 ──
_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "请将以下用户问题改写为 {n} 个不同角度的查询，"
        "以便从知识库中检索到更全面的信息。"
        "每个查询占一行，不要编号，不要加任何前缀。"
    )),
    ("human", "{query}"),
])


def _generate_hyde(llm, query: str) -> str | None:
    """生成 HyDE 假设回答。

    Args:
        llm: LLM 实例。
        query: 原始用户查询。

    Returns:
        假设回答字符串；如果 HyDE 禁用或不需要则返回 None。
    """
    if not HYDE_ENABLED:
        return None
    if len(query) >= HYDE_MIN_QUERY_LENGTH:
        return None  # 查询本身足够长，不需要 HyDE

    chain = _HYDE_PROMPT | llm
    response = chain.invoke({"query": query})
    hyde_answer = response.content.strip()
    print(f"  [HyDE] 假设回答: {hyde_answer[:80]}...")
    return hyde_answer


def _expand_queries(llm, query: str, hyde_answer: str | None) -> list[str]:
    """生成多查询变体（包含原查询和可选的 HyDE 回答）。

    Args:
        llm: LLM 实例。
        query: 原始用户查询。
        hyde_answer: HyDE 假设回答（可为 None）。

    Returns:
        查询列表，不含空字符串。
    """
    if MULTI_QUERY_COUNT <= 1:
        queries = [query]
        if hyde_answer:
            queries.append(hyde_answer)
        return queries

    chain = _MULTI_QUERY_PROMPT | llm
    response = chain.invoke({"query": query, "n": MULTI_QUERY_COUNT})
    variants = [
        line.strip()
        for line in response.content.strip().split("\n")
        if line.strip()
    ]

    # 组合：原查询 + HyDE + LLM 变体
    queries = [query]
    if hyde_answer:
        queries.append(hyde_answer)
    queries.extend(variants)

    # 去重并限制数量
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    print(f"  [Query] 多查询扩展: {len(unique_queries)} 个查询变体")
    return unique_queries


def _retrieve_parent_documents_hybrid(
    child_store: Chroma,
    parent_store: Chroma,
    queries: list[str],
    bm25_data: tuple | None,
    child_texts: list[str],
    doc_ids: list[str],
) -> list[Document]:
    """混合检索 + CrossEncoder 精排 → 父块映射。

    Args:
        child_store: 子块 Chroma 集合。
        parent_store: 父块 Chroma 集合。
        queries: 待检索的查询列表。
        bm25_data: (BM25Okapi, doc_ids) 元组；None 表示纯向量模式。
        child_texts: 子块文本列表（用于 hybrid_search 映射）。
        doc_ids: 子块 Chroma ID 列表。

    Returns:
        去重后的父块 Document 列表。
    """
    import sys

    from src.retrieval.hybrid_retriever import hybrid_search

    all_child_docs: dict[str, tuple[Document, float]] = {}

    for i, q in enumerate(queries, 1):
        sys.stdout.write(f"\r  [检索 {i}/{len(queries)}] {q[:40]}...")
        sys.stdout.flush()

        results = hybrid_search(
            query=q,
            vector_store=child_store,
            bm25_obj=bm25_data[0],
            doc_ids=doc_ids,
            child_texts=child_texts,
            alpha=RETRIEVER_HYBRID_ALPHA,
            k=RETRIEVER_HYBRID_K,
        )
        for doc, score in results:
            # 用 page_content 去重，取最高分
            content_key = doc.page_content
            if content_key not in all_child_docs or score > all_child_docs[content_key][1]:
                all_child_docs[content_key] = (doc, score)

    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

    if not all_child_docs:
        print("  [WARN] 未检索到相关文档")
        return []

    # 全局 top-15 合并
    merged = sorted(all_child_docs.values(), key=lambda x: x[1], reverse=True)
    merged = merged[:RETRIEVER_MERGE_TOP_K]
    print(f"  [Docs] 混合检索合并: {len(merged)} 个子块")

    # CrossEncoder 精排
    child_docs = [doc for doc, _ in merged]
    from src.retrieval.reranker import rerank
    ranked = rerank(queries[0], [d.page_content for d in child_docs])
    top_indices = [idx for idx, _ in ranked[:RETRIEVER_CROSSENCODER_TOP_K]]
    top_children = [child_docs[i] for i in top_indices]
    print(f"  [Reranker] 精排 top-{RETRIEVER_CROSSENCODER_TOP_K} 完成")

    # 子块 → 父块映射
    all_parent_ids: set[str] = set()
    for doc in top_children:
        pid = doc.metadata.get("parent_id")
        if pid:
            all_parent_ids.add(pid)

    if not all_parent_ids:
        print("  [WARN] 精排后未找到有效父块")
        return []

    # 从父块集合批量获取完整文档
    parent_docs: list[Document] = []
    results = parent_store.get(
        where={"parent_id": {"$in": list(all_parent_ids)}},
        include=["documents", "metadatas"],
    )
    if results["documents"]:
        for doc_text, meta in zip(results["documents"], results["metadatas"]):
            parent_docs.append(Document(page_content=doc_text, metadata=meta))

    print(f"  [Docs] 检索到 {len(all_parent_ids)} 个父块（来自 {len(queries)} 个查询）")
    return parent_docs


def _retrieve_parent_documents_fallback(
    child_store: Chroma,
    parent_store: Chroma,
    queries: list[str],
) -> list[Document]:
    """纯向量检索（BM25 不可用时的降级路径）。

    Args:
        child_store: 子块 Chroma 集合。
        parent_store: 父块 Chroma 集合。
        queries: 待检索的查询列表。

    Returns:
        去重后的父块 Document 列表。
    """
    import sys

    all_parent_ids: set[str] = set()

    for i, q in enumerate(queries, 1):
        sys.stdout.write(f"\r  [检索 {i}/{len(queries)}] {q[:40]}...")
        sys.stdout.flush()
        results = child_store.similarity_search_with_relevance_scores(
            q,
            k=RETRIEVER_K,
        )
        for doc, score in results:
            if score >= RETRIEVER_SCORE_THRESHOLD:
                pid = doc.metadata.get("parent_id")
                if pid:
                    all_parent_ids.add(pid)

    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

    if not all_parent_ids:
        print("  [WARN] 未检索到相关文档")
        return []

    parent_docs: list[Document] = []
    results = parent_store.get(
        where={"parent_id": {"$in": list(all_parent_ids)}},
        include=["documents", "metadatas"],
    )
    if results["documents"]:
        for doc_text, meta in zip(results["documents"], results["metadatas"]):
            parent_docs.append(Document(page_content=doc_text, metadata=meta))

    print(f"  [Docs] 检索到 {len(all_parent_ids)} 个父块（来自 {len(queries)} 个查询）")
    return parent_docs


def _load_bm25_if_available():
    """尝试加载 BM25 索引，不可用时返回 None 并打印 warning。"""
    import os
    from src.retrieval.bm25 import load_bm25_index

    if not os.path.exists(BM25_INDEX_PATH):
        print(f"  [WARN] BM25 索引未构建（{BM25_INDEX_PATH}），退化为纯向量检索")
        return None
    bm25, doc_ids = load_bm25_index(BM25_INDEX_PATH)
    print(f"  [BM25] 索引已加载（{len(doc_ids)} 个子块）")
    return bm25, doc_ids


def _get_all_child_texts(child_store: Chroma) -> tuple[list[str], list[str]]:
    """从 Chroma 获取所有子块的文本和 ID。

    Returns:
        (texts, ids) 两个列表，一一对应。
    """
    results = child_store.get(include=["documents"])
    # results["ids"] 和 results["documents"] 一一对应
    return results["documents"], results["ids"]


def create_rag_chain():
    """创建增强版 RAG 检索链。

    检索流程：HyDE → 多查询 → 混合检索(BM25+向量) → CrossEncoder 精排 → 父块映射。
    若 BM25 索引不存在，自动退化为纯向量检索。

    Returns:
        可调用的 chain，输入 {"input": "..."}，输出 {"answer": "...", "context": [...]}。
    """
    # 1. 创建 LLM
    llm = create_llm()

    # 2. 加载 Chroma 向量库
    print(f"[Connect] 正在连接 Ollama Embedding 服务 ({OLLAMA_BASE_URL})...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    child_store = Chroma(
        collection_name="aidev_knowledge",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    parent_store = Chroma(
        collection_name="aidev_parents",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # 检查向量库是否已构建且非空
    if child_store._collection.count() == 0:
        raise RuntimeError(
            "向量库为空或未构建，请先运行 `python -m src.retrieval.index_builder` 构建索引。"
        )

    # 3. 尝试加载 BM25 索引
    bm25_data = _load_bm25_if_available()

    # 4. 获取子块文本映射（BM25 模式需要）
    child_texts: list[str] = []
    child_ids: list[str] = []
    if bm25_data is not None:
        child_texts, child_ids = _get_all_child_texts(child_store)

    # 5. 定义 RAG 提示模板
    rag_system_prompt = (
        "你是一个企业内部知识库助手。请仅根据以下提供的上下文来回答用户的问题。"
        "如果上下文里没有明确答案，请直接说「根据现有文档无法回答该问题」，不要编造信息。\n\n"
        "上下文：\n{context}"
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        ("human", "{input}"),
    ])

    # 6. 构建 RAG 链
    stuff_chain = create_stuff_documents_chain(llm, rag_prompt)

    def enhanced_retrieve_and_generate(inputs: dict) -> dict:
        """增强检索 + 生成一体化函数。"""
        query = inputs["input"]

        # Step 1: HyDE
        hyde_answer = _generate_hyde(llm, query)

        # Step 2: 多查询扩展
        queries = _expand_queries(llm, query, hyde_answer)

        # Step 3: 检索（混合 or 纯向量降级）
        if bm25_data is not None:
            parent_docs = _retrieve_parent_documents_hybrid(
                child_store, parent_store, queries,
                bm25_data, child_texts, child_ids,
            )
        else:
            parent_docs = _retrieve_parent_documents_fallback(
                child_store, parent_store, queries,
            )

        # Step 4: 生成回答
        if not parent_docs:
            return {
                "answer": "根据现有文档无法回答该问题。",
                "context": [],
            }

        result = stuff_chain.invoke({
            "input": query,
            "context": parent_docs,
        })
        return {
            "answer": result,
            "context": parent_docs,
        }

    return RunnableLambda(enhanced_retrieve_and_generate)
