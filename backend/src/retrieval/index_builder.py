"""构建向量索引：从 data/ 加载 Markdown 文档，分块并存入 Chroma。

支持两种分块方法（通过 config.CHUNK_METHOD 切换）：
  - "semantic": 自定义 SemanticChunker，基于 Embedding 相似度检测语义边界
  - "fixed":    RecursiveCharacterTextSplitter，固定大小机械切分

同时构建父子文档结构，用于层级检索。
"""

import os
import shutil
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunker import SemanticChunker
from src.core.settings import (
    BM25_INDEX_PATH,
    CHUNK_METHOD,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PERSIST_DIR,
    SEMANTIC_CHUNK_PERCENTILE,
)
from src.ingestion.loader import load_documents


def _split_to_child_chunks(documents: list[Document]) -> list[Document]:
    """将文档切分为子块。

    Returns:
        子块列表，每个子块保留原始文档的 metadata。
    """
    if CHUNK_METHOD == "semantic":
        print(
            f"[Semantic] 使用自定义 SemanticChunker "
            f"(percentile={SEMANTIC_CHUNK_PERCENTILE})..."
        )
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        splitter = SemanticChunker(
            embeddings=embeddings,
            percentile=SEMANTIC_CHUNK_PERCENTILE,
        )

    elif CHUNK_METHOD == "fixed":
        print(
            f"[Fixed] 使用固定分块（RecursiveCharacterTextSplitter, "
            f"size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）..."
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )

    # ---- 如有需要，可在此添加 "llm" 分支，使用 LLM 驱动切分 ----
    # elif CHUNK_METHOD == "llm":
    #     from langchain_openai import ChatOpenAI
    #     splitter = LLMChunker(llm=ChatOpenAI(...))
    else:
        raise ValueError(
            f"不支持的分块方法: {CHUNK_METHOD}，可选: semantic, fixed"
        )

    chunks = splitter.split_documents(documents)
    print(f"[OK] 已分割为 {len(chunks)} 个子块。")
    return chunks


def _build_parent_chunks(child_chunks: list[Document]) -> list[Document]:
    """将子块合并为父块。

    每 PARENT_CHUNK_SIZE 个子块合并为一个父块，
    相邻父块之间重叠 PARENT_CHUNK_OVERLAP 个子块。

    Args:
        child_chunks: 已切好的子块列表。

    Returns:
        父块列表。
    """
    parent_chunks: list[Document] = []
    step = PARENT_CHUNK_SIZE - PARENT_CHUNK_OVERLAP  # 滑动步长

    for i in range(0, len(child_chunks), step):
        group = child_chunks[i : i + PARENT_CHUNK_SIZE]
        if not group:
            continue

        merged_text = "\n\n".join(
            chunk.page_content for chunk in group
        )
        merged_meta = dict(group[0].metadata)
        merged_meta["child_count"] = len(group)

        parent_chunks.append(Document(
            page_content=merged_text,
            metadata=merged_meta,
        ))

    # 给每个子块和父块标上所属的 parent_id
    for parent_idx, parent_doc in enumerate(parent_chunks):
        parent_id = f"p_{parent_idx}"
        parent_doc.metadata["doc_type"] = "parent"
        parent_doc.metadata["parent_id"] = parent_id
        parent_doc.metadata["doc_id"] = str(uuid.uuid4())

        # 给对应子块标注 parent_id（重叠区只保留首次分配，避免覆盖）
        start_child_idx = parent_idx * step
        for j in range(PARENT_CHUNK_SIZE):
            child_idx = start_child_idx + j
            if child_idx < len(child_chunks):
                child_chunks[child_idx].metadata["doc_type"] = "child"
                if "parent_id" not in child_chunks[child_idx].metadata:
                    child_chunks[child_idx].metadata["parent_id"] = parent_id

    print(
        f"[OK] 已合并为 {len(parent_chunks)} 个父块 "
        f"（每 {PARENT_CHUNK_SIZE} 子块 → 1 父块，重叠 {PARENT_CHUNK_OVERLAP}）。"
    )
    return parent_chunks


def build_index(data_dir: str = "./data/documents") -> Chroma:
    """从 data/ 加载文档 → 分块 → 生成父子结构 → 存入 Chroma。

    Returns:
        构建好的 Chroma 向量库实例（子块 collection）。
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"数据目录 {data_dir} 不存在，请先创建并放入 .md 文档。"
        )

    # 0. 清理旧数据（Chroma + BM25 索引同步清理）
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    if os.path.exists(BM25_INDEX_PATH):
        os.remove(BM25_INDEX_PATH)

    # 1. 加载文档
    docs = load_documents(data_dir)

    # 2. 切分为子块
    child_chunks = _split_to_child_chunks(docs)

    if not child_chunks:
        raise ValueError(
            "分块结果为空，请检查文档内容是否过短，或调整分块参数。"
        )

    # 3. 构建父块
    parent_chunks = _build_parent_chunks(child_chunks)

    # 4. 创建 Ollama Embedding
    print(f"[Connect] 正在连接 Ollama Embedding 服务 ({OLLAMA_BASE_URL})...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # 5. 清空旧数据
    # (已在函数开头完成)

    # 6. 存储子块 → 用于相似度检索
    print(f"[Save] 存储 {len(child_chunks)} 个子块到 Chroma...")
    child_ids = [f"c_{i}" for i in range(len(child_chunks))]
    vectorstore = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="aidev_knowledge",
        ids=child_ids,
    )

    # 7. 存储父块 → 用于返回完整上下文
    print(f"[Save] 存储 {len(parent_chunks)} 个父块到 Chroma...")
    parent_ids = [f"p_{i}" for i in range(len(parent_chunks))]
    Chroma.from_documents(
        documents=parent_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="aidev_parents",
        ids=parent_ids,
    )

    # 8. 构建 BM25 索引并持久化
    from src.retrieval.bm25 import build_bm25_index, save_bm25_index

    print(f"[BM25] 构建 BM25 索引（{len(child_chunks)} 个子块）...")
    texts = [c.page_content for c in child_chunks]
    tokenized_corpus, d_ids = build_bm25_index(texts, child_ids)
    save_bm25_index(tokenized_corpus, d_ids, BM25_INDEX_PATH)
    print(f"[BM25] BM25 索引已保存到 {BM25_INDEX_PATH}")

    print(f"[OK] 向量库已持久化到 {PERSIST_DIR}")
    print(f"   子块: {len(child_chunks)} 条（collection: aidev_knowledge）")
    print(f"   父块: {len(parent_chunks)} 条（collection: aidev_parents）")

    return vectorstore


if __name__ == "__main__":
    build_index()
