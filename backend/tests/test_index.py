"""索引构建测试 — 验证文档加载、分块和 Chroma 存储。

公共接口：src.retrieval.index_builder、src.ingestion.loader、src.ingestion.chunker、Chroma collection API。
"""

import tempfile
from pathlib import Path

import pytest


class TestExistingIndex:
    """验证已构建的索引的完整性（不重建，只读检查）。"""

    def test_child_collection_has_chunks(self):
        """子块集合非空 — 索引已正确构建。"""
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        from src.core.settings import PERSIST_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL

        emb = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        store = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name="aidev_knowledge",
            embedding_function=emb,
        )
        count = store._collection.count()
        assert count > 0, f"子块集合为空，请先运行 python -m src.retrieval.index_builder"
        assert count >= 50, f"子块数量过少({count})，预期 >= 50"

    def test_parent_collection_has_chunks(self):
        """父块集合非空。"""
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        from src.core.settings import PERSIST_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL

        emb = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        store = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name="aidev_parents",
            embedding_function=emb,
        )
        count = store._collection.count()
        assert count > 0, f"父块集合为空"
        assert count >= 10, f"父块数量过少({count})，预期 >= 10"

    def test_child_parent_ratio_is_reasonable(self):
        """子块/父块比例合理（PARENT_CHUNK_SIZE=4 → 约 3:1）。"""
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        from src.core.settings import PERSIST_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL

        emb = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        child = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name="aidev_knowledge",
            embedding_function=emb,
        )
        parent = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name="aidev_parents",
            embedding_function=emb,
        )
        child_count = child._collection.count()
        parent_count = parent._collection.count()
        ratio = child_count / parent_count
        # 每个父块 = 4 个子块，重叠 1，比例应在 2~5 之间
        assert 2 <= ratio <= 5, (
            f"子/父比例异常: {child_count}/{parent_count} = {ratio:.1f}"
        )


class TestDocumentLoading:
    """验证文档加载行为。"""

    def test_loader_finds_markdown_files(self):
        """src/ingestion/loader 能找到 data/documents/ 下的 Markdown 文件。"""
        from src.ingestion.loader import load_markdown_files

        docs = load_markdown_files("data/documents")
        assert len(docs) > 0, "应至少加载 1 篇文档"
        # 验证文档有内容
        for doc in docs:
            assert len(doc.page_content) > 100, (
                f"文档内容过短: {doc.metadata.get('source')}"
            )
            assert "source" in doc.metadata, "文档应有 source 元数据"


class TestIndexRebuild:
    """验证索引可重建且不报错（使用临时目录，不影响真实索引）。"""

    def test_build_index_produces_chunks(self):
        """重建索引不报错，且产生非零子块。"""
        from src.retrieval.index_builder import _split_to_child_chunks
        from src.ingestion.loader import load_markdown_files

        docs = load_markdown_files("data/documents")
        child_chunks = _split_to_child_chunks(docs)
        assert len(child_chunks) > 0, "子块不应为空"

    def test_build_parent_chunks_from_children(self):
        """从子块构建父块 — 父块数量合理。"""
        from src.retrieval.index_builder import _build_parent_chunks, _split_to_child_chunks
        from src.ingestion.loader import load_markdown_files

        docs = load_markdown_files("data/documents")
        child_chunks = _split_to_child_chunks(docs)
        parent_chunks = _build_parent_chunks(child_chunks)
        assert len(parent_chunks) > 0, "父块不应为空"
        assert len(parent_chunks) < len(child_chunks), (
            "父块数应少于子块数"
        )

    def test_full_index_rebuild_to_temp(self):
        """完整索引构建→临时目录 — 全流程无异常。"""
        import shutil

        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        from src.retrieval.index_builder import _build_parent_chunks, _split_to_child_chunks
        from src.core.settings import EMBEDDING_MODEL, OLLAMA_BASE_URL
        from src.ingestion.loader import load_markdown_files

        tmpdir = tempfile.mkdtemp(prefix="aidev_test_")
        try:
            emb = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

            docs = load_markdown_files("data/documents")
            child_chunks = _split_to_child_chunks(docs)
            parent_chunks = _build_parent_chunks(child_chunks)

            # 写入临时 Chroma
            child_store = Chroma(
                persist_directory=tmpdir,
                collection_name="test_knowledge",
                embedding_function=emb,
            )
            parent_store = Chroma(
                persist_directory=tmpdir,
                collection_name="test_parents",
                embedding_function=emb,
            )

            # 分批写入（模拟实际流程）
            batch_size = 50
            for i in range(0, len(child_chunks), batch_size):
                batch = child_chunks[i : i + batch_size]
                child_store.add_documents(batch)

            for i in range(0, len(parent_chunks), batch_size):
                batch = parent_chunks[i : i + batch_size]
                parent_store.add_documents(batch)

            assert child_store._collection.count() > 0, "临时子块集合为空"
            assert parent_store._collection.count() > 0, "临时父块集合为空"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
