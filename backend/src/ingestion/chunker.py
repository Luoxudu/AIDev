"""自定义语义分块器 — 不依赖 langchain-experimental。

原理：
  1. 按换行符粗切句子
  2. 计算相邻句子的 Embedding 余弦相似度
  3. 在相似度低于分位阈值的位点断开
"""

import re

import numpy as np
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


class SemanticChunker:
    """基于 Embedding 相似度的语义分块器。

    用法：
        chunker = SemanticChunker(embeddings, percentile=80)
        chunks = chunker.split_documents(docs)
    """

    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        percentile: int = 80,
        min_chunk_size: int = 50,
    ):
        self.embeddings = embeddings
        self.percentile = percentile
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> list[str]:
        """按换行 + 句尾标点分割为句子。"""
        # 先按段落分
        paragraphs = text.split("\n")
        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # 按中文句尾标点再分
            sub = re.split(r"(?<=[。！？；\n])", para)
            for s in sub:
                s = s.strip()
                if s:
                    sentences.append(s)
        return sentences

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """对文档列表进行语义分块。"""
        all_chunks: list[Document] = []

        for doc in documents:
            sentences = self._split_sentences(doc.page_content)
            if len(sentences) <= 1:
                all_chunks.append(doc)
                continue

            # 获取每个句子的 embedding
            sent_embeddings = self.embeddings.embed_documents(sentences)
            sent_embeddings = np.array(sent_embeddings)

            # 计算相邻句子相似度
            similarities: list[float] = []
            for i in range(len(sent_embeddings) - 1):
                sim = np.dot(sent_embeddings[i], sent_embeddings[i + 1])
                similarities.append(sim)

            # 确定分位阈值
            threshold = np.percentile(similarities, self.percentile)

            # 在低相似度处断开
            chunks: list[list[str]] = []
            current_chunk: list[str] = []

            for i, sentence in enumerate(sentences):
                current_chunk.append(sentence)
                if i < len(similarities) and similarities[i] < threshold:
                    # 断开（确保当前 chunk 达到最小长度）
                    if sum(len(s) for s in current_chunk) >= self.min_chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = []

            # 剩余的收尾
            if current_chunk:
                chunks.append(current_chunk)

            # 输出结果
            for chunk_sentences in chunks:
                text = "\n".join(chunk_sentences)
                all_chunks.append(Document(
                    page_content=text,
                    metadata=dict(doc.metadata),
                ))

        print(
            f"[Semantic] 使用自定义 SemanticChunker "
            f"(percentile={self.percentile}) 分割为 {len(all_chunks)} 个子块。"
        )
        return all_chunks
