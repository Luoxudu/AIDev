"""AIDev - AI Agent 项目配置。"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI 兼容的大模型配置 ──
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# ── 本地 Embedding 配置（通过 Ollama）──
# 默认使用 ctx512 变体避免 KV 缓存 OOM
# 原模型 32768 上下文需 736MB KV 缓存，512 上下文仅需 ~12MB
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "qwen3-embed:0.6b-q8_0-ctx512"
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── 向量库目录 ──
PERSIST_DIR = os.getenv("PERSIST_DIR", "./data/db/chroma_db")

# ══════════════════════════════════════════════════════════════
#  记忆系统配置
# ══════════════════════════════════════════════════════════════

MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_AUTO_EXTRACT = os.getenv("MEMORY_AUTO_EXTRACT", "true").lower() == "true"
MEMORY_PROFILE_PATH = os.getenv("MEMORY_PROFILE_PATH", "./data/memory/profile.json")

# ── 短期记忆（对话历史）──
CHAT_HISTORY_DB = os.getenv("CHAT_HISTORY_DB", "./data/db/chat_history.db")
CHAT_HISTORY_WEB_DB = os.getenv("CHAT_HISTORY_WEB_DB", "./data/db/chat_history_web.db")
CHAT_HISTORY_WINDOW = int(os.getenv("CHAT_HISTORY_WINDOW", "10"))

# ══════════════════════════════════════════════════════════════
#  分块配置
# ══════════════════════════════════════════════════════════════

# 分块方法："semantic"（语义分块）| "fixed"（固定大小分块）
CHUNK_METHOD = os.getenv("CHUNK_METHOD", "semantic")

# 语义分块参数（CHUNK_METHOD="semantic" 时生效）
# 相邻句子 Embedding 相似度低于此百分位时断句，范围 50-99
# 值越小 → 断点越多 → 子块越多越细（对 Qwen3-Embedding 建议 75-85）
SEMANTIC_CHUNK_PERCENTILE = int(os.getenv("SEMANTIC_CHUNK_PERCENTILE", "80"))

# 固定分块参数（CHUNK_METHOD="fixed" 时生效）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# ── 父块参数（层级检索）──
# 每 N 个子块合并为 1 个父块（改这个数字即可调整父块粒度，建议 3-5）
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "4"))
# 相邻父块之间重叠的子块数（平滑过渡，建议 1-2）
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "1"))

# ══════════════════════════════════════════════════════════════
#  检索增强配置
# ══════════════════════════════════════════════════════════════

# 检索参数
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
# Qwen3-Embedding 向量已归一化，但 Chroma 用 L2 距离折算相似度：
# cosine 0.5 → score 0.0, cosine 0.62 → score 0.15, cosine 0.76 → score 0.31
# 阈值 0.15 对应 cosine ≈ 0.62，保留强相关文档
RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.15"))

# 多查询检索：将用户问题扩展为 N 个不同角度的查询，分别检索后合并结果
# 设为 1 则禁用多查询（仅用原问题检索）
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))

# HyDE（假设文档嵌入）：对于短查询，先生成假设答案再用 Embedding 检索
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "true").lower() == "true"
# 查询长度 < 此值时触发 HyDE（短查询语义稀疏，HyDE 收益最大）
HYDE_MIN_QUERY_LENGTH = int(os.getenv("HYDE_MIN_QUERY_LENGTH", "10"))


# ══════════════════════════════════════════════════════════════
#  混合检索配置
# ══════════════════════════════════════════════════════════════

# 混合检索每路召回数量（BM25 和向量各取 top-K）
RETRIEVER_HYBRID_K = int(os.getenv("RETRIEVER_HYBRID_K", "10"))
# BM25 权重系数 α，融合公式：α·bm25_norm + (1-α)·vector_norm
RETRIEVER_HYBRID_ALPHA = float(os.getenv("RETRIEVER_HYBRID_ALPHA", "0.3"))
# CrossEncoder 精排后取 top-K 个子块
RETRIEVER_CROSSENCODER_TOP_K = int(os.getenv("RETRIEVER_CROSSENCODER_TOP_K", "5"))
# 混合检索合并后送入精排的最大子块数
RETRIEVER_MERGE_TOP_K = int(os.getenv("RETRIEVER_MERGE_TOP_K", "15"))
# BM25 索引持久化路径
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/db/bm25_index.pkl")
# Qwen3-Reranker 本地模型目录（不存在时自动从 ModelScope 下载到此目录）
RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    r"D:\Hugging_Face\models\Qwen3-Reranker-0.6B"
)
# 是否启用 Reranker 精排（true/false）
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

# ══════════════════════════════════════════════════════════════
#  共享工厂
# ══════════════════════════════════════════════════════════════

def create_llm(*, streaming: bool = False):
    """创建共享的 ChatOpenAI 实例。"""
    from langchain_openai import ChatOpenAI

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY 未设置，请在 .env 中配置。"
        )
    from pydantic import SecretStr

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        base_url=OPENAI_BASE_URL,
        streaming=streaming,
        temperature=0,
        max_retries=2,
    )
