# AIDev — AI Agent (RAG + 工具 + Web UI)

基于 LangChain / LangGraph 的智能助手，提供 CLI 和 Web 两种交互方式：

- 📚 **RAG 知识库检索**：HyDE + 多查询 + BM25/向量混合检索 + CrossEncoder 精排 + 父子块层级检索
- 🌤️ **天气查询**：实时天气信息
- 🧮 **数学计算**：四则运算（AST 安全求值）
- 🖥️ **Web UI**：React 19 + Tailwind CSS v4，SSE 流式打字机输出 + 工具调用可视化 + 会话树分支

## 架构

```
┌──────────────────────────────────────────────────┐
│                    AIDev                         │
│                                                  │
│  ┌─ CLI ────────────────────────────────────┐   │
│  │  main.py → AgentExecutor (LangChain)     │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌─ Web ────────────────────────────────────┐   │
│  │  frontend (Vite + React 19 + Tailwind)   │   │
│  │    │ POST /api/chat/stream (SSE)         │   │
│  │    ▼                                     │   │
│  │  backend/web_server.py (FastAPI)         │   │
│  │    │                                     │   │
│  │    ▼                                     │   │
│  │  create_react_agent (LangGraph)          │   │
│  │    │                                     │   │
│  │    ├── search_knowledge_base (RAG)       │   │
│  │    ├── get_weather (wttr.in)             │   │
│  │    └── calculate (AST safe-eval)         │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌─ 共享基础设施 ───────────────────────────┐   │
│  │  ChromaDB (向量库) + SQLite (会话历史)   │   │
│  │  Ollama (Qwen3-Embedding + Reranker)     │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

## 检索增强特性

| 特性 | 说明 | 配置项 |
|------|------|--------|
| **语义分块** | 基于 Embedding 相似度检测语义边界切分 | `CHUNK_METHOD=semantic` |
| **层级检索** | 子块精准匹配 → 父块完整上下文 | `PARENT_CHUNK_SIZE=4` |
| **多查询检索** | LLM 扩展 N 个角度查询，合并去重 | `MULTI_QUERY_COUNT=3` |
| **HyDE** | 短查询先生成假设答案再检索 | `HYDE_ENABLED=true` |
| **混合检索** | BM25 + 向量检索融合 | `RETRIEVER_HYBRID_ALPHA=0.3` |
| **CrossEncoder 精排** | Qwen3-Reranker 对候选文档重排序 | `RETRIEVER_CROSSENCODER_TOP_K=5` |

所有配置见 `backend/src/core/settings.py`。

## 快速开始

### 前置条件

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| Python | >= 3.12 | 后端运行时 |
| uv | >= 0.11.0 | Python 包管理（替代 pip） |
| Node.js | >= 18 | 前端构建 |
| Ollama | >= 0.18.2 | 本地 Embedding + Reranker 模型 |

### 1. 安装 Ollama 模型

```bash
# ① 拉取基础 Embedding 模型（Qwen3-Embedding 0.6B Q8_0）
ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0

# ② 创建低上下文版本的配置 tag（解决 Ollama KV 缓存 OOM）
#    模型声明了 32768 上下文窗口，Ollama 会预分配 ~736MB KV 缓存，
#    但 Embedding 只需处理短文本（实际最大父块仅 ~220 tokens）。
#    通过 Modelfile 限制 num_ctx=512，同一份权重，零额外磁盘占用。
echo "FROM dengcao/Qwen3-Embedding-0.6B:Q8_0
PARAMETER num_ctx 512" | ollama create qwen3-embed:0.6b-q8_0-ctx512 -f -

# Reranker 模型（可选，CrossEncoder 精排需要本地部署）
# 首次启动后端时会在 RERANKER_MODEL_PATH 加载
```

### 2. 配置后端

```bash
cd backend

# 使用 uv 初始化环境 
uv venv --python 3.12

# 安装 Python 依赖
uv sync

```

编辑 `backend/.env`，填入以下配置：

```env
# LLM — 使用 OpenAI 兼容 API（必须）
LLM_MODEL=deepseek-v4-flash
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://your-api-endpoint/v1

# Embedding — 本地 Ollama（默认配置通常无需改动）
EMBEDDING_MODEL=qwen3-embed:0.6b-q8_0-ctx512
OLLAMA_BASE_URL=http://localhost:11434
```

> 如果不使用 RAG 知识库检索，可以跳过 Ollama 和索引构建步骤。天气查询和数学计算工具仍可正常使用。

### 3. 构建向量索引（可选，RAG 功能需要）

将文档放入 `backend/data/documents/`，然后运行：

```bash
cd backend
uv run python -m src.retrieval.index_builder
```

支持 `.pdf`、`.docx`、`.txt`、`.md` 格式。也可以通过 Web UI 的文档管理面板上传文件并重建索引。

### 4. 安装前端依赖

```bash
cd frontend
npm install
```

### 启动

#### Web 模式

```bash
# 终端 1 — 后端
cd backend
uv run uvicorn web_server:app --reload --port 8000

# 终端 2 — 前端
cd frontend
npm run dev
```

浏览器打开 `http://localhost:5173`。

#### CLI 模式

```bash
cd backend
uv run python main.py
```

## Web UI 功能

| 功能 | 说明 |
|------|------|
| 流式打字机输出 | SSE 逐 token 推送，`⏹ 停止` 按钮 |
| 工具调用可视化 | 可折叠的工具输入/输出面板 |
| 会话列表 | 新建 / 切换 / 删除 / 重命名 |
| Markdown 渲染 | 代码高亮 + GFM 表格 |
| 会话树分支 | 编辑问题生成分支、分支切换 |
| 节点删除 | 硬删除 + 子节点自动重挂载 |
| 文档上传 | 上传文件 + 全量重建索引 |
| 系统提示词编辑 | 全局可编辑角色指令 |
| 节点复制 | 复制到其他会话（上方/下方/替换） |

## 项目结构

```
AIDev/
├── backend/                         # Python 后端
│   ├── main.py                      # CLI 入口
│   ├── web_server.py                # FastAPI 入口
│   ├── api/                         # Web API 路由
│   │   ├── chat.py                  # SSE 流式端点
│   │   ├── sessions.py              # 会话 & 消息 CRUD + 树结构 + 节点复制
│   │   ├── documents.py             # 文档上传 + 索引重建
│   │   └── memory.py                # 系统提示词 CRUD
│   ├── src/
│   │   ├── core/
│   │   │   ├── settings.py          # 全局配置（环境变量驱动）
│   │   │   └── logger.py
│   │   ├── agent/
│   │   │   ├── agent.py             # CLI Agent + Web Agent (LangGraph)
│   │   │   ├── tools.py             # 外部工具（天气、计算）
│   │   │   └── memory.py            # 长期记忆（profile.json）
│   │   ├── ingestion/
│   │   │   ├── loader.py            # 文档加载
│   │   │   └── chunker.py           # 语义/固定分块
│   │   └── retrieval/
│   │       ├── rag_chain.py         # RAG 检索链
│   │       ├── index_builder.py     # 向量索引构建
│   │       ├── hybrid_retriever.py  # BM25 + 向量混合检索
│   │       ├── reranker.py          # CrossEncoder 精排
│   │       └── bm25.py              # BM25 检索
│   ├── data/
│   │   ├── documents/               # 待索引文档
│   │   ├── db/                      # ChromaDB + SQLite
│   │   └── memory/                  # 用户画像 + 系统提示词
│   ├── tests/                       # pytest 测试套件
│   ├── pyproject.toml
│   └── uv.lock
├── frontend/                        # React 前端
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx                  # 三栏布局容器
│   │   ├── components/
│   │   │   ├── chat/                # 对话区域（消息列表 + 输入框 + 气泡 + 复制模态框）
│   │   │   ├── sidebar/             # 侧边栏（会话列表 + 文档管理 + 系统提示词）
│   │   │   └── tree/                # 右侧栏会话树
│   │   ├── lib/
│   │   │   ├── api.ts               # fetch 封装 + SSE 流式读取
│   │   │   ├── types.ts             # TypeScript 类型
│   │   │   └── AppContext.tsx        # React Context 状态管理
│   │   └── index.css                # Tailwind v4 入口
│   ├── vite.config.ts
│   ├── package.json
│   └── tsconfig.json
└── README.md
```

## 测试

```bash
cd backend
uv run pytest                    # 全量运行 (27 tests)
uv run pytest -k "rag"           # 仅 RAG 测试
uv run pytest -k "calculator"    # 仅计算器安全测试
```

| 模块 | 测试数 | 覆盖内容 |
|------|:------:|----------|
| `test_rag.py` | 4 | 知识库检索行为（考勤/架构/年假/部门） |
| `test_agent.py` | 3 | Agent 三类工具路由正确性 |
| `test_index.py` | 7 | 索引完整性、文档加载、分块重建 |
| `test_tools.py` | 13 | 模块导入、LLM 连通、计算器安全（6 种危险表达式拒绝） |

## 技术栈

| 层 | 技术 |
|----|------|
| Agent 引擎 | LangGraph `create_react_agent`（Web）/ LangChain `AgentExecutor`（CLI） |
| Web 框架 | FastAPI + SSE（Server-Sent Events） |
| 前端 | Vite + React 19 + TypeScript + Tailwind CSS v4 |
| 向量库 | ChromaDB + Ollama (Qwen3-Embedding) |
| 精排 | Qwen3-Reranker (CrossEncoder) |
| 存储 | SQLite（会话历史）+ JSON（用户画像） |
| 包管理 | uv（Python）/ npm（前端） |

## 示例问答

以下是与 Agent 的实际对话示例，展示了三类工具的路由决策和回答效果。

### 📚 知识库检索

```
你: 公司的考勤时间是什么？

助手: 根据公司制度，考勤时间为：
    - 周一至周五
    - 上午 9:00 至下午 6:00
    - 午休时间：12:00 至 13:30
```

```
你: StarBrain 平台用了什么后端框架？

助手: 根据技术架构文档，StarBrain 智能决策平台的后端框架
    是 FastAPI（Python 3.12）。
```

```
你: 公司有多少天年假？

助手: 入职满 1 年享有 5 天年假，之后每满 1 年增加 1 天，
    上限 15 天。未使用的年假可顺延至次年 3 月 31 日。
```

```
你: 公司年会什么时候举办？

助手: 每年 1 月举办公司年会，内容包括年度总结与展望、
    优秀员工颁奖、文艺汇演与抽奖、晚宴。
```

```
你: 公司的数据库用了什么技术？

助手: 根据技术架构文档，公司使用 PostgreSQL 16 作为主数据库，
    配合 Redis 7.2 作为缓存服务。
```

### 🌤️ 天气查询

```
你: 北京今天天气怎么样？

助手: 北京今天 Patchy rain nearby（零星小雨），
    当前气温约 22°C。出门建议带把伞。
```

```
你: 上海天气如何？

助手: 上海当前天气：Sunny +28°C
    今天阳光明媚，适合户外活动。
```

### 🧮 数学计算

```
你: (15 + 23) * 4 - 100 / 5 等于多少？

助手: 我来计算一下... (15+23)×4 - 100/5 = 152 - 20 = 132
```

```
你: 帮我算一下 365 除以 7 的余数

助手: 365 ÷ 7 = 52 余 1
```

### 🚫 知识库范围外（诚实应答）

```
你: 今天晚饭吃什么？

助手: 很抱歉，我查询了公司知识库，但目前没有找到关于
    晚餐菜单的相关信息。建议您：
    1. 直接前往 B1 层员工食堂查看当日菜单
    2. 咨询食堂工作人员
```

```
你: 公司股票代码是多少？

助手: 根据现有文档无法回答该问题。
    知识库中未包含公司上市或股票相关信息。
```

## 常见问题

### Ollama Embedding 报错 "memory layout cannot be allocated"

**现象**：首次查询时出现 `ollama._types.ResponseError: memory layout cannot be allocated (status code: 500)`。

**原因**：Qwen3-Embedding-0.6B 模型声明了 32768 的上下文窗口。Ollama 在加载模型时会预分配 KV 缓存，对于 32768 上下文需要约 736MB 的连续 CPU 内存缓冲区。即使系统总内存充足（如 9GB 空闲），Ollama 的 `ggml` 后端在 Windows 上可能因为内存碎片化而无法分配如此大的连续块。

**解决方案**：使用 Modelfile 创建一个限制上下文窗口的模型副本（同一份权重，零额外磁盘占用）：

```bash
# 1. 拉取基础模型（如果尚未拉取）
ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0

# 2. 创建低上下文版本（echo + 管道，跨平台兼容）
echo "FROM dengcao/Qwen3-Embedding-0.6B:Q8_0
PARAMETER num_ctx 512" | ollama create qwen3-embed:0.6b-q8_0-ctx512 -f -

# 3. 配置 .env 使用新 tag
EMBEDDING_MODEL=qwen3-embed:0.6b-q8_0-ctx512
```

该 tag 与原始模型共享同一份权重 blob（639MB），仅附加了 `num_ctx=512` 配置参数。512 上下文足以覆盖所有实际文档块（最大父块约 440 字 → ~220 tokens）。

**验证**：
```bash
uv run python -c "from langchain_ollama import OllamaEmbeddings; \
  e = OllamaEmbeddings(model='qwen3-embed:0.6b-q8_0-ctx512'); \
  print('OK, dim:', len(e.embed_query('测试')))"
```

### NumPy/OpenBLAS 内存分配失败

**现象**：索引构建或检索时出现 `module compiled against API version 0x10` 或 OpenBLAS 内存错误。

**原因**：多线程 BLAS 库在内存碎片环境下分配工作缓冲区失败。

**解决方案**：运行时限制线程数：

```bash
$env:OMP_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; python main.py
```

可将这一行写入启动脚本或 `.env`（对 Python 进程内的 OpenBLAS 有效）。
