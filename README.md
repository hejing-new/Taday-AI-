# Taday 企业级多智能体金融分析系统

基于 LlamaIndex + ChromaDB + FastAPI + Gradio 构建的智能金融分析平台，围绕宁德时代（CATL）财报数据提供混合检索（向量 + BM25 + 数字匹配）、财报智能分块、Cross-Encoder 精排、对话持久化、AI 自动巡检修复等功能。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户界面层                               │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  C 端对话界面     │  │  B 端管理后台     │                    │
│  │  (Gradio:7860)   │  │  (Gradio:7861)   │                    │
│  └────────┬─────────┘  └────────┬─────────┘                    │
├───────────┼──────────────────────┼──────────────────────────────┤
│           │      API 层          │                              │
│  ┌────────┴─────────┐  ┌────────┴─────────┐                    │
│  │  对话后端 API     │  │  管理后台 API     │                    │
│  │  (FastAPI:8002)  │  │  (FastAPI:8004)  │                    │
│  └────────┬─────────┘  └────────┬─────────┘                    │
├───────────┼──────────────────────┼──────────────────────────────┤
│           │      核心层          │                              │
│  ┌────────┴──────────────────────┴─────────┐                    │
│  │          ReAct 引擎 (react_engine)       │                    │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────┐  │                    │
│  │  │RAG 工具  │ │股价工具  │ │Web搜索  │  │                    │
│  │  └──────────┘ └──────────┘ └─────────┘  │                    │
│  │  ┌──────────────────────────────────┐    │                    │
│  │  │  混合检索引擎 (HybridQueryEngine) │    │                    │
│  │  │  向量 60% + BM25 30% + 数字 10%  │    │                    │
│  │  │  + Cross-Encoder 可选精排        │    │                    │
│  │  └──────────────────────────────────┘    │                    │
│  └──────────────────────────────────────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│                        数据存储层                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ ChromaDB │ │ SQLite   │ │ 文件存储  │ │ 对话持久化│           │
│  │ (向量库) │ │ (日志/QA)│ │ (PDF)    │ │ (SQLite) │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 方式一：本地运行

#### 1. 安装依赖

```bash
pip install -r requirement.txt
```

#### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写你的 API Key：

```bash
cp .env.example .env
# 编辑 .env 文件
```

必需的环境变量：

```env
# SiliconFlow Embedding API（用于向量检索）
BASE_URL=https://api.siliconflow.cn/v1
API_KEY=your_siliconflow_api_key

# LongCat Chat API（用于对话和工具调用）
LONGCAT_API_KEY=your_longcat_api_key
LONGCAT_BASE_URL=https://api.longcat.chat/openai/v1

# 管理员认证
ADMIN_USER=admin
ADMIN_PASS=your_secure_password

# 可选：Tavily 搜索（联网搜索）
TAVILY_API_KEY=your_tavily_api_key
```

#### 3. 启动

**一键启动（推荐）：**

```bash
python start.py
```

**单独启动各服务：**

```bash
# 启动 API 服务
python app_backend.py              # 对话后端 (8002)
python admin_backend.py            # 管理后台 API (8004)

# 启动 Gradio 界面
python app_frontend_network.py     # C 端对话 (7860)
python admin_frontend.py           # B 端管理 (7861)
```

**其他启动选项：**

```bash
# 只启动 API 服务（无界面）
python start.py --api-only

# 只启动界面（需要先启动 API）
python start.py --gui-only

# 只初始化数据库
python start.py --init-db
```

#### 4. 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| C 端对话界面 | http://127.0.0.1:7860 | 用户对话 |
| B 端管理后台 | http://127.0.0.1:7861 | 管理控制台 |
| 对话 API 文档 | http://127.0.0.1:8002/docs | Swagger UI |
| 管理 API 文档 | http://127.0.0.1:8004/docs | Swagger UI |

### 方式二：Docker 部署

#### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入 API Key（不需要配置 ADMIN_PASS，Docker 会从 .env 读取）
```

#### 2. 构建并启动

```bash
# 构建镜像并启动所有服务
docker-compose up -d --build

# 只启动 API 服务
docker-compose up -d --build api admin-api

# 只启动前端界面
docker-compose up -d --build chat admin
```

#### 3. 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看单个服务日志
docker-compose logs -f api
docker-compose logs -f chat
```

#### 4. 停止服务

```bash
# 停止服务（保留数据）
docker-compose down

# 停止并删除数据卷（谨慎！）
docker-compose down -v
```

#### 5. Docker 数据持久化

| Volume | 内容 | 容器内路径 |
|--------|------|-----------|
| `chroma_data` | ChromaDB 向量库 | `/app/chroma_db` |
| `app_data` | 上传的 PDF 文件 | `/app/data` |
| `temp_data` | 临时文件 | `/app/temp_storage` |
| bind mount | 对话历史 SQLite | `/app/conversations.db` |

## 运行测试

```bash
# 单元测试
python tests/test_rag_tool.py
python tests/test_chunker.py

# 测试覆盖：
# - test_rag_tool.py: 21 个测试 (分词 + BM25 + 数字boost + 查询扩展 + SQL注入)
# - test_chunker.py: 17 个测试 (章节检测 + 表格检测 + token估算 + 端到端分块)
```

## RAG 评测 (Ragas)

基于 [Ragas](https://github.com/explodinggradients/ragas) v0.2 框架的自动化评测系统，用于量化评估 RAG 检索质量，支持**固定回归集 + 基线对比**，确保每次代码改动可度量、可追溯。

### 评测指标

| 指标 | 说明 | 对应能力 |
|------|------|---------|
| **Faithfulness** | 回答是否基于检索结果生成，不编造 | 整体 RAG 可靠性 |
| **Answer Relevancy** | 回答是否切题，不偏题 | 语义理解质量 |
| **Context Precision** | 检索上下文中有效信息的比例 | 检索精准度（分块/精排） |
| **Context Recall** | 检索是否召回所有相关信息 | 检索覆盖率（BM25/查询扩展） |

### 评测模式

| 模式 | 命令 | 说明 |
|------|------|------|
| **动态出题** | `python eval/auto_eval.py --n 5` | 从 ChromaDB 随机抽题，每次题目不同 |
| **固定回归** | `python eval/auto_eval.py --fixed` | 使用 `regression_set.json` 固定 10 题，每次题目一致 |
| **保存基线** | `python eval/auto_eval.py --fixed --baseline` | 保存当前结果为 `baseline.json` |
| **基线对比** | `python eval/auto_eval.py --fixed --compare` | 和基线做逐指标对比，标注退化/提升 |
| **启用 CE** | 任意命令加 `--ce` | 启用 Cross-Encoder 精排 |

### 固定回归测试集

`eval/regression_set.json` — 10 题覆盖 5 种题型：

| 题型 | 数量 | 考察点 | 示例 |
|------|------|--------|------|
| `numeric` 数字类 | 3 | 精确数字匹配 | "动力电池系统营业收入是多少？" |
| `table` 表格类 | 2 | 表格完整性保护 | "现金流量表中经营活动净额？" |
| `narrative` 叙述类 | 2 | 语义理解能力 | "海外产能扩张战略布局？" |
| `synonym` 同义词类 | 2 | 同义词扩展 | "归母净利润是多少？" → 匹配"归属于上市公司股东的净利润" |
| `numeric` 产能类 | 1 | 产能数据精确检索 | "动力电池系统产能是多少GWh？" |

### 使用流程

```bash
# 1. 改动代码前：保存基线
python eval/auto_eval.py --fixed --baseline

# 2. 改动代码（如调分块器、加检索策略等）

# 3. 改动后：跑对比
python eval/auto_eval.py --fixed --save --compare
```

### 输出示例

```
📊 Taday Ragas 评测报告 — HybridQueryEngine [固定回归]
   时间: 2026-06-24 23:30:00
==========================================================================================

id        category  faithfulness  answer_relevancy  context_precision  context_recall
REV-001   numeric         0.8700            0.8500             0.8200          0.8000
REV-002   numeric         0.9200            0.8800             0.8500          0.8300
REV-003   numeric         0.9000            0.8200             0.7800          0.7500
REV-004   table           0.8500            0.8000             0.7600          0.7200
...

📈 汇总统计:
   faithfulness          = 0.8700 (±0.0421)
   answer_relevancy      = 0.8300 (±0.0350)
   context_precision     = 0.7900 (±0.0512)
   context_recall        = 0.7700 (±0.0633)

📂 按题型分类:
   [numeric] (4 题)
      faithfulness: 0.9025, answer_relevancy: 0.8625
   [table] (2 题)
      faithfulness: 0.8500, answer_relevancy: 0.8000
```

### 基线对比输出

```
==========================================================================================
📊 基线对比报告
   基线时间: 2026-06-24T23:16:00
   当前时间: 2026-06-24T23:30:00
==========================================================================================

指标                   基线       本次       变化     状态
------------------------------------------------------------
faithfulness         0.8200    0.8700   +0.0500   ✅ 提升
answer_relevancy     0.7500    0.8300   +0.0800   ✅ 提升
context_precision    0.7000    0.6800   -0.0200   ⚠️ 退化
context_recall       0.6500    0.8000   +0.1500   ✅ 提升
```

### 评测文件结构

```
eval/
├── auto_eval.py            # 评测脚本 (v3)
├── regression_set.json     # 固定回归测试集 (10 题)
├── baseline.json           # 基线数据 (运行 --baseline 后自动生成)
└── ragas_evaluation_results.json  # 详细结果 (运行 --save 后生成)
```

## 项目结构

```
finance_agent_3/
├── config.py                   # 统一配置管理（支持环境变量覆盖）
├── logger.py                   # 统一日志配置
├── start.py                    # 一键启动脚本
├── .env.example                # 环境变量模板
├── .env                        # 环境变量（不提交 Git）
│
├── core/
│   └── react_engine.py         # ReAct 引擎（工具调用循环）
│
├── tools/
│   ├── rag_tool.py             # RAG 财报检索（混合检索 + CE 精排）
│   ├── financial_chunker.py    # 财报智能分块器
│   ├── price_tool.py           # 股价查询工具
│   ├── web_search_tool.py      # 全网搜索工具（Tavily）
│   └── sql_tool.py             # SQL 财务数据查询
│
├── routes/                     # API 路由（B 端）
│   ├── feedback.py             # 反馈 / Bad Case
│   ├── documents.py            # 文档管理
│   ├── analytics.py            # BI 看板
│   └── auto_heal.py            # AI 自动巡检
│
├── utils/
│   ├── json_store.py           # 共享 JSON 存储（线程安全）
│   └── conversation_store.py   # 对话历史持久化（SQLite）
│
├── app_backend.py              # FastAPI 对话后端
├── admin_backend.py            # FastAPI 管理后台 API
├── app_frontend_network.py     # Gradio C 端对话界面
├── admin_frontend.py           # Gradio B 端管理后台
│
├── tests/                      # 单元测试
│   ├── test_rag_tool.py        # RAG 工具测试 (21 个)
│   └── test_chunker.py         # 分块器测试 (17 个)
│
├── eval/                       # RAG 评测 (Ragas)
│   ├── auto_eval.py            # 评测脚本 (固定回归 + 基线对比)
│   ├── regression_set.json     # 固定回归测试集 (10 题, 5 种题型)
│   └── baseline.json           # 基线数据 (自动生成)
│
├── data/                       # 财报 PDF
├── chroma_db/                  # 向量数据库
├── conversations.db            # 对话历史数据库（自动生成）
│
├── Dockerfile                  # Docker 构建文件
├── docker-compose.yml          # Docker Compose 编排
├── docker-readme.md            # Docker 部署文档
└── requirement.txt             # Python 依赖
```

## 核心功能

### C 端（普通用户）
- **智能对话**: 多轮对话，上下文自动恢复（SQLite 持久化）
- **混合检索**: 向量 60% + BM25 30% + 数字匹配 10%，可选 Cross-Encoder 精排
- **财报查询**: 宁德时代营收、利润、毛利率等详细数据
- **实时股价**: 查询最新股价、涨跌幅
- **全网搜索**: 新闻资讯、宏观政策（Tavily）
- **数据溯源**: 回答附带来源引用，支持点击查看
- **点赞/点踩**: 回答质量反馈

### B 端（管理员）
- **文档管理**: 上传 PDF → 自动解析切块 → 人工审核 → 发布入库
- **Bad Case 质检**: 用户点踩的问题自动分拣 → AI 修复 → 人工复核
- **BI 数据看板**: 搜索流量、响应延迟、热门问题统计
- **黄金答案库**: 人工审核通过的标准答案，可反哺 RAG
- **会话管理**: 查看所有活跃会话和对话历史

## 配置说明

所有配置集中在 `config.py`，支持 `.env` 环境变量覆盖：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `CHAT_MODEL` | LongCat-2.0-Preview | 对话模型 |
| `JUDGE_MODEL` | Qwen/Qwen2.5-72B-Instruct | 评测/裁判模型 |
| `HEAL_MODEL` | LongCat-Flash-Thinking-2601 | 自愈修复模型 |
| `EMBED_MODEL` | BAAI/bge-m3 | Embedding 模型 |
| `PORT_CHAT` | 7860 | C 端 Gradio 端口 |
| `PORT_ADMIN` | 7861 | B 端 Gradio 端口 |
| `PORT_API` | 8002 | FastAPI 对话端口 |
| `PORT_ADMIN_API` | 8004 | FastAPI 管理端口 |
| `MAX_FILE_SIZE` | 50MB | 上传文件大小限制 |

### 可选功能配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `USE_CROSS_ENCODER` | 未设置 | 设为 `1` 启用 Cross-Encoder 精排（需 sentence-transformers） |
| `TAVILY_API_KEY` | 未设置 | 启用联网搜索 |
| `DOCKER_HOST` | 未设置 | Docker 环境自动识别，使用服务名替代 127.0.0.1 |

## 技术栈

- **ReAct 引擎**: 自研 `react_engine.py`（5 次迭代上限，4 个工具）
- **RAG 引擎**: LlamaIndex + ChromaDB
- **混合检索**: 向量 + BM25 + 数字匹配 + 查询扩展
- **精排**: Cross-Encoder（可选，sentence-transformers）
- **分块**: 财报专用分块器（章节切割 → 表格保护 → 语义细分）
- **RAG 评测**: Ragas v0.2（Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall）
- **Web 框架**: FastAPI
- **前端**: Gradio
- **对话持久化**: SQLite（线程安全，WAL 模式）
- **部署**: Docker + Docker Compose

## 环境备忘

- **API 配置**: SiliconFlow (embeddings) + LongCat (chat) — 见 `config.py`
- **数据**: `data/宁德时代2025年度报告.pdf`
- **向量库**: `chroma_db/` (PersistentClient)
- **对话历史**: `conversations.db` (自动生成)
- **单元测试**: `tests/test_rag_tool.py` (21 个) + `tests/test_chunker.py` (17 个)
- **RAG 评测**: `eval/auto_eval.py` (Ragas v0.2, 固定回归 + 基线对比)
- **回归测试集**: `eval/regression_set.json` (10 题, 覆盖 numeric/table/narrative/synonym)
