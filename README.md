# Taday 企业级多智能体金融分析系统

基于 LangGraph + LlamaIndex + ChromaDB + FastAPI + Gradio 构建的智能金融分析平台，围绕宁德时代（CATL）财报数据提供 RAG 检索、实时股价查询、全网新闻搜索、AI 自动巡检修复等功能。

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    用户界面层                          │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ C 端对话界面  │  │ B 端管理后台  │                  │
│  │ (Gradio:7860)│  │ (Gradio:7861)│                  │
│  └──────┬───────┘  └──────┬───────┘                  │
├─────────┼─────────────────┼─────────────────────────┤
│         │     API 层      │                          │
│  ┌──────┴───────┐  ┌──────┴───────┐                  │
│  │ 对话后端 API  │  │ 管理后台 API  │                  │
│  │ (FastAPI:8000)│  │(FastAPI:8001)│                  │
│  └──────┬───────┘  └──────┬───────┘                  │
├─────────┼─────────────────┼─────────────────────────┤
│         │    核心层        │                          │
│  ┌──────┴──────────────────┴───────┐                  │
│  │      LangGraph Agent 编排       │                  │
│  │  ┌─────────┐ ┌──────────┐       │                  │
│  │  │RAG 工具 │ │股价工具  │ ...   │                  │
│  │  └─────────┘ └──────────┘       │                  │
│  └─────────────────────────────────┘                  │
├─────────────────────────────────────────────────────┤
│                   数据存储层                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ ChromaDB │ │ SQLite   │ │ 文件存储  │              │
│  │ (向量库) │ │ (日志/QA)│ │ (PDF)    │              │
│  └──────────┘ └──────────┘ └──────────┘              │
└─────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，填入你的 API Key：

```env
api_key=your_siliconflow_api_key
base_url=https://api.siliconflow.cn/v1
LONGCAT_API_KEY=your_longcat_key
TAVILY_API_KEY=your_tavily_key

# 管理员认证（修改为你自己的密码！）
ADMIN_USER=admin
ADMIN_PASS=your_secure_password
```

### 3. 启动

**一键启动（推荐）：**

```bash
python start.py
```

**单独启动各服务：**

```bash
# 初始化财务数据库
python init_sql_db.py

# 启动 API 服务
python app_backend.py          # 对话后端 (8000)
python admin_backend.py        # 管理后台 API (8001)

# 启动 Gradio 界面
python app_frontend_network.py # C 端对话 (7860)
python admin_frontend.py       # B 端管理 (7861)
```

启动后访问：
- **C 端对话**: http://127.0.0.1:7860
- **B 端管理**: http://127.0.0.1:7861

### 4. 运行测试

```bash
cd tests
python test_config.py
python test_sql_tool.py
```

## 项目结构

```
finance_agent_3/
├── config.py              # 统一配置管理
├── logger.py              # 统一日志配置
├── main.py                # CLI 对话入口
├── start.py               # 一键启动脚本
├── .env                   # 环境变量（不提交 Git）
│
├── core/
│   └── graph.py           # LangGraph Agent 编排
│
├── tools/
│   ├── rag_tool.py         # RAG 财报检索工具
│   ├── price_tool.py       # 股价查询工具
│   ├── web_search_tool.py  # 全网搜索工具
│   └── sql_tool.py         # SQL 财务数据查询
│
├── routes/                # API 路由（B 端）
│   ├── feedback.py        # 反馈 / Bad Case
│   ├── documents.py       # 文档管理
│   ├── analytics.py       # BI 看板
│   └── auto_heal.py       # AI 自动巡检
│
├── frontend/              # 前端界面
│   ├── app_frontend_network.py  # C 端对话
│   └── admin_frontend.py       # B 端管理
│
├── eval/                  # 评测脚本
│   ├── auto_eval.py       # Ragas 自动化评测
│   └── test_set.json      # 测试集
│
├── tests/                 # 单元测试
│   ├── test_config.py
│   ├── test_sql_tool.py
│   └── test_auto_healer.py
│
├── data/                  # 财报 PDF
├── chroma_db/             # 向量数据库
└── logs/                  # 日志文件
```

## 核心功能

### C 端（普通用户）
- **智能对话**: 像 ChatGPT 一样自然提问，支持多轮对话
- **财报查询**: 宁德时代营收、利润、毛利率等详细数据
- **实时股价**: 查询最新股价、涨跌幅
- **全网搜索**: 新闻资讯、宏观政策
- **数据溯源**: 回答附带来源引用，支持点击查看
- **点赞/点踩**: 回答质量反馈

### B 端（管理员）
- **文档管理**: 上传 PDF → 自动解析切块 → 人工审核 → 发布入库
- **Bad Case 质检**: 用户点踩的问题自动分拣 → AI 修复 → 人工复核
- **BI 数据看板**: 搜索流量、响应延迟、热门问题统计
- **黄金答案库**: 人工审核通过的标准答案，可反哺 RAG

## 配置说明

所有配置集中在 `config.py`，支持 `.env` 环境变量覆盖：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `CHAT_MODEL` | Qwen2.5-7B-Instruct | 对话模型 |
| `JUDGE_MODEL` | Qwen2.5-72B-Instruct | 评测/裁判模型 |
| `HEAL_MODEL` | LongCat-Flash-Thinking-2601 | 自愈修复模型 |
| `PORT_CHAT` | 7860 | C 端 Gradio 端口 |
| `PORT_ADMIN` | 7861 | B 端 Gradio 端口 |
| `PORT_API` | 8000 | FastAPI 对话端口 |
| `PORT_ADMIN_API` | 8001 | FastAPI 管理端口 |
| `MAX_FILE_SIZE` | 50MB | 上传文件大小限制 |

## 技术栈

- **Agent 编排**: LangGraph
- **RAG 引擎**: LlamaIndex
- **向量数据库**: ChromaDB
- **结构化数据库**: SQLite
- **Web 框架**: FastAPI
- **前端**: Gradio
- **评测**: Ragas
- **LLM**: Qwen2.5 / LongCat


