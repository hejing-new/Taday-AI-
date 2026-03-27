<img width="2880" height="1462" alt="9ec8e197c4a65fda96748eea344724d3" src="https://github.com/user-attachments/assets/c68c50cd-a7a6-414a-9212-8785a653a803" />Taday AI 企业级大模型质检与智能客服中台 🤖

基于 FastAPI + Gradio + LlamaIndex + 双模型架构的 RAG 自动巡检系统



请务必在项目根目录配置好 .env 文件，其中 api_key 和 LONGCAT_API_KEY 需要替换为你实际申请的硅基流动（SiliconFlow）和 LongCat 平台的 API Key，否则双模型巡检流水线将无法启动。

📖 项目简介

Taday AI 是一款面向企业级高净值垂直领域（如金融财报、企业知识库）的大模型智能问答与自动巡检中台。系统以前端 Gradio 为交互与 RLHF 数据收集载体，后端基于现代化的 FastAPI 与 LlamaIndex 架构构建。首创“RAG-as-a-Judge”双模型开卷判案机制，有效缓解大模型在生产环境中的幻觉、用户恶意反馈（RLHF 噪音）以及知识库迭代难的问题。

✨ 核心特性与全新升级

🕵️‍♂️ RAG-as-a-Judge 开卷判案 (Ground Truth Verification)：

抛弃大模型“盲猜”逻辑。当处理错题时，系统会优先提取底层真实切片资料（Ground Truth），让老中医模型（Qwen2.5）对照原文进行判案。同时，系统具备极强的业务边界感，能精准识别拦截两类数据诉求：对于财报指标、历史战略等本地库本该有的“静态数据（STATIC）”如果翻车，坚决打回重做；对于今日股价、实时新闻等超纲的“动态数据（DYNAMIC）”，则判定为非知识库缺陷并直接归档。避免让 AI 浪费算力去“修复”根本不存在的实时信息。

🧠 状态机驱动的纯净 RLHF (State Machine Driven Feedback)：

防抖与撤销：在 Gradio 界面中深度嵌入 gr.State 状态机短时记忆，完美支持点赞/点踩的即时渲染与“双击撤销点踩”机制。
从物理源头降噪：确保收集并落库到 B 端的每一条 Negative Feedback 都是用户深思熟虑后的结果。

🛡️ 深度思考与黄金重写 (Deep Thinking & Golden QA)：

高肺活量推理：引入 LongCat 等具备 <think> 逻辑链的深度思考模型作为修复师，并为其开放超大 max_tokens（4096+）限制，防止生成截断。

严守红线：配合严苛的 Prompt 工程，修复师只能基于检索到的真实依据重写答案，若无依据宁可转人工，实现知识库自愈零幻觉。


🏗 系统架构图 (System Architecture)

Plaintext
┌─────────────────────────────────────────────────────────────┐
│               C端: 主业务交互流 (Port: 8000 / 7860)            │
│ ┌─────────────────────────┐         ┌────────────────────┐  │
│ │ Gradio 对话界面 & 状态机 │  👎点踩  │ SQLite 错题/草稿箱   │  │
│ │ (含双击撤销、流式渲染)   ├────────►│ (pending 状态落库)   │  │
│ └────────────┬────────────┘         └─────────┬──────────┘  │
│              │                                │             │
│ ┌────────────▼────────────┐                   │             │
│ │ LlamaIndex RAG 检索引擎 │                   │             │
│ └────────────┬────────────┘                   │             │
│              │                                │             │
└──────────────┼────────────────────────────────┼─────────────┘
               │ HTTP 探针 (规避锁死)            │
┌──────────────▼────────────────────────────────▼─────────────┐
│               B端: 自动巡检与修复中枢 (Port: 8001 / 7861)       │
│                                                             │
│ ┌─────────────────────────┐         ┌────────────────────┐  │
│ │ 1. 探针下潜 (捞取真实切片)│────────►│ 2. Qwen2.5 老中医  │  │
│ └─────────────────────────┘         │   (开卷判案/分类)    │  │
│                                     └─────────┬──────────┘  │
│                                               │             │
│                                     ┌─────────▼──────────┐  │
│                                     │ 3. LongCat 修复师  │  │
│                                     │   (生成黄金答案)     │  │
│                                     └─────────┬──────────┘  │
│                                               ▼             │
│                                     [更新 DB: auto_fixed]   │
└─────────────────────────────────────────────────────────────┘
📂 目录结构

Taday-Finance-Agent/
├── core/                       # 🧠 核心大脑层
│   └── graph.py                # LangGraph 状态图与智能体路由编排中枢
├── tools/                      # 🛠️ 智能体多模态工具箱
│   ├── rag_tool.py             # LlamaIndex 财报本地检索工具 (核心 RAG)
│   ├── web_search_tool.py      # 联网实时检索工具 (补充时效性数据)
│   ├── price_tool.py           # 金融股价实时查询工具 (处理 Dynamic 数据)
│   └── rag_tool_old.py         # 历史 RAG 逻辑备份
├── data/                       # 📚 本地知识库底座
│   └── 宁德时代2025年度报告.pdf  # 实体财报切片源
├── eval/                       # ⚖️ 评测与对齐中心 (LLM-as-a-Judge)
│   ├── run_ragas.py            # 接入 Ragas 框架进行 RAG 质量自动评估
│   ├── auto_eval.py            # 自动化批量评测脚本
│   └── test_set.json           # 标准测试集 (Golden Dataset)
├── temp_storage/               # 🗂️ 运行时的临时文件缓存
├── app_backend.py              # 🚀 C端 主业务后端 API (FastAPI)
├── app_frontend_network.py     # 🖥️ C端 面向用户的交互前端 (Gradio + 状态机)
├── admin_backend.py            # 🛡️ B端 质检与自动巡检后端中枢
├── admin_frontend.py           # 📊 B端 错题管理与大盘看板 (Gradio)
├── auto_healer.py              # 🚑 独立封装的 AI 自动修复流水线脚本
├── init_sql_db.py              # SQLite 数据库建表与初始化脚本
├── knowledge_draft.db          # 🗄️ 本地 SQLite 知识库 (存储 Golden QA 与错题)
├── bad_cases_staging.json      # 暂存区：等待 AI 巡检的错题队列
├── dynamic_cases_archive.json  # 归档区：被识别为超纲/时效性的弃用错题
└── .env                        # 环境变量与 API 密钥配置 (需用户自行创建)

大模型 API 凭证
系统底层依赖多模型协作，请在项目根目录创建 .env 文件并配置以下变量：

Code snippet
# 硅基流动 API (用于 Qwen2.5 老中医诊断与基础 Embedding)
api_key="sk-your_siliconflow_api_key"
base_url="https://api.siliconflow.cn/v1"

# LongCat API (用于深度思考修复师大模型)
LONGCAT_API_KEY="sk-your_longcat_api_key"

🛠 核心工具集 (Tools)

rag_tool：本地非结构化长文本（STATIC 静态知识）。基于 LlamaIndex 与本地 ChromaDB 构建。专门用于“啃”《宁德时代2025年度报告.pdf》等数百页的复杂研报。内置 SentenceSplitter 与全局热缓存机制，不仅能精准提取财务数据，还会强制将溯源切片（Source Nodes）打包返回，为后台诊断提供绝对的“开卷证据”。
price_tool：极高频的金融交易数据（DYNAMIC 强时效数据）。垂直领域的“特种兵”工具。专门对接外部金融行情 API（如Yahoo Finance 等）。能够绕开大模型的知识盲区，毫秒级抓取个股的实时现价、涨跌幅、市盈率等绝对动态指标。
web_search_tool：全网最新资讯与行业动态（DYNAMIC 泛知识）。大模型的“眼睛”。通过外挂搜索引擎 API（如 Tavily、Bing Search），打破 LLM 训练数据的知识截止日期限制。用于交叉验证本地财报数据的市场反馈，或补充突发性的行业政策新闻。

🔄 中间件与状态流转机制

系统通过完整的流水线实现对错误回答的精准捕获与重构：

Code snippet
graph TD
    A[一键启动自动巡检] --> B[HTTP探针获取真实依据]
    B --> C{Qwen 老中医开卷判案}
    
    C -->|资料有且AI答对| D1[FALSE_ALARM]
    D1 -->|直接作废| E1[丢弃/标记ignored]
    
    C -->|询问实时行情| D2[DYNAMIC]
    D2 -->|时效问题| E2[归档跳过]
    
    C -->|AI漏答或出现幻觉| D3[STATIC 静态错误]
    D3 --> F[LongCat 修复师]
    
    F -->|携带真实依据| G[深度思考重写黄金答案]
    G --> H[写入数据库, 状态变更为 auto_fixed]
📚 知识库管理 (RAG)


<img width="2880" height="1462" alt="9882ebd4061cb7efe73d81faf98ef5a7" src="https://github.com/user-attachments/assets/7a2f3cf3-543a-4e6d-9077-0aa1000e70bc" />
<img width="2880" height="1462" alt="6dbd6a92b5c10532a660b7bcd5560565" src="https://github.com/user-attachments/assets/9b26319a-332c-4e85-b36e-7798f27a88a4" />
<img width="2880" height="1462" alt="e02334b396518bebb3b8585b2783960f" src="https://github.com/user-attachments/assets/dce59574-8a08-4f52-ac95-9987cc7131fb" />
<img width="2880" height="1462" alt="c5ce07c44ba981e9f2be6fcfab28b8b0" src="https://github.com/user-attachments/assets/1901ccb0-00f4-4337-a5b5-f586771ff338" />
<img width="1910" height="850" alt="cecd88506e6c29c3c3589da10156faea" src="https://github.com/user-attachments/assets/fa891624-8232-4220-8cf5-8d9bea28ba93" />
<img width="2880" height="1462" alt="9510cb4f4152a9bde7b470f4650b4521" src="https://github.com/user-attachments/assets/12a4cdb2-4919-4712-9cf6-6dd56e4d7807" />
<img width="2880" height="1462" alt="9ec8e197c4a65fda96748eea344724d3" src="https://github.com/user-attachments/assets/77411554-8fd3-4c51-bd43-31583c70e05a" />
<img width="1932" height="378" alt="cbe1c64eb4e2d17d1a4aad0b86915fad" src="https://github.com/user-attachments/assets/f8235e1e-22c0-494f-8212-df550504de6a" />











