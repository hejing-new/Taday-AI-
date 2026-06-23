# TODO — 项目优化清单

> 每完成一项打 ✅，按优先级排序。最后更新: 2026-06-24

---

## 🔴 高优先级（生产安全 + 性能瓶颈）

### 安全修复
- [x] **1. 移除 price_tool.py 中硬编码代理** ✅
  - 文件: `tools/price_tool.py`
  - 删除硬编码代理，改为使用系统环境变量

- [x] **2. 修复 admin 默认密码硬编码** ✅
  - 文件: `config.py`
  - 移除默认密码，未设置时输出安全警告

- [x] **3. 加固 SQL 注入防护** ✅
  - 文件: `tools/sql_tool.py`
  - 新增分号检测，禁止多语句注入
  - 保留危险关键词黑名单

### 性能优化
- [x] **4. BM25 统计信息改为预计算（缓存）** ✅
  - 文件: `tools/rag_tool.py`
  - `_compute_bm25_stats()` 在 `__init__` 中计算一次
  - 新增 `doc_freq` 参数传入 `_compute_bm25()`，使用缓存的语料库级 DF
  - 查询时不再重复遍历全部节点

- [x] **5. LLM 客户端改为单例复用** ✅
  - 文件: `core/react_engine.py` + `tools/sql_tool.py`
  - 新增 `_get_llm()` 函数，首次调用时创建，后续复用
  - 减少连接池创建开销

- [x] **6. temp_engines 内存泄漏 — 添加 LRU 淘汰** ✅
  - 文件: `app_backend.py`
  - 使用 `OrderedDict` 实现 LRU，最多保留 50 个会话
  - 新增 `_get_temp_engine()` / `_set_temp_engine()` 封装访问

---

## 🟡 中优先级（代码质量 + 可靠性）

### 死代码 / 冗余清理
- [x] **7. 删除未使用的 import: SemanticSplitterNodeParser** ✅
  - 文件: `tools/rag_tool.py`

- [x] **8. 删除废弃文件 eval/auto_eval1.py** ✅
  - 确认无引用后已删除

- [x] **9. web_search_tool.py 中 print → logger** ✅
  - 文件: `tools/web_search_tool.py`
  - 新增 `from logger import logger`，`print(...)` → `logger.info(...)`

- [x] **10. web_search_tool.py 内容截断提升 + 启用 include_answer** ✅
  - `clean_content[:200]` → `[:1000]`
  - `include_answer=False` → `True`，并附加 Tavily 摘要到输出

### 重复逻辑 / 共享代码
- [x] **11. auto_heal.py 中 fetch_local_knowledge_context 去重** ✅
  - 第二次调用复用第一次的 `ground_truth_context`，减少一次 HTTP 请求

- [x] **12. auto_heal.py 中 LongCat base_url 使用 config 常量** ✅
  - 硬编码 URL → `LONGCAT_BASE_URL`（含 `/openai/` 段）

- [x] **13. feedback.py 和 auto_heal.py 的 JSON 文件操作提取为共享工具** ✅
  - 新建 `utils/json_store.py`，统一管理读写 + 线程锁
  - feedback.py 和 auto_heal.py 都已迁移使用

### 反馈 / vote 机制
- [x] **14. handle_vote 中 user_query 提取逻辑加固** ✅
  - 反向遍历找最后一条 `role == "user"` 的消息

- [x] **15. feedback JSON 文件添加文件锁** ✅
  - 已在 #13 中通过 `utils/json_store.py` 的 `threading.Lock()` 解决

---

## 🟢 低优先级（体验改善 + 额外功能）

### 可观测性
- [x] **16. 为所有 HTTP 外部调用添加 timeout** ✅
  - admin_frontend.py: 3 处已添加 timeout=30
  - app_frontend_network.py: upload 已添加 timeout=60
  - auto_heal.py: fetch_local_knowledge_context 已有 timeout=15
  - app_backend.py: chat_endpoint 内部调用无需额外 timeout

- [x] **17. source_cards 长度设置上限** ✅
  - 单条最多 800 字，总量最多 2000 字

### 配置改进
- [x] **18. 添加 .env.example 文件** ✅
  - 列出所有必需和可选的环境变量

- [x] **19. 模型名称可配置化** ✅
  - 4 个模型名称全部支持环境变量覆盖

### 测试覆盖
- [x] **20. 添加 BM25/RRF 的单元测试** ✅
  - 文件: `tests/test_rag_tool.py`
  - 21 个测试: 分词(5) + BM25(6) + 数字boost(3) + 查询扩展(3) + SQL注入(4)
  - 全部通过

- [x] **21. 添加 financial_chunker 的边界测试** ✅
  - 文件: `tests/test_chunker.py`
  - 17 个测试: 章节检测(5) + 表格检测(4) + token估算(4) + 端到端分块(4)
  - 修复了 `_merge_small_chunks` 中 `token_count` 未初始化的 bug
  - 全部通过

### 对话持久化
- [x] **22. 对话记忆持久化** ✅
  - 新建 `utils/conversation_store.py` — SQLite 存储，线程安全，自动清理 7 天过期数据
  - `store_exchange()`: 每轮对话自动持久化到数据库
  - `get_history()` / `get_history_as_messages()`: 页面加载时恢复对话
  - `clear_history()`: 清空指定会话
  - `get_all_sessions()` / `get_session_stats()`: 管理后台会话列表
  - 后端新增 3 个 API: `GET /api/v1/history/{session_id}`, `DELETE /api/v1/history/{session_id}`, `GET /api/v1/sessions`
  - 前端: 页面加载自动恢复历史，"新聊天"按钮同步清空后端
  - `core/react_engine.py`: `run_react()` 每次回答后自动调用 `store_exchange()`
  - `conversations.db` 文件自动创建

### 部署运维
- [x] **23. Docker 化部署** ✅
  - `Dockerfile`: 多阶段构建 (deps → runtime)，python:3.11-slim
  - `docker-compose.yml`: 4 个服务 (api, admin-api, chat, admin)
  - `.dockerignore`: 排除缓存/数据/密钥
  - `config.py`: `_get_service_url()` 自动识别 Docker 环境 (DOCKER_HOST 环境变量)
  - 数据持久化: Volume (chroma_db, data, temp_storage) + bind mount (conversations.db)
  - 健康检查: curl /docs 端点
  - `docker-readme.md`: 部署文档

---

## 统计

| 优先级 | 总数 | 已完成 | 待办 |
|--------|------|--------|------|
| 🔴 高 | 6 | **6** | 0 |
| 🟡 中 | 9 | **9** | 0 |
| 🟢 低 | 7 | **7** | 0 |
| **合计** | **23** | **23** | **0** |
