import os
import uuid
import sqlite3
import uvicorn
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import requests

app = FastAPI(title="Taday 知识库管理后台 API", version="2.0.0")
DB_FILE = "knowledge_draft.db"


# 定义暂存 JSON 文件的路径 (主草稿箱)
JSON_LOG_FILE = "bad_cases_staging.json"

# 🌟 新增：定义动态失效数据的归档文件 (冷库)
DYNAMIC_JSON_FILE = "dynamic_cases_archive.json"

if not os.path.exists(JSON_LOG_FILE):
    with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

if not os.path.exists(DYNAMIC_JSON_FILE):
    with open(DYNAMIC_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

# ==========================================
# 1. 数据库初始化 (增加时间戳与状态流转)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE, timeout=10) # 增加 timeout 防止多线程锁表
    cursor = conn.cursor()
    # 文档表：记录资产全生命周期
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT, -- 'processing', 'pending', 'published', 'failed'
            created_at TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            text_content TEXT,
            chunk_index INTEGER,
            status TEXT
        )
    ''')
    # 在 init_db() 函数的 conn.commit() 前方加入这段代码，记录用户点踩事件，bad_cases 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bad_cases (
            case_id TEXT PRIMARY KEY,
            user_query TEXT,
            ai_response TEXT,
            status TEXT, -- 'pending' (待处理), 'fixed' (已修复), 'ignored' (忽略)
            created_at TEXT
        )
    ''')
    # 在 init_db() 的 conn.commit() 前追加： 记录用户搜索日志，search_logs 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_logs (
            log_id TEXT PRIMARY KEY,
            user_query TEXT,
            session_id TEXT,
            latency REAL, -- 记录大模型思考耗时 (秒)
            created_at TEXT
        )
    ''')
    # 🌟 新增：存放经过人工质检的黄金答案库
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS golden_qa (
            id TEXT PRIMARY KEY,
            original_query TEXT,
            corrected_answer TEXT,
            source_case_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. Pydantic 数据模型
# ==========================================
class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    created_at: str
    chunk_count: int

class ChunkUpdateParams(BaseModel):
    new_text: str

class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    text_content: str
    chunk_index: int
    status: str

# ==========================================
# 🚀 新增：RLHF / Bad Case 收集与管理接口
# ==========================================
class FeedbackRequest(BaseModel):
    user_query: str
    ai_response: str

# ==========================================
# 🚀 异步质检流水线 API (JSON 缓冲 -> DB 入库)
# ==========================================
import requests

def fetch_local_knowledge_context(query: str) -> str:
    """通过 HTTP 向前台主业务借答案，防止本地 ChromaDB 锁死"""
    try:
        # ⚠️ 换成你真实的主业务提问 API 接口
        url = "http://127.0.0.1:8000/api/v1/chat" 
        payload = {"query": f"请仅依靠内部知识库查证：{query}"} 
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("response", resp.text)
    except Exception as e:
        print(f"❌ [探针异常] {e}")
    return "未找到资料"

# ==========================================
# 🚀 C端专用：将点踩数据写入 JSON (防弹+去重幂等版)
# ==========================================
@app.post("/api/v1/feedback", summary="C端：记录 Bad Case 到 JSON 暂存区")
def add_feedback_to_json(req: FeedbackRequest):
    try:
        case_id = f"bc_{uuid.uuid4().hex[:8]}"
        new_case = {
            "case_id": case_id,
            "user_query": req.user_query,
            "ai_response": req.ai_response,
            "status": "pending",  # 默认待处理
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        cases = []
        if os.path.exists(JSON_LOG_FILE):
            try:
                with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        cases = json.loads(content)
            except Exception as e:
                print(f"[⚠️ JSON恢复] {e}")
                cases = []
                
        # ==========================================
        # 🌟 核心防雷：幂等性去重拦截 (Idempotency)
        # ==========================================
        for existing_case in cases:
            # 如果草稿箱里已经有一模一样的问题、一模一样的回答，并且还在 pending 状态
            if (existing_case.get('user_query') == req.user_query and 
                existing_case.get('ai_response') == req.ai_response and 
                existing_case.get('status') == 'pending'):
                
                print(f"🛡️ [拦截] 检测到重复点踩，已静默去重。")
                # 直接给前端返回成功，假装写入了，但其实什么都没干！
                return {"status": "success", "message": "already_exists", "case_id": existing_case.get('case_id')}
                
        # 如果没有重复，才真正追加进去
        cases.append(new_case)
        
        with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "case_id": case_id}
        
    except Exception as e:
        print(f"❌ 严重错误 (点踩写入): {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
        
# ==========================================
# 🚀 B端专用：读取 JSON 里的待质检列表 (防弹装甲版)
# ==========================================
@app.get("/admin/api/bad_cases", summary="B端：读取 JSON 暂存数据")
def get_cases_from_json():
    # 🌟 防雷 1：文件被删了？没关系，直接返回空列表
    if not os.path.exists(JSON_LOG_FILE):
        return []
        
    try:
        with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            # 🌟 防雷 2：文件变成 0KB 的空文件了？也返回空列表
            if not content.strip():
                return []
            cases = json.loads(content)
            
        # 按时间倒序返回
        return sorted(cases, key=lambda x: x.get('created_at', ''), reverse=True)
        
    except Exception as e:
        # 🌟 防雷 3：文件格式彻底坏了？打印报错，但不让前端崩溃
        print(f"❌ 读取草稿箱失败: {e}")
        return []
# ==========================================
# 🚀 C端专用：撤销点踩 (防弹装甲版)
# ==========================================
@app.post("/api/v1/feedback/cancel", summary="C端：撤销点踩，从 JSON 移除记录")
def cancel_feedback_in_json(req: FeedbackRequest):
    try:
        if not os.path.exists(JSON_LOG_FILE):
            return {"status": "success"}

        cases = []
        try:
            with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    cases = json.loads(content)
        except:
            return {"status": "success"} # 文件都坏了，也就不需要撤销了
        
        # 核心过滤逻辑：使用 get 防止旧数据缺少字段导致 KeyError 崩溃
        new_cases = [
            c for c in cases 
            if not (c.get('user_query') == req.user_query and 
                    c.get('ai_response') == req.ai_response and 
                    c.get('status') == 'pending')
        ]
        
        with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_cases, f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "message": "已从待处理列表中移除"}
    except Exception as e:
        print(f"❌ 严重错误 (撤销点踩): {e}")
        return {"status": "error", "message": str(e)}

# 3. 接收请求的模型参数
class CorrectionRequest(BaseModel):
    correct_answer: str

# 4. B端专用：人工质检完毕，写入 DB 黄金库，并更新 JSON 状态
@app.post("/admin/api/bad_cases/{case_id}/fix", summary="B端：人工质检完毕，入库并更新状态")
def fix_bad_case(case_id: str, req: CorrectionRequest):
    # 1. 遍历 JSON，找到这条数据并更新状态
    with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        
    target_case = None
    for c in cases:
        if c['case_id'] == case_id:
            c['status'] = "fixed"
            target_case = c
            break
            
    if not target_case:
        return {"status": "error", "message": "未找到该案列"}

    # 写回更新后的 JSON
    with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
        
    # 2. 将原问题和【人工写的正确答案】写入 SQLite 的 golden_qa 表
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO golden_qa (id, original_query, corrected_answer, source_case_id) VALUES (?, ?, ?, ?)",
        (f"qa_{uuid.uuid4().hex[:8]}", target_case['user_query'], req.correct_answer, case_id)
    )
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "质检通过！已归档至黄金问答库(DB)"}

# ==========================================
# 🚀 补充：C端取消点踩 / 改为点赞的撤销接口
# ==========================================
@app.post("/api/v1/feedback/cancel", summary="C端：撤销点踩，从 JSON 暂存区移除记录")
def cancel_feedback_in_json(req: FeedbackRequest):
    try:
        if not os.path.exists(JSON_LOG_FILE):
            return {"status": "success"}

        with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        # 🌟 核心过滤逻辑：保留那些“不是这条数据”或者“已经处理过”的数据
        # 换句话说：把匹配到原问题、原回答，且状态还是 pending 的脏数据给删掉
        new_cases = [
            c for c in cases 
            if not (c['user_query'] == req.user_query and 
                    c['ai_response'] == req.ai_response and 
                    c['status'] == 'pending')
        ]
        
        # 写回 JSON
        with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_cases, f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "message": "已从待处理列表中移除"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
# ==========================================
# 🚀 新增：BI 数据看板接口 (搜索埋点与大盘)
# ==========================================
class SearchLogRequest(BaseModel):
    user_query: str
    session_id: str
    latency: float

@app.post("/api/v1/log_search", summary="C端专用：记录一次搜索行为")
def log_search(req: SearchLogRequest):
    log_id = f"log_{uuid.uuid4().hex[:8]}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO search_logs VALUES (?, ?, ?, ?, ?)", 
        (log_id, req.user_query, req.session_id, req.latency, now_str)
    )
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/admin/api/analytics", summary="B端专用：拉取运营大盘数据")
def get_analytics():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. 拉取所有搜索明细 (按时间倒序)
    cursor.execute("SELECT user_query, session_id, latency, created_at FROM search_logs ORDER BY created_at DESC")
    logs = cursor.fetchall()
    
    # 2. 统计今日核心指标 (这里做个简单的汇总传给前端)
    total_searches = len(logs)
    avg_latency = sum([r[2] for r in logs]) / total_searches if total_searches > 0 else 0
    
    conn.close()
    
    return {
        "metrics": {
            "total_searches": total_searches,
            "avg_latency": round(avg_latency, 2)
        },
        "logs": [{"query": r[0], "session": r[1], "latency": r[2], "time": r[3]} for r in logs]
    }

# ==========================================
# 🚀 3. 核心异步解析任务 (Background Task)
# ==========================================
def process_document_task(doc_id: str, temp_path: str, filename: str):
    """这个函数将在 FastAPI 的后台线程池中默默运行，绝不阻塞主线程"""
    print(f"⏳ [后台任务开始] 正在努力切分文档: {filename}...")
    try:
        # 1. 耗时的 PDF 解析
        loader = PyMuPDFReader()
        docs = loader.load_data(file_path=temp_path)
        
        # 2. 耗时的文本切块
        parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(docs)
        
        # 3. 存入数据库
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        
        for i, node in enumerate(nodes):
            chunk_id = f"chk_{uuid.uuid4().hex[:8]}"
            cursor.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", 
                (chunk_id, doc_id, node.get_content(), i, 'active')
            )
            
        # 4. 状态机流转：处理中 -> 待审核
        cursor.execute("UPDATE documents SET status = 'pending' WHERE doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()
        print(f"✅ [后台任务完成] {filename} 切分完毕，共产生 {len(nodes)} 个切片。")
        
    except Exception as e:
        print(f"❌ [后台任务失败] {filename} 解析异常: {str(e)}")
        # 状态机流转：处理中 -> 解析失败
        conn = sqlite3.connect(DB_FILE)
        conn.execute("UPDATE documents SET status = 'failed' WHERE doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()
        
    finally:
        # 清理占硬盘的临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==========================================
# 4. 业务接口 (API Routes)
# ==========================================

@app.post("/admin/api/upload", summary="步骤1：极速上传 (触发异步解析)")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """接收文件并立刻返回，将耗时的切片任务甩给 BackgroundTasks"""
    doc_id = str(uuid.uuid4())
    temp_path = f"./temp_{file.filename}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. 以最快速度把文件落盘
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # 2. 在数据库中注册一条 "processing (解析中)" 的记录
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents VALUES (?, ?, ?, ?)", 
                   (doc_id, file.filename, 'processing', now_str))
    conn.commit()
    conn.close()
    
    # 🚀 3. 将苦力活交给后台任务！
    background_tasks.add_task(process_document_task, doc_id, temp_path, file.filename)
    
    # 4. 接口秒回！(不让前端转圈圈)
    return {
        "status": "success", 
        "doc_id": doc_id, 
        "message": "文件已接收，后台正在拼命解析中，请稍后在看板查看状态..."
    }

@app.get("/admin/api/docs", response_model=List[DocumentResponse], summary="新增：获取全局文档看板")
def get_all_documents():
    """统筹整个知识库的资产情况，支持前端渲染全局 Dashboard"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # 连表查询：获取文档基础信息，并统计它拥有的 active 切片数量
    cursor.execute('''
        SELECT d.doc_id, d.filename, d.status, d.created_at, COUNT(c.chunk_id)
        FROM documents d
        LEFT JOIN chunks c ON d.doc_id = c.doc_id AND c.status = 'active'
        GROUP BY d.doc_id
        ORDER BY d.created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    return [
        DocumentResponse(doc_id=r[0], filename=r[1], status=r[2], created_at=r[3], chunk_count=r[4]) 
        for r in rows
    ]

# ==========================================
# 🚀 进阶：LongCat 560B 驱动的自动巡检与自愈合接口
# ==========================================
from langchain_openai import ChatOpenAI
load_dotenv()
# ==========================================
# 🧠 1. 初始化 Qwen 客户端 (老中医专属)
# ==========================================
qwen_api_key = os.getenv("api_key") 
qwen_base_url = os.getenv("base_url", "https://api.siliconflow.cn/v1")
qwen_client = OpenAI(api_key=qwen_api_key, base_url=qwen_base_url)

# ==========================================
# 👨‍⚕️ 2. 初始化 LongCat 客户端 (修复师专属)
# ==========================================
longcat_api_key = os.getenv("LONGCAT_API_KEY")
longcat_client = OpenAI(api_key=longcat_api_key, base_url="https://api.longcat.chat/v1")
MODEL_NAME = "LongCat-Flash-Thinking-2601"

# 使用 LangChain 的标准大模型接口连接硅基流动
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key=qwen_api_key,
    base_url=qwen_base_url,
    temperature=0.1 # 金融场景，温度设低一点，保证严谨性
)

# 1. 初始化 LongCat 客户端 (全局单例)
longcat_client = OpenAI(
    api_key=longcat_api_key, 
    base_url="https://api.longcat.chat/openai" # 替换为真实的 Base URL
)
MODEL_NAME = "LongCat-Flash-Thinking-2601"

@app.post("/admin/api/bad_cases/auto_heal", summary="B端：触发一键 AI 自动巡检与修复")
def trigger_auto_heal():
    if not os.path.exists(JSON_LOG_FILE):
        return {"status": "success", "message": "草稿箱为空，无需巡检。"}

    with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        
    with open(DYNAMIC_JSON_FILE, 'r', encoding='utf-8') as f:
        dynamic_archive = json.load(f)

    healed_count = 0
    manual_count = 0
    archived_count = 0
    has_changes = False
    remaining_cases = []

    # 🌟 核心架构升级：设置单次批处理上限（比如 3 条）
    # 这样能保证每次请求的总耗时绝对控制在 1-2 分钟内，彻底告别 500/Timeout
    BATCH_SIZE = 3
    current_processed = 0

    for case in cases:
        if case.get('status') == 'pending':
            
            # 🌟 熔断机制：如果这次已经处理了 3 条，剩下的直接跳过，留给下次点击
            if current_processed >= BATCH_SIZE:
                remaining_cases.append(case)
                continue

            current_processed += 1
            query = str(case['user_query'])
            ai_failed_ans = str(case.get('ai_response', ''))

            print(f"\n🔍 [系统调度] 正在为错题 '{query}' 提取真实依据...")
            ground_truth_context = fetch_local_knowledge_context(query)
            # ==========================================
            # 🕵️‍♂️ 角色 1：AI 老中医 (只看数据，不看暗号)
            # ==========================================
            diagnosis_prompt = f"""你是一个极其严厉的知识库法官。
            有人举报 AI 的回答翻车了，你需要对照【系统真实资料】来核实这起冤假错案。
            
            🗣️ 【用户提问】：“{query}”
            🤖 【AI 原回答】：“{ai_failed_ans}”
            
            📚 【系统真实资料 (Ground Truth)】：
            {ground_truth_context}
            
            🚨【判案法则 - 请对照资料严格执行】：
            1. **FALSE_ALARM (冤假错案)**：仔细对比【AI 原回答】和【系统真实资料】。如果 AI 的核心数据、结论与真实资料**完全一致**，说明 AI 答对了，是用户在瞎点踩！必须判定为 FALSE_ALARM。
            2. **STATIC (AI 漏答/幻觉)**：如果【系统真实资料】里明明有答案，但【AI 原回答】却说找不到，或者 AI 回答的数据与真实资料**存在冲突（幻觉）**，判定为 STATIC，必须打回重做。
            3. **DYNAMIC (时效问题)**：如果问的是今天的股价/新闻，且系统资料里确实没有，判定为 DYNAMIC。
            
            严格按此 JSON 格式输出：
            {{
                "type": "FALSE_ALARM 或 STATIC 或 DYNAMIC", 
                "reason": "详细说明 AI 的回答与真实资料是否一致，以及判定的理由"
            }}
            """
            
            try:
                # 依然让 Qwen2.5 来当老中医
                diag_response = qwen_client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[{"role": "user", "content": diagnosis_prompt}],
                    temperature=0.1
                )
                raw_content = diag_response.choices[0].message.content or ""
                json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                
                if json_match:
                    diagnosis_result = json.loads(json_match.group(0))
                else:
                    raise ValueError("未返回 JSON")
                    
                case_type = diagnosis_result.get("type", "STATIC")
                fail_reason = diagnosis_result.get("reason", "未知错误")
                # 🌟 新增这一行：安全地获取 tool_used，如果没有就给个默认值
                tool_used = diagnosis_result.get("tool_used", "知识库检索")
                
            except Exception as e:
                print(f"❌ [老中医判案崩溃] {e}")
                case_type = "STATIC"
                fail_reason = "判案异常"
                tool_used = "未知(解析异常)"

            # ------------------------------------------
            # 🗂️ 动态数据归档
            # ------------------------------------------
            if case_type == "DYNAMIC":
                case['status'] = "ignored_dynamic"
                case['admin_note'] = f"🤖 AI诊断：[动态] 工具 `{tool_used}`。死因: {fail_reason}"
                dynamic_archive.append(case)
                archived_count += 1
                has_changes = True
                continue
            # ------------------------------------------
            # 🗂️ 动态归档与误报过滤
            # ------------------------------------------
            if case_type == "FALSE_ALARM":
                case['status'] = "ignored" # 标记为垃圾废弃
                case['admin_note'] = f"🤖 AI诊断：[用户误踩] 原回答无误。判定依据: {fail_reason}"
                has_changes = True
                continue

            # ==========================================
            # 👨‍⚕️ 角色 2：AI 修复师 (真·开卷考试版)
            # ==========================================
            # 🚀 删掉 mock 数据，调用我们在文件开头写的检索探针！
            # 让它带着用户的 query，去你真正的 Chroma/FAISS 库里捞切片
            rag_context = fetch_local_knowledge_context(query)
            
            heal_prompt = f"""你是一位极其严谨的首席金融分析师。你需要对用户的历史提问进行“完美修正”。

            📚【以下是系统从内部财报数据库中为你检索到的可靠依据】：
            {rag_context}
            
            🗣️【用户提问】：“{query}”
            
            请根据上述提供的【可靠依据】，直接回答用户的问题。如果依据中提到了具体的文档来源（如《2024年年度报告》），请在回答中自然地引用。
            
            ⚠️【最高红线警告】：
            1. 你的回答**必须且只能**基于我为你提供的依据！
            2. 如果我提供的依据里没有具体数据，或者数据不足以回答该问题，请直接输出：【呼叫人工】。
            """
            
            try:
                # 🌟 依然使用 longcat_client 进行深度推演
                heal_response = longcat_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": heal_prompt}],
                    temperature=0.3, 
                    max_tokens=4096
                )
                full_content = heal_response.choices[0].message.content or ""
                if "</think>" in full_content:
                    clean_ans = full_content.split("</think>")[-1].strip()
                else:
                    # 传统的正则替换
                    clean_ans = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
                
                if not clean_ans or "【呼叫人工】" in clean_ans:
                    golden_ans = "【呼叫人工】"
                else:
                    golden_ans = f"【AI 自动纠偏】{clean_ans}"
                    
            except Exception as e:
                print(f"\n❌ [修复师崩溃] 案例 {case['case_id']} 失败原因: {str(e)}")
                golden_ans = "【呼叫人工】"

            # ------------------------------------------
            # ⚖️ 兜底分流机制
            # ------------------------------------------
            if "【呼叫人工】" in golden_ans:
                case['status'] = "manual_review"
                case['admin_note'] = f"🤖 LongCat巡检：无绝对把握。原涉事工具: `{tool_used}`。转人工。"
                # 清空答案字段
                case['corrected_answer'] = "" 
                manual_count += 1
            else:
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS golden_qa (
                        id TEXT PRIMARY KEY,
                        original_query TEXT,
                        corrected_answer TEXT,
                        source_case_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute(
                    "INSERT INTO golden_qa (id, original_query, corrected_answer, source_case_id) VALUES (?, ?, ?, ?)",
                    (f"qa_{uuid.uuid4().hex[:8]}", query, golden_ans, case['case_id'])
                )
                conn.commit()
                conn.close()

                case['status'] = "auto_fixed"
                case['admin_note'] = f"🤖 LongCat巡检：已自动修复。原涉事工具: `{tool_used}`。"
                
                # 🌟 核心新增：把黄金答案写回 JSON，让前端能看见！
                case['corrected_answer'] = golden_ans  
                
                healed_count += 1
            
            has_changes = True
            remaining_cases.append(case)
        else:
            remaining_cases.append(case)

    if has_changes:
        with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(remaining_cases, f, ensure_ascii=False, indent=2)
            
        with open(DYNAMIC_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(dynamic_archive, f, ensure_ascii=False, indent=2)

    # 🌟 为了防止前端再出现“✅ 报错”，我们把 status 统一设为 success，具体结果在 message 里体现
    return {
        "status": "success", 
        "message": f"巡检完毕！LongCat 修复 {healed_count} 条，转人工 {manual_count} 条，归档 {archived_count} 条。"
    }

# --- 下方保留原有的 chunks 获取、更新、删除、发布接口 ---
@app.get("/admin/api/docs/{doc_id}/chunks", response_model=List[ChunkResponse])
def get_chunks(doc_id: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,))
    rows = cursor.fetchall()
    conn.close()
    return [ChunkResponse(chunk_id=r[0], doc_id=r[1], text_content=r[2], chunk_index=r[3], status=r[4]) for r in rows]

@app.put("/admin/api/chunks/{chunk_id}")
def update_chunk(chunk_id: str, params: ChunkUpdateParams):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE chunks SET text_content = ? WHERE chunk_id = ?", (params.new_text, chunk_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/admin/api/chunks/{chunk_id}")
def delete_chunk(chunk_id: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE chunks SET status = 'deleted' WHERE chunk_id = ?", (chunk_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/admin/api/docs/{doc_id}/publish")
def publish_document(doc_id: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE documents SET status = 'published' WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "已成功发布入库！"}

@app.delete("/admin/api/bad_cases/{case_id}", summary="B端专用：物理删除无价值的脏数据")
def delete_bad_case(case_id: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # 物理删除这条数据
    cursor.execute("DELETE FROM bad_cases WHERE case_id = ?", (case_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"脏数据 {case_id} 已彻底清除"}

# 在 admin_backend.py 底部追加：
class StatusUpdateRequest(BaseModel):
    status: str  # 接收 'fixed' 或 'ignored'

@app.put("/admin/api/bad_cases/{case_id}/status", summary="B端专用：更新 Bad Case 处理状态")
def update_bad_case_status(case_id: str, req: StatusUpdateRequest):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE bad_cases SET status = ? WHERE case_id = ?", (req.status, case_id))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"状态已更新为 {req.status}"}


if __name__ == "__main__":
    print("🚀 Taday 后端 API v2 (支持异步与看板) 运行在 8001 端口...")
    uvicorn.run(app, host="127.0.0.1", port=8001)