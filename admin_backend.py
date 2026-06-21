"""
Taday 知识库管理后台 API

路由已拆分到 routes/ 子模块，本文件只保留：
  - FastAPI app 实例
  - 数据库初始化
  - 路由注册
  - 公共数据模型
"""
import os
import sqlite3
import uvicorn
from fastapi import FastAPI, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List
import secrets

from config import (
    ADMIN_USER, ADMIN_PASS, DB_FILE, CHROMA_DB_PATH, DATA_PATH,
    TEMP_STORAGE_PATH, ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE,
    ANALYTICS_PAGE_SIZE, API_KEY, BASE_URL, CHAT_MODEL, JUDGE_MODEL,
    LONGCAT_API_KEY, HEAL_MODEL, PORT_ADMIN_API
)
from logger import logger

# ================= 安全认证 =================
security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """验证管理员身份"""
    is_user_correct = secrets.compare_digest(credentials.username, ADMIN_USER)
    is_pass_correct = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (is_user_correct and is_pass_correct):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未授权访问：管理员凭据无效",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ================= FastAPI 实例 =================
app = FastAPI(title="Taday 知识库管理后台 API", version="2.1.0")

# ================= 公共数据模型 =================
class FeedbackRequest(BaseModel):
    user_query: str
    ai_response: str

class CorrectionRequest(BaseModel):
    correct_answer: str

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

class StatusUpdateRequest(BaseModel):
    status: str

# ================= 数据库初始化 =================
def init_db():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT,
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bad_cases (
            case_id TEXT PRIMARY KEY,
            user_query TEXT,
            ai_response TEXT,
            status TEXT,
            created_at TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_logs (
            log_id TEXT PRIMARY KEY,
            user_query TEXT,
            session_id TEXT,
            latency REAL,
            created_at TEXT
        )
    ''')
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

# ================= 路由注册 =================
from routes.feedback import router as feedback_router
from routes.documents import router as documents_router
from routes.analytics import router as analytics_router
from routes.auto_heal import router as auto_heal_router

# C 端路由（无需认证）
app.include_router(feedback_router)
app.include_router(analytics_router)

# B 端路由（需认证）
app.include_router(
    documents_router,
    dependencies=[Depends(verify_admin)]
)
app.include_router(
    auto_heal_router,
    dependencies=[Depends(verify_admin)]
)

# 为 feedback_router 和 analytics_router 中的 B 端路由单独添加认证
# 由于 feedback_router 混用了 C/B 端路由，需要手动覆盖
from fastapi import APIRouter
_admin_only_router = APIRouter(dependencies=[Depends(verify_admin)])

# 重新注册 B 端专属路由（覆盖 feedback_router 中的同名路由）
@_admin_only_router.get("/admin/api/bad_cases", summary="B端：读取 JSON 暂存数据")
def _get_cases_from_json():
    from routes.feedback import get_cases_from_json
    return get_cases_from_json()

@_admin_only_router.post("/admin/api/bad_cases/{case_id}/fix", summary="B端：人工质检完毕，入库并更新状态")
def _fix_bad_case(case_id: str, req: CorrectionRequest):
    from routes.feedback import fix_bad_case
    return fix_bad_case(case_id, req)

@_admin_only_router.delete("/admin/api/bad_cases/{case_id}", summary="B端：物理删除脏数据")
def _delete_bad_case(case_id: str):
    from routes.feedback import delete_bad_case
    return delete_bad_case(case_id)

@_admin_only_router.put("/admin/api/bad_cases/{case_id}/status", summary="B端：更新处理状态")
def _update_bad_case_status(case_id: str, req: StatusUpdateRequest):
    from routes.feedback import update_bad_case_status
    return update_bad_case_status(case_id, req)

app.include_router(_admin_only_router)


# ================= 启动入口 =================
if __name__ == "__main__":
    logger.info(f"Taday 后端 API v2.1 启动，端口 {PORT_ADMIN_API}")
    uvicorn.run(app, host="127.0.0.1", port=PORT_ADMIN_API)
