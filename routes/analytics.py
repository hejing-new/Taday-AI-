"""
运营数据 / BI 看板相关路由
"""
import os
import uuid
import sqlite3
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from config import DB_FILE, ANALYTICS_PAGE_SIZE
from logger import logger

router = APIRouter()

class SearchLogRequest(BaseModel):
    user_query: str
    session_id: str
    latency: float


# ==========================================
# C 端接口（无需认证）
# ==========================================

@router.post("/api/v1/log_search", summary="C端专用：记录一次搜索行为")
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


# ==========================================
# B 端接口（需认证）
# ==========================================

@router.get("/admin/api/analytics", summary="B端专用：拉取运营大盘数据")
def get_analytics(page: int = 1, page_size: int = None):
    """
    获取 BI 大盘数据，支持分页。

    参数:
        page: 页码，从 1 开始
        page_size: 每页条数，默认使用 ANALYTICS_PAGE_SIZE (50)
    """
    if page_size is None:
        page_size = ANALYTICS_PAGE_SIZE

    offset = (page - 1) * page_size

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 总记录数
    cursor.execute("SELECT COUNT(*) FROM search_logs")
    total_count = cursor.fetchone()[0]

    # 分页拉取日志
    cursor.execute(
        "SELECT user_query, session_id, latency, created_at FROM search_logs ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (page_size, offset)
    )
    logs = cursor.fetchall()
    conn.close()

    total_searches = total_count
    avg_latency = sum([r[2] for r in logs]) / total_searches if total_searches > 0 else 0

    return {
        "metrics": {
            "total_searches": total_searches,
            "avg_latency": round(avg_latency, 2),
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size if page_size > 0 else 0
        },
        "logs": [{"query": r[0], "session": r[1], "latency": r[2], "time": r[3]} for r in logs]
    }
