"""
反馈 / Bad Case 相关路由（C 端 + B 端）
"""
import os
import json
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from config import DB_FILE
from logger import logger

router = APIRouter()

# 文件路径
JSON_LOG_FILE = "bad_cases_staging.json"
DYNAMIC_JSON_FILE = "dynamic_cases_archive.json"

# 数据模型
class FeedbackRequest(BaseModel):
    user_query: str
    ai_response: str

class CorrectionRequest(BaseModel):
    correct_answer: str


def _ensure_json_files():
    """确保 JSON 文件存在"""
    if not os.path.exists(JSON_LOG_FILE):
        with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
    if not os.path.exists(DYNAMIC_JSON_FILE):
        with open(DYNAMIC_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)


# ==========================================
# C 端接口（无需认证）
# ==========================================

@router.post("/api/v1/feedback", summary="C端：记录 Bad Case 到 JSON 暂存区")
def add_feedback_to_json(req: FeedbackRequest):
    try:
        _ensure_json_files()
        case_id = f"bc_{uuid.uuid4().hex[:8]}"
        new_case = {
            "case_id": case_id,
            "user_query": req.user_query,
            "ai_response": req.ai_response,
            "status": "pending",
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
                logger.warning(f"JSON 恢复失败: {e}")
                cases = []

        # 幂等性去重
        for existing_case in cases:
            if (existing_case.get('user_query') == req.user_query and
                existing_case.get('ai_response') == req.ai_response and
                existing_case.get('status') == 'pending'):
                logger.info("检测到重复点踩，已静默去重")
                return {"status": "success", "message": "already_exists", "case_id": existing_case.get('case_id')}

        cases.append(new_case)

        with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)

        return {"status": "success", "case_id": case_id}

    except Exception as e:
        logger.error(f"点踩写入失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/feedback/cancel", summary="C端：撤销点踩，从 JSON 移除记录")
def cancel_feedback_in_json(req: FeedbackRequest):
    try:
        if not os.path.exists(JSON_LOG_FILE):
            return {"status": "success"}

        with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
            cases = json.load(f)

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
        logger.error(f"撤销点踩失败: {e}")
        return {"status": "error", "message": str(e)}


# ==========================================
# B 端接口（需认证，在 main 文件中通过 dependencies 注入）
# ==========================================

@router.get("/admin/api/bad_cases", summary="B端：读取 JSON 暂存数据")
def get_cases_from_json():
    if not os.path.exists(JSON_LOG_FILE):
        return []

    try:
        with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            cases = json.loads(content)
        return sorted(cases, key=lambda x: x.get('created_at', ''), reverse=True)

    except Exception as e:
        logger.error(f"读取草稿箱失败: {e}")
        return []


@router.post("/admin/api/bad_cases/{case_id}/fix", summary="B端：人工质检完毕，入库并更新状态")
def fix_bad_case(case_id: str, req: CorrectionRequest):
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

    with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    # 写入黄金答案库
    import sqlite3
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO golden_qa (id, original_query, corrected_answer, source_case_id) VALUES (?, ?, ?, ?)",
        (f"qa_{uuid.uuid4().hex[:8]}", target_case['user_query'], req.correct_answer, case_id)
    )
    conn.commit()
    conn.close()

    return {"status": "success", "message": "质检通过！已归档至黄金问答库(DB)"}


@router.delete("/admin/api/bad_cases/{case_id}", summary="B端专用：物理删除无价值的脏数据")
def delete_bad_case(case_id: str):
    import sqlite3
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM bad_cases WHERE case_id = ?", (case_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"脏数据 {case_id} 已彻底清除"}


class StatusUpdateRequest(BaseModel):
    status: str

@router.put("/admin/api/bad_cases/{case_id}/status", summary="B端专用：更新 Bad Case 处理状态")
def update_bad_case_status(case_id: str, req: StatusUpdateRequest):
    import sqlite3
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE bad_cases SET status = ? WHERE case_id = ?", (req.status, case_id))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"状态已更新为 {req.status}"}
