"""
AI 自动巡检与自愈合路由（B 端）
"""
import os
import json
import re
import uuid
import sqlite3
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
import requests
from config import (
    DB_FILE, API_KEY, BASE_URL, CHAT_MODEL, JUDGE_MODEL,
    LONGCAT_API_KEY, LONGCAT_BASE_URL, HEAL_MODEL
)
from logger import logger
from utils.json_store import (
    load_bad_cases, save_bad_cases,
    load_dynamic_archive, save_dynamic_archive,
)

router = APIRouter()

# 数据模型
class SearchLogRequest(BaseModel):
    user_query: str
    session_id: str
    latency: float


def fetch_local_knowledge_context(query: str) -> str:
    """通过 HTTP 向前台主业务借答案"""
    try:
        from config import API_URL
        url = f"{API_URL}/api/v1/chat"
        payload = {"query": f"请仅依靠内部知识库查证：{query}"}
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("response", resp.text)
    except Exception as e:
        logger.error(f"知识库探针异常: {e}")
    return "未找到资料"


def _init_qwen_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _init_longcat_client():
    return OpenAI(api_key=LONGCAT_API_KEY, base_url=LONGCAT_BASE_URL)


# ==========================================
# AI 自动巡检与修复
# ==========================================

@router.post("/admin/api/bad_cases/auto_heal", summary="B端：触发一键 AI 自动巡检与修复")
def trigger_auto_heal():
    if not os.path.exists(JSON_LOG_FILE):
        return {"status": "success", "message": "草稿箱为空，无需巡检。"}

    cases = load_bad_cases()
    dynamic_archive = load_dynamic_archive()

    healed_count = 0
    manual_count = 0
    archived_count = 0
    has_changes = False
    remaining_cases = []

    BATCH_SIZE = 3
    current_processed = 0

    qwen_client = _init_qwen_client()
    longcat_client = _init_longcat_client()

    for case in cases:
        if case.get('status') == 'pending':

            if current_processed >= BATCH_SIZE:
                remaining_cases.append(case)
                continue

            current_processed += 1
            query = str(case['user_query'])
            ai_failed_ans = str(case.get('ai_response', ''))

            logger.info(f"正在为错题提取真实依据: {query}")
            ground_truth_context = fetch_local_knowledge_context(query)

            # 角色 1：AI 老中医 — 判定错误类型
            diagnosis_prompt = f"""你是一个极其严厉的知识库法官。
            有人举报 AI 的回答翻车了，你需要对照【系统真实资料】来核实。

            🗣️ 【用户提问】："{query}"
            🤖 【AI 原回答】："{ai_failed_ans}"

            📚 【系统真实资料】：
            {ground_truth_context}

            🚨【判案法则】：
            1. FALSE_ALARM：AI 回答与真实资料完全一致 → 用户误踩
            2. STATIC：真实资料有答案但 AI 答错/幻觉 → 需修复
            3. DYNAMIC：问的是实时数据且资料中没有 → 归档

            严格按此 JSON 输出：
            {{"type": "FALSE_ALARM 或 STATIC 或 DYNAMIC", "reason": "详细理由"}}
            """

            try:
                diag_response = qwen_client.chat.completions.create(
                    model=CHAT_MODEL,
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

            except Exception as e:
                logger.error(f"老中医判案崩溃: {e}")
                case_type = "STATIC"
                fail_reason = "判案异常"

            # 动态数据归档
            if case_type == "DYNAMIC":
                case['status'] = "ignored_dynamic"
                case['admin_note'] = f"🤖 AI诊断：[动态] 死因: {fail_reason}"
                dynamic_archive.append(case)
                archived_count += 1
                has_changes = True
                continue

            # 误报过滤
            if case_type == "FALSE_ALARM":
                case['status'] = "ignored"
                case['admin_note'] = f"🤖 AI诊断：[用户误踩] 判定依据: {fail_reason}"
                has_changes = True
                continue

            # 角色 2：AI 修复师（复用第一次查询结果，避免重复 HTTP 请求）
            rag_context = ground_truth_context

            heal_prompt = f"""你是一位极其严谨的首席金融分析师。

            📚【可靠依据】：
            {rag_context}

            🗣️【用户提问】："{query}"

            请根据上述依据直接回答。如果依据中没有，请输出：【呼叫人工】

            ⚠️【红线】：必须且只能基于提供的依据！
            """

            try:
                heal_response = longcat_client.chat.completions.create(
                    model=HEAL_MODEL,
                    messages=[{"role": "user", "content": heal_prompt}],
                    temperature=0.3,
                    max_tokens=4096
                )
                full_content = heal_response.choices[0].message.content or ""
                if "</think>" in full_content:
                    clean_ans = full_content.split("</think>")[-1].strip()
                else:
                    clean_ans = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()

                if not clean_ans or "【呼叫人工】" in clean_ans:
                    golden_ans = "【呼叫人工】"
                else:
                    golden_ans = f"【AI 自动纠偏】{clean_ans}"

            except Exception as e:
                logger.error(f"修复师崩溃，案例 {case['case_id']} 失败原因: {e}")
                golden_ans = "【呼叫人工】"

            # 兜底分流
            if "【呼叫人工】" in golden_ans:
                case['status'] = "manual_review"
                case['admin_note'] = f"🤖 LongCat巡检：无绝对把握。转人工。"
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
                case['admin_note'] = f"🤖 LongCat巡检：已自动修复。"
                case['corrected_answer'] = golden_ans
                healed_count += 1

            has_changes = True
            remaining_cases.append(case)
        else:
            remaining_cases.append(case)

    if has_changes:
        save_bad_cases(remaining_cases)
        save_dynamic_archive(dynamic_archive)

    return {
        "status": "success",
        "message": f"巡检完毕！LongCat 修复 {healed_count} 条，转人工 {manual_count} 条，归档 {archived_count} 条。"
    }
