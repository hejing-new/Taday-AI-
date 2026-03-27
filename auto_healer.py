import json
import sqlite3
import os
import uuid
import time
# 假设你已经有了一个可以调用的 llm 实例，这里根据你的实际项目路径导入
# from core.llm_config import llm 

JSON_LOG_FILE = "bad_cases_staging.json"
DB_FILE = "knowledge_draft.db"

def classify_error_type(query: str, error_ans: str) -> str:
    """
    步骤 1：智能分拣。使用轻量级 Prompt 让大模型判断错误类型。
    """
    prompt = f"""你是一个 AI 系统的异常诊断专家。
    用户提问：{query}
    AI原错误回答：{error_ans}
    
    请判断这个提问属于哪种类型，只允许输出以下两个词之一，不要有任何多余字符：
    STATIC (静态知识，例如财报数据、公司战略、历史常识)
    DYNAMIC (动态时效查询，例如今天的新闻、当前的股价、实时热点)
    """
    
    print(f"  [🔍 分拣器] 正在分析提问属性...")
    # 模拟 LLM 调用，你需要替换为真实的 llm.invoke(prompt).content
    # response = llm.invoke(prompt).content.strip().upper()
    
    # 这里为了让你直接跑通，做一个简单的规则模拟（真实情况请用大模型）
    if "新闻" in query or "今天" in query or "股价" in query:
        return "DYNAMIC"
    return "STATIC"

def heal_static_knowledge(query: str) -> str:
    """
    步骤 2：静态知识深度修复。调用强大的“阅卷老师”模型得出绝对正确的答案。
    """
    print(f"  [📚 老师出马] 正在查阅底层资料，重写标准答案...")
    prompt = f"""你是一位极其严谨的金融审核专家。
    请针对问题：“{query}”，给出绝对准确、客观的正确答案。
    如果没有把握，请回答“需要人工核实”。
    """
    # 模拟老师重写答案，替换为真实的 llm 调用
    # return llm.invoke(prompt).content.strip()
    return f"【AI自动纠偏结论】经过后台深度比对资料，正确答案为：... (模拟数据)"

def run_auto_patrol():
    """主控流水线：遍历草稿箱，执行分拣与修复"""
    print("========================================")
    print("🤖 Taday 金融大脑 - 自动巡检治愈系统启动")
    print("========================================\n")
    
    if not os.path.exists(JSON_LOG_FILE):
        print("✅ 草稿箱为空，无需巡检。")
        return

    with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        
    has_changes = False
    for case in cases:
        if case.get('status') == 'pending':
            print(f"\n🚨 发现待处理 Bad Case ID: {case['case_id']}")
            print(f"   提问: {case['user_query']}")
            
            # 1. 智能分拣
            case_type = classify_error_type(case['user_query'], case['ai_response'])
            
            # 2. 动态路由分发
            if case_type == "DYNAMIC":
                print(f"   ⚠️ 判定为【动态时效问题】。拒绝写入数据库！仅标记存档。")
                case['status'] = "ignored_dynamic"
                case['admin_note'] = "自动巡检：此类为实时工具调用失败，无需固化为静态知识。"
                
            elif case_type == "STATIC":
                print(f"   ✅ 判定为【静态知识缺失】。启动自动修复入库流程...")
                golden_ans = heal_static_knowledge(case['user_query'])
                
                # 🌟 修复：连接正确的数据库，并加上防御性建表
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                
                # 暴力防坑：不管表在不在，先尝试建一次，确保100%安全
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS golden_qa (
                        id TEXT PRIMARY KEY,
                        original_query TEXT,
                        corrected_answer TEXT,
                        source_case_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 正常执行插入
                cursor.execute(
                    "INSERT INTO golden_qa (id, original_query, corrected_answer, source_case_id) VALUES (?, ?, ?, ?)",
                    (f"qa_{uuid.uuid4().hex[:8]}", case['user_query'], golden_ans, case['case_id'])
                )
                conn.commit()
                conn.close()
                
                print(f"   💾 修复完毕！黄金答案已存入 DB。")
                case['status'] = "auto_fixed"
            
            has_changes = True
            time.sleep(1) # 模拟处理时间
            
    # 3. 统一将更新后的状态写回 JSON
    if has_changes:
        with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        print("\n🎉 本轮自动巡检结束，草稿箱状态已更新。")
    else:
        print("✅ 本轮巡检未发现需要处理的新数据。")

if __name__ == "__main__":
    run_auto_patrol()