"""
Taday 金融智能体 — 自动化 RAG 评测脚本 (合并版)

基于 Ragas v0.2 指标，从 ChromaDB 随机抽题 → 让被测 Agent 答题 → 用大模型裁判判卷。
支持：动态出题、自定义题目数量、结果保存到 JSON、详细中间过程输出。

使用方式：
    python auto_eval.py              # 默认 3 题
    python auto_eval.py --n 5        # 出 5 题
    python auto_eval.py --save       # 保存结果到 JSON
"""
import sys
import os
import io
import argparse

# UTF-8 环境防护
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 路径配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import random
import json
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from tools.rag_tool import report_query_engine

load_dotenv()
api_key = os.getenv("api_key")
base_url = os.getenv("base_url", "https://api.siliconflow.cn/v1")

# ================= 1. 配置裁判大模型 =================
print("🔧 正在初始化 Ragas 裁判系统 (v0.2 纯血架构)...")

openai_client = OpenAI(api_key=api_key, base_url=base_url)

# LLM 裁判
evaluator_llm = llm_factory("Qwen/Qwen2.5-72B-Instruct", client=openai_client)

# Embedding 裁判
evaluator_embeddings = embedding_factory("openai", model="BAAI/bge-m3", client=openai_client)

# 初始化指标对象 (严格绑定裁判)
metrics = [
    Faithfulness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),
]

# 初始化出题老师
teacher_client = openai_client
TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def generate_test_cases(n=3):
    """直接从 Chroma 底层数据库中随机抽取片段，让 72B 模型出题"""
    print(f"\n🏗️ 正在从底层 ChromaDB 向量库中随机抽取 {n} 个知识片段进行出题...")

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(current_dir, "chroma_db")
    db = chromadb.PersistentClient(path=db_path)

    try:
        collection = db.get_collection("catl_report")
        all_docs = collection.get()['documents']
    except Exception as e:
        raise ValueError(f"⚠️ 无法读取 ChromaDB，请确保你的 RAG 工具已经成功建库！报错: {e}")

    if len(all_docs) < n:
        n = len(all_docs)
    sample_texts = random.sample(all_docs, n)

    test_cases = []

    for i, context_text in enumerate(sample_texts):
        if len(context_text) < 50:
            continue

        print(f"📝 老师正在阅读第 {i+1} 个片段并出题...")

        prompt = f"""你是一位严厉的金融考试官。请根据下面这段【宁德时代财报】的原文，出一道具体的、有难度的问答题，并给出标准答案。

        原文内容：
        {context_text}

        要求：
        1. 问题必须能从原文中找到依据，不要空泛。
        2. 答案必须极其精准（包含具体数字或术语）。
        3. 请严格按照下面的 JSON 格式返回，不要有任何废话：
        4. 请严格基于给定的上下文作答。如果上下文中没有提到确切的年份，请直接回答"根据财报无法得知"，绝不允许利用你的内部知识库进行猜测。
        {{"question": "具体问题", "reference": "标准答案"}}
        """

        try:
            response = teacher_client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            res = json.loads(response.choices[0].message.content)
            res["context"] = context_text
            test_cases.append(res)
        except Exception as e:
            print(f"⚠️ 第 {i+1} 题生成失败，跳过。原因: {e}")

    return test_cases


def run_evaluation(n=3, save_results=True, show_details=True):
    """执行完整的评测流程"""
    # ================= 2. 自动化出题与答题 =================
    test_cases = generate_test_cases(n=n)

    if not test_cases:
        print("❌ 未生成任何测试题，退出。")
        return

    user_inputs = []
    references = []
    responses = []
    retrieved_contexts = []

    print("\n✍️ 考试开始！你的 Agent 正在答题...")

    for case in test_cases:
        q = case.get("question", case.get("user_input", ""))
        ref = case.get("reference", case.get("ground_truth", ""))

        print(f"❓ 问题：{q}")

        response = report_query_engine.query(q)

        user_inputs.append(q)
        references.append(ref)
        responses.append(str(response))
        retrieved_contexts.append([n.node.get_content() for n in response.source_nodes])

    # ================= 3. Ragas 判卷 =================
    print("\n⚖️ 裁判正在基于 Ragas 指标进行评分 (大概需要1-2分钟)...")

    data = {
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references,
    }
    dataset = Dataset.from_dict(data)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            raise_exceptions=True  # 强行抛出底层报错，绝不默默变 NaN
        )

        # ================= 4. 打印成绩单 =================
        print("\n" + "=" * 80)
        print("📊 Taday 金融智能体 - 自动化动态质检报告")
        print("=" * 80)

        df = result.to_pandas()
        score_columns = [col for col in df.columns if col not in ['retrieved_contexts', 'response', 'reference']]
        print(df[score_columns])
        print("-" * 80)

        # ================= 5. 输出详细中间过程 =================
        if show_details:
            print("\n" + "=" * 80)
            print("🧠 偷看裁判的底层思考过程 (以第一题为例)：")
            print("=" * 80)

            detailed_results = dataset.to_dict()
            print(f"原问题: {detailed_results['user_input'][0]}")
            print(f"\n隐藏的中间过程列: {list(df.columns)}")

            first_row = df.iloc[0]
            for col in df.columns:
                if col not in score_columns and col not in ['retrieved_contexts', 'response', 'reference']:
                    print(f"\n🔍 {col}: \n{first_row[col]}")

        # ================= 6. 保存结果到 JSON =================
        if save_results:
            output_json_path = "ragas_evaluation_results.json"
            df.to_json(output_json_path, orient='records', force_ascii=False, indent=4)
            print(f"\n✅ 评测结果已保存至: 📂 {os.path.abspath(output_json_path)}")

        print("\n" + "=" * 80)
        print("✅ 评测大功告成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 评测过程中发生错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taday RAG 自动化评测")
    parser.add_argument("--n", type=int, default=3, help="出题数量 (默认: 3)")
    parser.add_argument("--save", action="store_true", help="保存结果到 JSON")
    parser.add_argument("--no-details", action="store_true", help="不显示中间过程")
    args = parser.parse_args()

    run_evaluation(
        n=args.n,
        save_results=args.save,
        show_details=not args.no_details,
    )
