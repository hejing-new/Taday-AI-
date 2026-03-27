import sys
import os
# 强行把当前文件所在目录的“上一级目录”（即根目录）加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb
import random
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset

from ragas import evaluate
# ⬇️ 核心升级：使用 v0.2 的最新指标名
from ragas.metrics import (
    Faithfulness, 
    # ResponseRelevancy, 
    LLMContextPrecisionWithReference,
    LLMContextRecall
)

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from tools.rag_tool import report_query_engine

load_dotenv()
api_key = os.getenv("api_key")
base_url = os.getenv("base_url", "https://api.siliconflow.cn/v1")

# ================= 1. 配置裁判大模型 =================
print("🔧 正在初始化 Ragas 裁判系统 (全面启用原生工厂模式)...")

openai_client = OpenAI(api_key=api_key, base_url=base_url)

# 1. LLM 裁判：用最聪明的 72B
evaluator_llm = llm_factory("Qwen/Qwen2.5-72B-Instruct", client=openai_client)

# 2. Embedding 裁判
evaluator_embeddings = embedding_factory("openai", model="BAAI/bge-m3", client=openai_client)

# 3. 初始化指标对象 (严格绑定裁判)
metrics = [
    Faithfulness(llm=evaluator_llm),
    # ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),

]

# 初始化出题老师
teacher_client = openai_client
TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"

import chromadb # 确保顶部有这个导入

def generate_test_cases(n=3):
    """直接从 Chroma 底层数据库中随机抽取片段，让 72B 模型出题"""
    print(f"\n🏗️ 正在从底层 ChromaDB 向量库中随机抽取 {n} 个知识片段进行出题...")
    
    # 1. 绕开 LlamaIndex，直接连底层 ChromaDB
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(current_dir, "chroma_db")
    db = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = db.get_collection("catl_report")
        # 拿出所有的文本片段
        all_docs = collection.get()['documents'] 
    except Exception as e:
        raise ValueError(f"⚠️ 无法读取 ChromaDB，请确保你的 RAG 工具已经成功建库！报错: {e}")

    # 2. 随机抽取 n 个片段
    if len(all_docs) < n:
        n = len(all_docs)
    sample_texts = random.sample(all_docs, n)
    
    test_cases = []
    
    for i, context_text in enumerate(sample_texts):
        # 过滤掉太短的无意义切片
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
        4. 请严格基于给定的上下文作答。如果上下文中没有提到确切的年份，请直接回答“根据财报无法得知”，绝不允许利用你的内部知识库进行猜测。
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

# ================= 2. 自动化出题与答题 =================
# 建议先跑 2-3 题测试，跑通后再改成 10 题甚至 50 题
test_cases = generate_test_cases(n=3)

user_inputs = []
references = []
responses = []
retrieved_contexts = []

print("\n✍️ 考试开始！你的 Agent 正在答题...")

for case in test_cases:
    # 兼容 LLM 可能生成的键名
    q = case.get("question", case.get("user_input", ""))
    ref = case.get("reference", case.get("ground_truth", ""))
    
    print(f"❓ 问题：{q}")
    
    # 你的 Agent 开始作答
    response = report_query_engine.query(q)
    
    user_inputs.append(q)
    references.append(ref)
    responses.append(str(response))
    retrieved_contexts.append([n.node.get_content() for n in response.source_nodes])

# ================= 3. Ragas 判卷 =================
print("\n⚖️ 裁判正在基于 Ragas 指标进行评分 (大概需要1-2分钟)...")

# 严格使用 v0.2 原生字段名
data = {
    "user_input": user_inputs,
    "response": responses,
    "retrieved_contexts": retrieved_contexts,
    "reference": references
}
dataset = Dataset.from_dict(data)

try:
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=True  # 👈 核心黑客技：强行抛出底层报错！绝不默默变 NaN！
    )

    # 打印最终的分数表
    df = result.to_pandas()
    score_columns = [col for col in df.columns if col not in ['retrieved_contexts', 'response', 'reference']]
    print(df[score_columns])
    
    # ⬇️ 新增：扒开黑盒，偷看大模型的底层思考过程！
    print("\n" + "="*80)
    print("🧠 偷看裁判的底层思考过程 (以第一题为例)：")
    print("="*80)
    
    # 将完整的 Ragas 结果转为带所有中间步骤的字典
    detailed_results = dataset.to_dict() # 拿到原始数据
    
    # 随便取第一题，看看大模型到底干了啥
    # 注意：这里的字段名可能会因为 Ragas 版本微调有差异，通常它会把思考过程附加在原始 dataset 里
    print(f"原问题: {detailed_results['user_input'][0]}")
    
    # 打印所有的列名，你会发现里面多了一些隐藏列（比如 faithfulness_statements 等）
    print(f"\n隐藏的中间过程列: {list(df.columns)}")
    
    # 我们把第一题的所有评测明细（包括报错或生成的中间变量）打印出来
    first_row = df.iloc[0]
    for col in df.columns:
        if col not in score_columns and col not in ['retrieved_contexts', 'response', 'reference']:
            print(f"\n🔍 {col}: \n{first_row[col]}")
            
    
except Exception as e:
    print(f"\n❌ 评测过程中发生错误: {e}")