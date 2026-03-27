import sys
import os
# 强行把当前文件所在目录的“上一级目录”加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate

# ⬇️ 终极护甲：全部使用 Ragas v0.2 纯血版的新指标！
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,                  # 取代了旧版的 AnswerRelevancy
    LLMContextPrecisionWithReference,   # 取代了旧版的 context_precision
    LLMContextRecall                    # 取代了旧版的 context_recall
)

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from tools.rag_tool import report_query_engine

load_dotenv()
api_key = os.getenv("api_key")
base_url = os.getenv("base_url", "https://api.siliconflow.cn/v1")

# ================= 1. 配置裁判系统 =================
print("🔧 正在初始化 Ragas 裁判系统 (v0.2 纯血架构)...")
openai_client = OpenAI(api_key=api_key, base_url=base_url)

# 实例化大模型裁判与向量裁判
evaluator_llm = llm_factory("Qwen/Qwen2.5-72B-Instruct", client=openai_client)
evaluator_embeddings = embedding_factory("openai", model="BAAI/bge-m3", client=openai_client)

import sys
import os
# 强行把当前文件所在目录的“上一级目录”加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate

# ⬇️ 终极护甲：全部使用 Ragas v0.2 纯血版的新指标！
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,                  # 取代了旧版的 AnswerRelevancy
    LLMContextPrecisionWithReference,   # 取代了旧版的 context_precision
    LLMContextRecall                    # 取代了旧版的 context_recall
)

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from tools.rag_tool import report_query_engine

load_dotenv()
api_key = os.getenv("api_key")
base_url = os.getenv("base_url", "https://api.siliconflow.cn/v1")

# ================= 1. 配置裁判系统 =================
print("🔧 正在初始化 Ragas 裁判系统 (v0.2 纯血架构)...")
openai_client = OpenAI(api_key=api_key, base_url=base_url)

# 实例化大模型裁判与向量裁判
evaluator_llm = llm_factory("Qwen/Qwen2.5-72B-Instruct", client=openai_client)
evaluator_embeddings = embedding_factory("openai", model="BAAI/bge-m3", client=openai_client)

# 实例化四大指标 (严格将裁判绑定给它们)
metrics = [
    Faithfulness(llm=evaluator_llm),        # 
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm)
]

# ================= 2. 构造测试数据 =================
# 在 v0.2 中，标准答案的官方名称叫 reference
questions = ["宁德时代2025年动力电池系统的营业收入是多少？"]
references = ["316,506,369千元"] 

answers = []
contexts = []

print("🧪 正在运行 RAG 引擎获取答案与上下文...")
for q in questions:
    response = report_query_engine.query(q)
    answers.append(str(response))
    contexts.append([n.node.get_content() for n in response.source_nodes])

# 严格使用 v0.2 的原生字段名，彻底告别报错
data = {
    "user_input": questions,           # 对应原来的 question
    "response": answers,               # 对应原来的 answer
    "retrieved_contexts": contexts,    # 对应原来的 contexts
    "reference": references            # 对应原来的 ground_truth
}
dataset = Dataset.from_dict(data)

# ================= 3. 执行评测 =================
print("\n⚖️ 裁判正在打分 (大概需要1-2分钟，请耐心等待进度条跑完)...")
try:
    result = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    print("\n" + "="*70)
    print("📊 Taday 金融智能体 - Ragas 基础评测报告")
    print("="*70)
    
    df = result.to_pandas()
    # 动态过滤掉长文本列，只打印最终得分，防止表格过宽
    score_columns = [col for col in df.columns if col not in ['retrieved_contexts', 'response', 'reference']]
    print(df[score_columns])
    print("-" * 70)
    
except Exception as e:
    print(f"\n❌ 评测发生错误: {e}")

# ================= 2. 构造测试数据 =================
# 在 v0.2 中，标准答案的官方名称叫 reference
questions = ["宁德时代2025年动力电池系统的营业收入是多少？"]
references = ["316,506,369千元"] 

answers = []
contexts = []

print("🧪 正在运行 RAG 引擎获取答案与上下文...")
for q in questions:
    response = report_query_engine.query(q)
    answers.append(str(response))
    contexts.append([n.node.get_content() for n in response.source_nodes])

# 严格使用 v0.2 的原生字段名，彻底告别报错
data = {
    "user_input": questions,           # 对应原来的 question
    "response": answers,               # 对应原来的 answer
    "retrieved_contexts": contexts,    # 对应原来的 contexts
    "reference": references            # 对应原来的 ground_truth
}
dataset = Dataset.from_dict(data)

# ================= 3. 执行评测 =================
print("\n⚖️ 裁判正在打分 (大概需要1-2分钟，请耐心等待进度条跑完)...")
try:
    result = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    print("\n" + "="*70)
    print("📊 Taday 金融智能体 - Ragas 基础评测报告")
    print("="*70)
    
    df = result.to_pandas()
    # 动态过滤掉长文本列，只打印最终得分，防止表格过宽
    score_columns = [col for col in df.columns if col not in ['retrieved_contexts', 'response', 'reference']]
    print(df[score_columns])
    print("-" * 70)
    
except Exception as e:
    print(f"\n❌ 评测发生错误: {e}")