"""
Taday 金融智能体 — 自动化 RAG 评测脚本 (v3 — 固定回归集 + 基线对比)

基于 Ragas v0.2 指标，支持两种模式：
  1. 固定回归模式 (--fixed): 使用 regression_set.json 固定题目，用于前后对比
  2. 动态出题模式: 从 ChromaDB 随机抽题

评测路径对齐生产引擎:
  HybridQueryEngine (向量60% + BM25 30% + 数字匹配10% + 可选 Cross-Encoder)

使用方式:
  python auto_eval.py                    # 动态出题 3 题
  python auto_eval.py --fixed            # 固定回归集 (用于基线对比)
  python auto_eval.py --fixed --n 10     # 固定回归集 (最多10题)
  python auto_eval.py --ce               # 启用 Cross-Encoder
  python auto_eval.py --fixed --baseline # 保存为基线
  python auto_eval.py --fixed --compare  # 和基线做对比
"""
import sys
import os
import io
import time
import argparse
from datetime import datetime

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

# 从项目根目录加载 .env
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

# 对接新的混合检索引擎
from tools.rag_tool import get_query_engine, _format_hybrid_response
from tools.rag_tool import HybridQueryEngine

api_key = os.getenv("api_key")
base_url = os.getenv("base_url", "https://api.siliconflow.cn/v1")
# LongCat API 用于裁判 (独立 TPM 限额，不与 SiliconFlow embedding 竞争)
longcat_api_key = os.getenv("LONGCAT_API_KEY", os.getenv("API_KEY"))
longcat_base_url = os.getenv("LONGCAT_BASE_URL", "https://api.longcat.chat/openai/v1")

# ================= 路径常量 =================
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REGRESSION_SET_PATH = os.path.join(EVAL_DIR, "regression_set.json")
BASELINE_PATH = os.path.join(EVAL_DIR, "baseline.json")


# ================= 0. 引擎初始化 =================
def _init_engine(use_cross_encoder: bool = False):
    """初始化被测引擎 (HybridQueryEngine)"""
    if use_cross_encoder:
        os.environ["USE_CROSS_ENCODER"] = "1"
    else:
        os.environ["USE_CROSS_ENCODER"] = "0"

    print("🔧 正在初始化被测引擎 (HybridQueryEngine)...")
    engine = get_query_engine(collection_name="catl_report")

    if isinstance(engine, HybridQueryEngine):
        ce_status = "Cross-Encoder 精排已启用" if engine._use_cross_encoder else "Cross-Encoder 未启用"
        print(f"   引擎: HybridQueryEngine (向量60% + BM25 30% + 数字匹配10% | {ce_status})")
    else:
        print(f"   警告: 引擎类型为 {type(engine).__name__} (非混合引擎)")

    return engine


# ================= 1. 配置裁判大模型 =================
print("🔧 正在初始化 Ragas 裁判系统 (v0.2)...")

# 裁判模型配置 — 使用 LongCat API (独立 TPM 限额，不与 SiliconFlow embedding 竞争)
from ragas.run_config import RunConfig

judge_run_config = RunConfig(
    timeout=600,        # 单次请求超时 10 分钟
    max_wait=60,        # 重试最大等待 1 分钟
    max_workers=1,      # 串行执行，稳定优先
    max_retries=15,     # 限流时多重试
)

# 裁判模型 — LongCat API (独立于 SiliconFlow 的 TPM 通道)
# ⚠️ 关键: ChatOpenAI 内部读取 OPENAI_API_KEY 环境变量作为 API key
# 在创建 LLM 前临时切换环境变量为 LongCat，避免和 SiliconFlow key 冲突
_saved_openai_key = os.environ.get("OPENAI_API_KEY")
_saved_openai_base = os.environ.get("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = longcat_api_key
os.environ["OPENAI_BASE_URL"] = longcat_base_url

evaluator_llm = llm_factory(
    model="LongCat-2.0-Preview",
    base_url=longcat_base_url,
    run_config=judge_run_config,
)

# 增加 max_tokens 避免 LongCat 生成被截断
# Faithfulness 指标需要生成大量 NLI 判断，默认 max_tokens 可能不够
if hasattr(evaluator_llm, 'langchain_llm'):
    inner_llm = evaluator_llm.langchain_llm
    if hasattr(inner_llm, 'max_tokens'):
        inner_llm.max_tokens = 4096
    if hasattr(inner_llm, 'max_completion_tokens'):
        inner_llm.max_completion_tokens = 4096

# 恢复 SiliconFlow 环境变量 (供 embedding 裁判使用)
if _saved_openai_key is not None:
    os.environ["OPENAI_API_KEY"] = _saved_openai_key
else:
    os.environ.pop("OPENAI_API_KEY", None)
if _saved_openai_base is not None:
    os.environ["OPENAI_BASE_URL"] = _saved_openai_base
else:
    os.environ.pop("OPENAI_BASE_URL", None)

# Embedding 裁判 — 通过环境变量注入 SiliconFlow base_url
os.environ["OPENAI_BASE_URL"] = base_url
os.environ["OPENAI_API_KEY"] = api_key
evaluator_embeddings = embedding_factory("BAAI/bge-m3")

metrics = [
    Faithfulness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),
]

# 动态出题老师 — 也用 LongCat
teacher_client = OpenAI(api_key=longcat_api_key, base_url=longcat_base_url, timeout=300)
TEACHER_MODEL = "LongCat-2.0-Preview"

# Ragas 四大指标名 (用于基线对比和报告)
RAGAS_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ================= 2. 固定回归集 =================
def load_regression_set(max_n: int = None) -> list:
    """
    加载固定回归测试集。

    Args:
        max_n: 最多取多少题，None 表示全部

    Returns:
        test_cases 列表
    """
    if not os.path.exists(REGRESSION_SET_PATH):
        raise FileNotFoundError(
            f"固定回归集不存在: {REGRESSION_SET_PATH}\n"
            f"请创建 eval/regression_set.json"
        )

    with open(REGRESSION_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    if max_n is not None:
        cases = cases[:max_n]

    print(f"📋 固定回归模式: 已加载 {len(cases)} 题 "
          f"(版本: {data.get('version', 'unknown')})")

    # 统计分类
    categories = {}
    for c in cases:
        cat = c.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count} 题")

    return cases


def generate_test_cases(n=3):
    """动态出题: 从 ChromaDB 随机抽取片段，让 72B 模型出题"""
    print(f"\n🎲 动态出题模式: 从 ChromaDB 随机抽取 {n} 个知识片段...")

    current_dir = os.path.dirname(EVAL_DIR)
    db_path = os.path.join(current_dir, "chroma_db")
    db = chromadb.PersistentClient(path=db_path)

    try:
        collection = db.get_collection("catl_report")
        all_docs = collection.get()['documents']
    except Exception as e:
        raise ValueError(f"无法读取 ChromaDB，请确保你的 RAG 工具已经成功建库！报错: {e}")

    if len(all_docs) < n:
        n = len(all_docs)
    sample_texts = random.sample(all_docs, n)

    test_cases = []

    for i, context_text in enumerate(sample_texts):
        if len(context_text) < 50:
            continue

        print(f"  老师正在阅读第 {i+1} 个片段并出题...")

        prompt = f"""你是一位严厉的金融考试官。请根据下面这段【宁德时代财报】的原文，出一道具体的、有难度的问答题，并给出标准答案。

        原文内容：
        {context_text}

        要求：
        1. 问题必须能从原文中找到依据，不要空泛。
        2. 答案必须极其精准（包含具体数字或术语）。
        3. 请严格按照下面的 JSON 格式返回，不要有任何废话。
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
            res["id"] = f"DYN-{i+1:03d}"
            res["category"] = "dynamic"
            test_cases.append(res)
        except Exception as e:
            print(f"  第 {i+1} 题生成失败，跳过。原因: {e}")

    return test_cases


# ================= 3. 基线管理 =================
def load_baseline() -> dict:
    """加载基线文件，不存在返回 None"""
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(df, metadata: dict):
    """保存当前评测结果为基线"""
    baseline = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "metadata": metadata,
        "summary": {},
        "cases": []
    }

    # 保存汇总指标
    for metric in RAGAS_METRIC_NAMES:
        if metric in df.columns:
            baseline["summary"][metric] = {
                "mean": round(float(df[metric].mean()), 4),
                "std": round(float(df[metric].std()), 4),
                "min": round(float(df[metric].min()), 4),
                "max": round(float(df[metric].max()), 4),
            }

    # 保存逐题详情
    for idx, row in df.iterrows():
        case_data = {"index": idx}
        for metric in RAGAS_METRIC_NAMES:
            if metric in row.index and row[metric] is not None:
                case_data[metric] = round(float(row[metric]), 4)
        baseline["cases"].append(case_data)

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)

    return baseline


def compare_with_baseline(df, baseline: dict) -> str:
    """
    将当前评测结果与基线做对比，返回对比报告字符串。

    Args:
        df: 当前评测结果的 DataFrame
        baseline: 基线数据 dict

    Returns:
        对比报告文本
    """
    if baseline is None:
        return "⚠️ 无基线数据，无法对比。使用 --baseline 保存当前结果为基线。"

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append(f"📊 基线对比报告")
    lines.append(f"   基线时间: {baseline.get('created_at', 'unknown')}")
    lines.append(f"   当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)
    lines.append("")

    # 表头
    header = f"{'指标':<20} {'基线':>8} {'本次':>8} {'变化':>8} {'状态':>6}"
    lines.append(header)
    lines.append("-" * 60)

    has_regression = False

    for metric in RAGAS_METRIC_NAMES:
        if metric not in df.columns:
            continue

        current_mean = round(float(df[metric].mean()), 4)
        baseline_summary = baseline.get("summary", {}).get(metric, {})
        baseline_mean = baseline_summary.get("mean", None)

        if baseline_mean is None:
            lines.append(f"{metric:<20} {'N/A':>8} {current_mean:>8.4f} {'N/A':>8} {'🔵':>6}")
            continue

        delta = round(current_mean - baseline_mean, 4)

        if delta > 0.01:
            status = "✅ 提升"
        elif delta < -0.01:
            status = "⚠️ 退化"
            has_regression = True
        else:
            status = "➡️ 持平"

        sign = "+" if delta >= 0 else ""
        lines.append(f"{metric:<20} {baseline_mean:>8.4f} {current_mean:>8.4f} {sign}{delta:>7.4f} {status:>6}")

    lines.append("-" * 60)

    # 分类别细分
    lines.append("")
    lines.append("📂 按题型分类:")

    # 从 baseline metadata 中获取分类信息
    baseline_meta = baseline.get("metadata", {})
    categories = baseline_meta.get("categories", {})

    if categories:
        for cat, cat_info in categories.items():
            lines.append(f"\n  [{cat}] ({cat_info.get('count', '?')} 题)")
            for metric in RAGAS_METRIC_NAMES:
                cat_mean = cat_info.get(f"{metric}_mean", None)
                if cat_mean is not None:
                    lines.append(f"    {metric:<20} = {cat_mean:.4f}")

    lines.append("")

    if has_regression:
        lines.append("⚠️ 发现指标退化! 请检查最近改动是否影响了检索质量。")
    else:
        lines.append("✅ 无退化，所有指标持平或提升。")

    lines.append("=" * 90)

    return "\n".join(lines)


# ================= 4. 主评测流程 =================
def run_evaluation(n=3, save_results=True, show_details=True,
                   use_cross_encoder=False, fixed_mode=False,
                   save_baseline=False, compare_baseline=False):
    """
    执行完整的评测流程

    Args:
        n: 题目数量 (动态模式下)
        save_results: 是否保存结果到 JSON
        show_details: 是否显示详细中间过程
        use_cross_encoder: 是否启用 Cross-Encoder
        fixed_mode: 是否使用固定回归集
        save_baseline: 是否保存当前结果为基线
        compare_baseline: 是否与基线做对比
    """
    # ================= 0. 初始化被测引擎 =================
    engine = _init_engine(use_cross_encoder=use_cross_encoder)

    # ================= 1. 获取测试用例 =================
    if fixed_mode:
        test_cases = load_regression_set(max_n=n)
    else:
        test_cases = generate_test_cases(n=n)

    if not test_cases:
        print("未获取到任何测试题，退出。")
        return

    # ================= 2. 答题 =================
    user_inputs = []
    references = []
    responses = []
    retrieved_contexts = []
    case_ids = []
    case_categories = []

    print(f"\n📝 开始答题 ({len(test_cases)} 题)...")

    for case in test_cases:
        q = case.get("question", "")
        ref = case.get("reference", "")
        case_id = case.get("id", "")
        category = case.get("category", "unknown")

        print(f"  [{case_id or '?'}] {q[:50]}...")

        hybrid_response = engine.query(q)
        formatted_answer = _format_hybrid_response(hybrid_response)

        user_inputs.append(q)
        references.append(ref)
        responses.append(formatted_answer)
        retrieved_contexts.append([node.get_content() for node in hybrid_response.source_nodes])
        case_ids.append(case_id)
        case_categories.append(category)

    # ================= 3. 冷却期 — 让 TPM 限额重置 =================
    print("\n⏳ 冷却 60 秒，等待 SiliconFlow TPM 限额重置...")
    time.sleep(60)

    # ================= 4. Ragas 判卷 =================
    print("\n⚖️ 裁判正在评分 (大概需要2-3分钟)...")

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
            raise_exceptions=True,
            run_config=judge_run_config,  # 传递裁判的 RunConfig (含 timeout=600)
        )

        # ================= 4. 生成报告 =================
        ce_tag = " + CE" if use_cross_encoder else ""
        mode_tag = "固定回归" if fixed_mode else "动态出题"
        print("\n" + "=" * 90)
        print(f"📊 Taday Ragas 评测报告 — HybridQueryEngine{ce_tag} [{mode_tag}]")
        print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 90)

        df = result.to_pandas()
        score_columns = [col for col in df.columns
                         if col not in ['retrieved_contexts', 'response', 'reference']]

        # 添加 ID 和分类列
        df.insert(0, "id", case_ids)
        df.insert(1, "category", case_categories)

        print(df[["id", "category"] + score_columns].to_string(index=False))
        print("-" * 90)

        # 汇总统计
        print("\n📈 汇总统计:")
        for metric in RAGAS_METRIC_NAMES:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                print(f"   {metric:<25} = {mean_val:.4f} (±{std_val:.4f})")

        # 分类别统计
        unique_cats = set(case_categories)
        if len(unique_cats) > 1:
            print("\n📂 按题型分类:")
            for cat in sorted(unique_cats):
                cat_mask = [c == cat for c in case_categories]
                cat_count = sum(cat_mask)
                parts = [f"   [{cat}] ({cat_count} 题)"]
                for metric in RAGAS_METRIC_NAMES:
                    if metric in df.columns:
                        cat_mean = df[metric][cat_mask].mean()
                        parts.append(f"      {metric}: {cat_mean:.4f}")
                print("\n".join(parts))

        # ================= 5. 基线对比 =================
        baseline = load_baseline() if compare_baseline else None

        if baseline:
            report = compare_with_baseline(df, baseline)
            print(report)

        # ================= 6. 保存基线 =================
        if save_baseline:
            categories_info = {}
            for cat in set(case_categories):
                cat_mask = [c == cat for c in case_categories]
                cat_count = sum(cat_mask)
                cat_info = {"count": cat_count}
                for metric in RAGAS_METRIC_NAMES:
                    if metric in df.columns:
                        cat_info[f"{metric}_mean"] = round(
                            float(df[metric][cat_mask].mean()), 4
                        )
                categories_info[cat] = cat_info

            meta = {
                "mode": mode_tag,
                "use_cross_encoder": use_cross_encoder,
                "engine": "HybridQueryEngine",
                "num_cases": len(test_cases),
                "categories": categories_info,
            }

            saved = save_baseline(df, meta)
            print(f"\n✅ 基线已保存至: {os.path.abspath(BASELINE_PATH)}")

        # ================= 7. 保存结果 =================
        if save_results:
            output_path = os.path.join(EVAL_DIR, "ragas_evaluation_results.json")
            df.to_json(output_path, orient='records', force_ascii=False, indent=4)
            print(f"📄 评测结果已保存至: {os.path.abspath(output_path)}")

        # ================= 8. 详细中间过程 =================
        if show_details:
            print("\n" + "=" * 90)
            print("🔍 检索详情 (以第一题为例):")
            print("=" * 90)

            detailed = dataset.to_dict()
            print(f"问题: {detailed['user_input'][0]}")
            print(f"答案: {detailed['reference'][0]}")

            print(f"\n检索到 {len(detailed['retrieved_contexts'][0])} 条上下文:")
            for i, ctx in enumerate(detailed['retrieved_contexts'][0]):
                print(f"  [{i+1}] {ctx[:200]}...")

            print(f"\nAgent 回答 (前500字):")
            print(detailed['response'][0][:500])

        print("\n" + "=" * 90)
        print("✅ 评测完成！")
        print("=" * 90)

    except Exception as e:
        print(f"\n❌ 评测过程中发生错误: {e}")


# ================= 入口 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Taday RAG 自动化评测 (HybridQueryEngine + 基线对比)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_eval.py                          # 动态出题 3 题
  python auto_eval.py --fixed                  # 固定回归集 (全量)
  python auto_eval.py --fixed --n 5            # 固定回归集 (最多5题)
  python auto_eval.py --fixed --baseline       # 保存当前结果为基线
  python auto_eval.py --fixed --compare        # 和基线做对比
  python auto_eval.py --fixed --baseline --ce  # 固定集 + CE + 保存基线
        """
    )
    parser.add_argument("--n", type=int, default=3,
                        help="题目数量 (默认: 3, 固定模式下最多取多少题)")
    parser.add_argument("--save", action="store_true",
                        help="保存结果到 ragas_evaluation_results.json")
    parser.add_argument("--no-details", action="store_true",
                        help="不显示详细中间过程")
    parser.add_argument("--ce", action="store_true",
                        help="启用 Cross-Encoder 精排 (USE_CROSS_ENCODER=1)")
    parser.add_argument("--fixed", action="store_true",
                        help="使用固定回归测试集 (来自 regression_set.json)")
    parser.add_argument("--baseline", action="store_true",
                        help="保存当前评测结果为基线 (baseline.json)")
    parser.add_argument("--compare", action="store_true",
                        help="评测后和基线做对比")
    args = parser.parse_args()

    run_evaluation(
        n=args.n,
        save_results=args.save,
        show_details=not args.no_details,
        use_cross_encoder=args.ce,
        fixed_mode=args.fixed,
        save_baseline=args.baseline,
        compare_baseline=args.compare,
    )
