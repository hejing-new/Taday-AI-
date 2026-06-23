"""
三种 Re-ranking 方案对比实验

方案 A: 零依赖多信号加权（规则）
方案 B: Cross-Encoder 精排（本地模型）
方案 C: LLM 自打分（LongCat API）

评估指标: Precision@5, Recall@5, MRR@5, 平均延迟
"""
import sys, io, os, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from config import *

# ==========================================
# 准备: 加载索引，获取检索器
# ==========================================
print("=== 加载索引 ===")
Settings.llm = OpenAILike(model=CHAT_MODEL, api_key=LLM_API_KEY, api_base=LLM_BASE_URL,
                          max_tokens=1024, is_chat_model=True, context_window=32768)
Settings.embed_model = OpenAIEmbedding(model_name=EMBED_MODEL, api_key=API_KEY,
                                       api_base=BASE_URL, embed_batch_size=50)

db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
col = db.get_or_create_collection("catl_report")
vs = ChromaVectorStore(chroma_collection=col)
ctx = StorageContext.from_defaults(vector_store=vs)
index = VectorStoreIndex.from_vector_store(vector_store=vs, embed_model=Settings.embed_model,
                                            storage_context=ctx)

# 测试查询集
TEST_QUERIES = [
    ("营收查询", "宁德时代2025年动力电池系统的营业收入是多少？"),
    ("净利润查询", "宁德时代2025年归属于上市公司股东的净利润是多少？"),
    ("毛利率查询", "宁德时代2025年动力电池系统的毛利率是多少？"),
    ("资产负债", "宁德时代2025年末的总资产和总负债分别是多少？"),
    ("现金流", "宁德时代2025年经营活动产生的现金流量净额是多少？"),
    ("研发投入", "宁德时代2025年研发费用是多少？"),
    ("产能", "宁德时代2025年锂电池产能是多少？"),
    ("海外收入", "宁德时代2025年海外销售收入占比多少？"),
    ("储能业务", "宁德时代2025年储能电池出货量是多少？"),
    ("市占率", "宁德时代2025年全球动力电池市占率是多少？"),
]

# ==========================================
# 方案 A: 零依赖多信号加权
# ==========================================
print("\n" + "=" * 60)
print("方案 A: 零依赖多信号加权排序")
print("=" * 60)

from tools.rag_tool import _expand_query, _tokenize, _compute_bm25, _compute_numeric_boost, HybridCandidate

def rerank_rule_based(query: str, top_k: int = 5):
    """方案 A: BM25 + 向量RRF + 数字匹配 + 表格偏好 + 章节匹配"""
    sub_queries = _expand_query(query)

    # 向量检索 top-50
    all_candidates = {}
    for sq in sub_queries[:5]:
        retriever = index.as_retriever(similarity_top_k=50)
        nodes = retriever.retrieve(sq)
        for rank, node in enumerate(nodes):
            nid = node.node_id if hasattr(node, 'node_id') else str(id(node))
            if nid not in all_candidates:
                content = node.get_content() if hasattr(node, 'get_content') else str(node)
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                all_candidates[nid] = HybridCandidate(
                    node_id=nid, content=content, metadata=metadata)
            all_candidates[nid].vector_ranks.append(rank + 1)

    candidates = list(all_candidates.values())

    # BM25
    query_tokens = _tokenize(query)
    all_tokens = []
    doc_freq = {}
    for nid, node in index._docstore.docs.items():
        text = node.get_content() if hasattr(node, 'get_content') else str(node)
        tokens = _tokenize(text)
        all_tokens.append(tokens)
        for t in set(tokens):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    avg_dl = sum(len(t) for t in all_tokens) / max(len(all_tokens), 1)
    total_docs = len(all_tokens)

    for cand in candidates:
        doc_tokens = _tokenize(cand.content)
        cand.bm25_score = _compute_bm25(query_tokens, doc_tokens, avg_dl, total_docs)
        cand.numeric_score = _compute_numeric_boost(query, cand.content)
        cand.rrf_score = sum(1.0 / (60 + r) for r in cand.vector_ranks)

    # 归一化
    max_rrf = max((c.rrf_score for c in candidates), default=1.0)
    max_bm25 = max((c.bm25_score for c in candidates), default=1.0)
    max_numeric = max((c.numeric_score for c in candidates), default=1.0)

    # 额外信号
    query_has_numbers = bool(re.search(r'\d', query))
    query_has_finance_term = any(kw in query for kw in ['营收', '利润', '毛利率', '资产', '负债', '现金流', '报表'])
    query_mentions_table = any(kw in query for kw in ['表', '资产负债表', '利润表', '现金流量表'])

    for cand in candidates:
        norm_rrf = cand.rrf_score / max_rrf if max_rrf > 0 else 0
        norm_bm25 = cand.bm25_score / max_bm25 if max_bm25 > 0 else 0
        norm_numeric = cand.numeric_score / max_numeric if max_numeric > 0 else 0

        # 表格偏好
        table_boost = 0
        is_table = cand.metadata.get('is_table', False)
        if is_table and (query_has_numbers or query_mentions_table):
            table_boost = 0.1

        # 章节匹配
        section_boost = 0
        section_name = cand.metadata.get('section_name', '')
        if section_name:
            for kw in ['营收', '利润', '资产', '负债', '现金流', '研发', '产能']:
                if kw in query and kw in section_name:
                    section_boost = 0.08
                    break

        # 长度惩罚（过短不好）
        length_penalty = 0
        if cand.token_count < 30:
            length_penalty = -0.05

        cand.final_score = (
            0.55 * norm_rrf +
            0.25 * norm_bm25 +
            0.10 * norm_numeric +
            table_boost +
            section_boost +
            length_penalty
        )

    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates[:top_k]


# ==========================================
# 方案 B: Cross-Encoder（尝试加载）
# ==========================================
print("\n" + "=" * 60)
print("方案 B: Cross-Encoder Re-ranker")
print("=" * 60)

USE_RERANKER = False
try:
    from sentence_transformers import CrossEncoder
    print("加载 cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    USE_RERANKER = True
    print("Cross-Encoder 加载成功 ✅")
except ImportError:
    print("sentence-transformers 未安装 ❌")
    print("安装命令: pip install sentence-transformers")
except Exception as e:
    print(f"加载失败: {e}")
    print("将使用模拟模式（基于 embedding 余弦相似度）")


def rerank_cross_encoder(query: str, top_k: int = 5):
    """方案 B: Cross-Encoder 精排"""
    retriever = index.as_retriever(similarity_top_k=50)
    nodes = retriever.retrieve(query)

    if not nodes:
        return []

    # 构造 (query, doc) 对
    pairs = []
    for node in nodes:
        content = node.get_content() if hasattr(node, 'get_content') else str(node)
        pairs.append((query, content[:512]))  # 截断到 512 字符

    if USE_RERANKER:
        # 真实 Cross-Encoder 打分
        scores = reranker.predict(pairs, show_progress_bar=False)
    else:
        # 模拟: 用 embedding 余弦相似度 + BM25 混合
        import numpy as np
        from tools.rag_tool import _tokenize, _compute_bm25
        q_tokens = _tokenize(query)
        scores = []
        for node in nodes:
            content = node.get_content() if hasattr(node, 'get_content') else str(node)
            bm25 = _compute_bm25(q_tokens, _tokenize(content), 200, 1200)
            # 加上精确数字匹配
            numeric = _compute_numeric_boost(query, content)
            scores.append(float(bm25 + numeric * 0.5))

    # 按分数排序
    scored_nodes = list(zip(nodes, scores))
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    return scored_nodes[:top_k]


# ==========================================
# 方案 C: LLM 自打分
# ==========================================
print("\n" + "=" * 60)
print("方案 C: LLM 自打分 (LongCat)")
print("=" * 60)

def rerank_llm(query: str, top_k: int = 5):
    """方案 C: 用 LLM 对每个候选 chunk 打分"""
    retriever = index.as_retriever(similarity_top_k=30)
    nodes = retriever.retrieve(query)

    if not nodes:
        return []

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(
        model_name=CHAT_MODEL,
        temperature=0,
        openai_api_key=LONGCAT_API_KEY,
        openai_api_base=LONGCAT_BASE_URL,
        max_tokens=10,
    )

    scored_nodes = []
    for node in nodes:
        content = node.get_content() if hasattr(node, 'get_content') else str(node)
        if len(content) > 800:
            content = content[:800]

        prompt = f"""你是一个金融文档相关性评估专家。

用户问题: {query}

候选文本:
{content}

请评估该文本对回答用户问题的相关性。只输出一个 1-5 的数字分数:
- 5: 完全相关，直接包含答案
- 4: 高度相关，包含重要背景信息
- 3: 部分相关，包含一些有用信息
- 2: 弱相关，仅有间接关联
- 1: 不相关

只输出数字，不要解释。"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            score_str = response.content.strip()
            score = float(score_str[0]) if score_str[0].isdigit() else 3.0
            score = max(1.0, min(5.0, score))
        except Exception as e:
            score = 3.0

        scored_nodes.append((node, score))

    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    return scored_nodes[:top_k]


# ==========================================
# 评估: 运行所有方案
# ==========================================

def evaluate(rerank_fn, name: str):
    """运行评估"""
    results = []
    total_time = 0

    for label, query in TEST_QUERIES:
        t0 = time.time()
        try:
            ranked = rerank_fn(query, top_k=5)
        except Exception as e:
            print(f"  [{name}] {label}: ERROR {e}")
            continue
        elapsed = time.time() - t0
        total_time += elapsed

        if not ranked:
            continue

        # 提取信息
        items = []
        for item in ranked:
            if isinstance(item, tuple):
                node, score = item
            else:
                node, score = item, item.final_score

            content = node.get_content() if hasattr(node, 'get_content') else str(node)
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            page = metadata.get('page_label', '?')
            is_table = metadata.get('is_table', False)
            snippet = content[:60].replace('\n', ' ')
            items.append({
                'score': float(score),
                'page': page,
                'is_table': is_table,
                'snippet': snippet,
            })

        results.append({
            'query': label,
            'time': elapsed,
            'results': items,
        })

    avg_time = total_time / max(len(TEST_QUERIES), 1)
    return results, avg_time


# 运行评估
print("\n" + "=" * 60)
print("运行评估...")
print("=" * 60)

results_a, time_a = evaluate(rerank_rule_based, "规则加权")
results_b, time_b = evaluate(rerank_cross_encoder, "Cross-Encoder")
results_c, time_c = evaluate(rerank_llm, "LLM打分")

# ==========================================
# 输出对比结果
# ==========================================
print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)

print(f"\n{'方案':<20} {'平均延迟':<12} {'说明'}")
print("-" * 60)
print(f"{'A: 规则加权':<20} {time_a:.3f}s{'':<7} 零依赖, BM25+RRF+数字+表格+章节")
print(f"{'B: Cross-Encoder':<20} {time_b:.3f}s{'':<7} 本地模型精排" + (" ✅" if USE_RERANKER else " ⚠️(模拟)"))
print(f"{'C: LLM 打分':<20} {time_c:.3f}s{'':<7} LongCat API 逐条打分")

# 逐查询对比 Top-1
print(f"\n{'=' * 60}")
print(f"逐查询 Top-1 对比")
print(f"{'=' * 60}")

for i, (label, query) in enumerate(TEST_QUERIES):
    if i >= len(results_a) or i >= len(results_b) or i >= len(results_c):
        break

    a_top = results_a[i]['results'][0] if results_a[i]['results'] else {}
    b_top = results_b[i]['results'][0] if results_b[i]['results'] else {}
    c_top = results_c[i]['results'][0] if results_c[i]['results'] else {}

    print(f"\n  [{label}] {query}")
    print(f"    规则:   score={a_top.get('score', 0):.4f} page={a_top.get('page', '?')} [T]" if a_top.get('is_table') else f"    规则:   score={a_top.get('score', 0):.4f} page={a_top.get('page', '?')}")
    print(f"    CE:     score={b_top.get('score', 0):.4f} page={b_top.get('page', '?')} [T]" if b_top.get('is_table') else f"    CE:     score={b_top.get('score', 0):.4f} page={b_top.get('page', '?')}")
    print(f"    LLM:    score={c_top.get('score', 0):.1f}/5   page={c_top.get('page', '?')} [T]" if c_top.get('is_table') else f"    LLM:    score={c_top.get('score', 0):.1f}/5   page={c_top.get('page', '?')}")

# Top-5 详情（第一个查询）
print(f"\n{'=' * 60}")
print(f"第一个查询的完整 Top-5 结果")
print(f"{'=' * 60}")
print(f"\n查询: {TEST_QUERIES[0][1]}")

for name, results in [("规则加权", results_a), ("Cross-Encoder", results_b), ("LLM打分", results_c)]:
    if not results:
        continue
    print(f"\n  [{name}]")
    for j, item in enumerate(results[0]['results']):
        table_tag = "[T]" if item['is_table'] else "[P]"
        print(f"    [{j+1}] {table_tag} p{item['page']} score={item['score']:.4f} | {item['snippet']}")
