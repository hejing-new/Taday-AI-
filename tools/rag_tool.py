"""
RAG 财报检索工具 (CATL) — 增强版

工厂模式 + 懒加载，第一次调用 get_query_engine() 时才初始化连接。

=== 增强功能 ===
1. 语义分块 (SemanticSplitter) — 按语义边界切分，表格更完整
2. 查询扩展 — 金融同义词扩展 + 多子查询生成
3. 混合检索重排序 — 向量相似度 + BM25关键词 + 精确数字匹配
"""
import sys
import os
import io
import re
import time
import math
import chromadb
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dotenv import load_dotenv
from langchain_core.tools import tool
from tenacity import retry, wait_exponential, stop_after_attempt

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

from tools.financial_chunker import chunk_financial_report, ChunkCandidate

from config import API_KEY, BASE_URL, CHAT_MODEL, EMBED_MODEL, CHROMA_DB_PATH, DATA_PATH, LLM_BASE_URL, LLM_API_KEY, LONGCAT_API_KEY
from logger import logger

# UTF-8 环境防护
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"


def _setup_utf8_stdio():
    """延迟包装 stdout/stderr 为 UTF-8，避免影响 pydantic 初始化"""
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# 全局缓存（懒加载，第一次调用时才初始化）
# 结构: {collection_name: query_engine}
_QUERY_ENGINE_CACHE = {}


def _setup_settings():
    """配置 LlamaIndex 全局设置（幂等，可安全多次调用）"""
    current = Settings.llm
    if current is not None and hasattr(current, 'api_base') and hasattr(current, 'api_key'):
        if (current.api_base == LLM_BASE_URL and current.api_key == LLM_API_KEY
                and hasattr(current, 'model') and current.model == CHAT_MODEL):
            return
    _setup_utf8_stdio()
    logger.info(f"初始化 LLM 引擎 [model={CHAT_MODEL}, base={LLM_BASE_URL}]...")
    Settings.llm = OpenAILike(
        model=CHAT_MODEL,
        api_key=LLM_API_KEY,
        api_base=LLM_BASE_URL,
        max_tokens=1024,
        is_chat_model=True,
        context_window=32768
    )


class SafeSiliconFlowEmbedding(OpenAIEmbedding):
    """企业级 API 限流与指数退避重试"""
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        time.sleep(0.5)
        return super()._get_text_embeddings(texts)


def _get_embedding_model():
    """获取 Embedding 模型（带限流保护）

    Embedding 必须走 SiliconFlow (支持 BAAI/bge-m3)，
    LongCat 不支持 embedding 接口。
    """
    current = Settings.embed_model
    if current is not None and hasattr(current, 'api_base') and hasattr(current, 'api_key'):
        if current.api_base == BASE_URL and current.api_key == API_KEY:
            return current
    Settings.embed_model = SafeSiliconFlowEmbedding(
        model_name=EMBED_MODEL,
        api_key=API_KEY,
        api_base=BASE_URL,
        embed_batch_size=50
    )
    return Settings.embed_model


# ==========================================
# 金融查询扩展器
# ==========================================

_FINANCIAL_SYNONYMS: Dict[str, List[str]] = {
    # 营收相关
    "营收": ["营业收入", "销售收入", "总收入", "销售额"],
    "营业收入": ["营收", "销售收入", "总收入"],
    "净利润": ["归母净利润", "归属于上市公司股东的净利润", "净利"],
    "毛利": ["毛利润", "销售毛利"],
    "毛利率": ["销售毛利率", "综合毛利率"],
    # 资产相关
    "总资产": ["资产合计", "资产总计"],
    "净资产": ["所有者权益", "股东权益", "净资产合计"],
    "负债": ["负债合计", "总负债"],
    # 现金流
    "现金流": ["现金流量", "经营活动产生的现金流量"],
    "经营现金流": ["经营活动现金流", "经营活动产生的现金流量净额"],
    "投资现金流": ["投资活动现金流", "投资活动产生的现金流量净额"],
    # 业务相关
    "动力电池": ["动力电池系统", "电池系统", "动力电池产品"],
    "储能电池": ["储能系统", "储能电池系统", "储能产品"],
    "产能": ["生产能力", "产量", "出货量"],
    "市占率": ["市场份额", "市场占有率"],
    # 时间相关
    "2025年": ["2025", "2025年度", "二零二五年"],
    "2024年": ["2024", "2024年度", "二零二四年"],
    "2025上半年": ["2025H1", "2025年半年度", "2025年1-6月"],
    "2025全年": ["2025年年度", "2025年度"],
    # 报表相关
    "资产负债表": ["合并资产负债表", "资产负债表"],
    "利润表": ["合并利润表", "损益表", "利润表"],
    "现金流量表": ["合并现金流量表", "现金流量表"],
    # 指标
    "ROE": ["净资产收益率", "加权平均净资产收益率"],
    "ROA": ["总资产收益率"],
    "资产负债率": ["负债率", "杠杆率"],
    "研发投入": ["研发费用", "研发支出", "R&D投入"],
}


def _expand_query(query: str) -> List[str]:
    """
    查询扩展：生成语义等价的子查询列表。

    策略：
    1. 金融同义词替换
    2. 数字格式归一化
    3. 返回原始查询 + 扩展查询（去重）
    """
    sub_queries = [query]  # 原始查询始终保留

    # 同义词扩展：对每个匹配的关键词，生成一个替换后的查询
    expanded = set()
    for term, synonyms in _FINANCIAL_SYNONYMS.items():
        if term in query:
            for syn in synonyms:
                if syn != term and syn not in query:
                    new_query = query.replace(term, syn)
                    expanded.add(new_query)

    # 限制扩展数量，避免查询爆炸
    sub_queries.extend(list(expanded)[:5])

    return sub_queries


# ==========================================
# BM25 关键词评分器
# ==========================================

def _tokenize(text: str) -> List[str]:
    """简单中英文混合分词：中文按字/词，英文按空格"""
    # 提取中文词组（2-8字）
    tokens = re.findall(r'[\u4e00-\u9fff]{2,8}', text)
    # 提取数字（含小数、百分号、单位）
    tokens.extend(re.findall(r'\d+\.?\d*[万亿千百]?元?', text))
    tokens.extend(re.findall(r'\d+\.?\d*%', text))
    # 提取英文单词
    tokens.extend(re.findall(r'[a-zA-Z]{2,}', text.lower()))
    return tokens


def _compute_bm25(query_tokens: List[str], doc_tokens: List[str],
                  avg_dl: float, total_docs: int,
                  doc_freq: Counter = None,
                  k1: float = 1.5, b: float = 0.75) -> float:
    """
    计算 BM25 分数。

    Args:
        query_tokens: 查询词列表
        doc_tokens: 文档词列表
        avg_dl: 平均文档长度
        total_docs: 总文档数
        doc_freq: 语料库级文档频率（可选，用于 IDF 计算）
        k1: BM25 参数，控制词频饱和度
        b: BM25 参数，控制文档长度归一化
    """
    if not doc_tokens or not query_tokens or avg_dl <= 0:
        return 0.0

    doc_len = len(doc_tokens)
    tf = Counter(doc_tokens)

    # 使用语料库级 DF（如果提供），否则退化为单文档 DF
    if doc_freq is None:
        df = Counter()
        for t in set(doc_tokens):
            df[t] = 1
    else:
        df = doc_freq

    score = 0.0
    for qt in set(query_tokens):
        if qt not in tf:
            continue
        # 逆文档频率（使用语料库级 DF）
        n = df.get(qt, 0)
        idf = math.log((total_docs - n + 0.5) / (n + 0.5) + 1)
        # 词频（饱和）
        f = tf[qt]
        tf_norm = (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avg_dl))
        score += idf * tf_norm

    return score


def _compute_numeric_boost(query: str, doc_text: str) -> float:
    """
    精确数字匹配加权。

    如果文档中包含查询里提到的具体数字（如 "3165.06"），给予额外加权。
    财报查询中，精确数字匹配是最强的相关性信号。
    """
    # 提取查询中的数字
    query_numbers = set(re.findall(r'\d+\.?\d*', query))
    if not query_numbers:
        return 0.0

    # 提取文档中的数字
    doc_numbers = set(re.findall(r'\d+\.?\d*', doc_text))

    # 计算交集
    overlap = query_numbers & doc_numbers
    if not overlap:
        return 0.0

    # 每命中一个数字加 0.15 分
    return len(overlap) * 0.15


# ==========================================
# 混合检索查询引擎
# ==========================================

class HybridQueryEngine:
    """
    混合检索查询引擎：向量检索 + BM25 + 精确数字匹配 + RRF 融合 + 可选 Cross-Encoder 精排。

    工作流程：
    1. 查询扩展（同义词替换）
    2. 多子查询向量检索（每个 top_k=15）
    3. 合并候选集
    4. BM25 关键词评分
    5. 精确数字匹配加权
    6. RRF 融合排序
    7. (可选) Cross-Encoder 精排
    8. 返回 top_k
    """

    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 8,
                 hybrid_k: int = 15, vector_weight: float = 0.6,
                 bm25_weight: float = 0.3, numeric_weight: float = 0.1,
                 use_cross_encoder: bool = False,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._index = index
        self._top_k = similarity_top_k
        self._hybrid_k = hybrid_k
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._numeric_weight = numeric_weight
        self._use_cross_encoder = use_cross_encoder
        self._cross_encoder_model_name = cross_encoder_model
        self._reranker = None

        # 预计算 BM25 统计信息
        self._bm25_stats = self._compute_bm25_stats()

        # 按需加载 Cross-Encoder
        if use_cross_encoder:
            self._load_cross_encoder()

    def _load_cross_encoder(self):
        """懒加载 Cross-Encoder 模型（可选依赖）"""
        if self._reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载 Cross-Encoder 模型: {self._cross_encoder_model_name} ...")
            self._reranker = CrossEncoder(self._cross_encoder_model_name)
            logger.info("Cross-Encoder 加载成功 ✅")
        except ImportError:
            logger.warning("sentence-transformers 未安装，Cross-Encoder 不可用。安装: pip install sentence-transformers")
            self._use_cross_encoder = False
        except Exception as e:
            logger.warning(f"Cross-Encoder 加载失败: {e}，回退到规则排序")
            self._use_cross_encoder = False

    def _apply_cross_encoder(self, query: str, candidates: list, top_k: int) -> list:
        """
        用 Cross-Encoder 对候选项二次排序。

        策略：对候选集计算 query-document 相关性分数，
        然后与规则分数加权融合（CE 70% + 规则 30%）。
        """
        if not candidates or self._reranker is None:
            return candidates

        # 构造 (query, doc) 对
        pairs = []
        for cand in candidates:
            text = cand.content[:512]  # 截断以加速 CE
            pairs.append((query, text))

        # Cross-Encoder 打分
        try:
            ce_scores = self._reranker.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Cross-Encoder 预测失败: {e}，回退到规则排序")
            return candidates

        # 归一化 CE 分数到 [0, 1]
        ce_min = float(ce_scores.min()) if len(ce_scores) > 0 else 0
        ce_max = float(ce_scores.max()) if len(ce_scores) > 0 else 1
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

        for i, cand in enumerate(candidates):
            norm_ce = (float(ce_scores[i]) - ce_min) / ce_range
            # 融合: CE 70% + 规则 30%
            cand.final_score = 0.7 * norm_ce + 0.3 * cand.final_score

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[:top_k]

    def _compute_bm25_stats(self) -> dict:
        """
        预计算语料库级 BM25 统计信息（仅执行一次）。

        优化：将 corpus 级文档频率（DF）缓存，避免每次查询遍历全部节点。
        """
        try:
            raw_nodes = self._index._docstore.docs  # LlamaIndex 内部文档存储
        except AttributeError:
            logger.warning("无法访问内部文档存储，BM25 统计将使用默认值")
            return {"avg_dl": 200, "total_docs": 1000, "doc_freq": Counter()}

        all_tokens = []
        doc_freq = Counter()
        for node_id, node in raw_nodes.items():
            text = node.get_content() if hasattr(node, 'get_content') else str(node)
            tokens = _tokenize(text)
            all_tokens.append(tokens)
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1

        total_docs = len(all_tokens)
        avg_dl = sum(len(t) for t in all_tokens) / max(total_docs, 1)

        logger.info(f"BM25 语料库统计完成: {total_docs} 篇, 平均长度 {avg_dl:.0f} tokens, "
                    f"词表大小 {len(doc_freq)}")

        return {
            "avg_dl": avg_dl,
            "total_docs": total_docs,
            "doc_freq": doc_freq,
        }

    def query(self, query_str: str) -> 'HybridQueryResponse':
        """执行混合检索查询"""
        # 1. 查询扩展
        sub_queries = _expand_query(query_str)
        logger.info(f"查询扩展: '{query_str[:30]}...' → {len(sub_queries)} 个子查询")

        # 2. 多子查询向量检索
        all_candidates: Dict[str, 'HybridCandidate'] = {}
        for sq in sub_queries:
            retriever = self._index.as_retriever(similarity_top_k=self._hybrid_k)
            nodes = retriever.retrieve(sq)
            for rank, node in enumerate(nodes):
                node_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
                if node_id not in all_candidates:
                    content = node.get_content() if hasattr(node, 'get_content') else str(node)
                    metadata = node.metadata if hasattr(node, 'metadata') else {}
                    all_candidates[node_id] = HybridCandidate(
                        node_id=node_id,
                        content=content,
                        metadata=metadata,
                        vector_score=0.0,
                        vector_ranks=[],
                    )
                # 记录向量检索排名（RRF 用）
                all_candidates[node_id].vector_ranks.append(rank + 1)

        candidates = list(all_candidates.values())
        logger.info(f"向量检索合并后候选数: {len(candidates)}")

        if not candidates:
            return HybridQueryResponse(nodes=[], source_nodes=[])

        # 3. 计算 BM25 分数（使用缓存的语料库级 DF）
        query_tokens = _tokenize(query_str)
        avg_dl = self._bm25_stats["avg_dl"]
        total_docs = self._bm25_stats["total_docs"]
        doc_freq = self._bm25_stats.get("doc_freq")

        for cand in candidates:
            doc_tokens = _tokenize(cand.content)
            cand.bm25_score = _compute_bm25(query_tokens, doc_tokens, avg_dl, total_docs, doc_freq)
            cand.numeric_score = _compute_numeric_boost(query_str, cand.content)

        # 4. RRF 融合向量检索排名
        for cand in candidates:
            cand.rrf_score = sum(1.0 / (60 + rank) for rank in cand.vector_ranks)

        # 5. 归一化分数并加权融合
        max_rrf = max((c.rrf_score for c in candidates), default=1.0)
        max_bm25 = max((c.bm25_score for c in candidates), default=1.0)
        max_numeric = max((c.numeric_score for c in candidates), default=1.0)

        for cand in candidates:
            norm_rrf = cand.rrf_score / max_rrf if max_rrf > 0 else 0
            norm_bm25 = cand.bm25_score / max_bm25 if max_bm25 > 0 else 0
            norm_numeric = cand.numeric_score / max_numeric if max_numeric > 0 else 0

            cand.final_score = (
                self._vector_weight * norm_rrf +
                self._bm25_weight * norm_bm25 +
                self._numeric_weight * norm_numeric
            )

        # 6. 规则排序
        candidates.sort(key=lambda c: c.final_score, reverse=True)

        # 7. (可选) Cross-Encoder 精排
        if self._use_cross_encoder and self._reranker is not None:
            logger.info("Cross-Encoder 精排中...")
            # 对 top-N 候选做 CE 精排（多于最终 top_k，留余量）
            ce_candidates = candidates[:max(self._top_k * 3, 15)]
            top_candidates = self._apply_cross_encoder(query_str, ce_candidates, self._top_k)
        else:
            top_candidates = candidates[:self._top_k]

        # 8. 构造返回结果
        source_nodes = []
        for cand in top_candidates:
            # 构造与 LlamaIndex SourceNode 兼容的对象
            source_nodes.append(HybridSourceNode(
                content=cand.content,
                metadata=cand.metadata,
                score=cand.final_score,
            ))

        return HybridQueryResponse(nodes=source_nodes, source_nodes=source_nodes)


class HybridCandidate:
    """候选项，用于混合检索打分"""
    __slots__ = ['node_id', 'content', 'metadata', 'vector_score',
                 'vector_ranks', 'bm25_score', 'numeric_score',
                 'rrf_score', 'final_score']

    def __init__(self, node_id: str, content: str, metadata: dict,
                 vector_score: float = 0.0, vector_ranks: list = None):
        self.node_id = node_id
        self.content = content
        self.metadata = metadata
        self.vector_score = vector_score
        self.vector_ranks = vector_ranks or []
        self.bm25_score = 0.0
        self.numeric_score = 0.0
        self.rrf_score = 0.0
        self.final_score = 0.0


class HybridSourceNode:
    """兼容 LlamaIndex SourceNode 的轻量包装"""
    def __init__(self, content: str, metadata: dict, score: float = 0.0):
        self._content = content
        self.metadata = metadata
        self.score = score

    def get_content(self) -> str:
        return self._content


class HybridQueryResponse:
    """兼容 LlamaIndex QueryResponse 的轻量包装"""
    def __init__(self, nodes: list, source_nodes: list):
        self.nodes = nodes
        self.source_nodes = source_nodes

    def __str__(self) -> str:
        return f"HybridQueryResponse({len(self.nodes)} nodes)"


# ==========================================
# 主入口
# ==========================================

def get_query_engine(collection_name: str = "catl_report", pdf_filename: str = None,
                    use_hybrid: bool = True):
    """
    获取查询引擎（懒加载模式）。
    首次调用时初始化，后续调用返回缓存。

    Args:
        collection_name: ChromaDB 集合名，不同公司/文档使用不同集合
        pdf_filename: PDF 文件名（位于 data/ 目录下），为 None 时默认为宁德时代2025年度报告.pdf
        use_hybrid: 是否启用混合检索（默认 True）
    """
    if collection_name in _QUERY_ENGINE_CACHE:
        return _QUERY_ENGINE_CACHE[collection_name]

    _setup_settings()
    embed_model = _get_embedding_model()

    from llama_index.core import Settings as _Settings
    _Settings.embed_model = embed_model

    logger.info(f"正在检查或初始化本地知识库状态 [collection: {collection_name}]...")
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(current_dir, CHROMA_DB_PATH)
    data_path = os.path.join(current_dir, DATA_PATH)
    if pdf_filename is None:
        pdf_filename = "宁德时代2025年度报告.pdf"
    pdf_path = os.path.join(data_path, pdf_filename)

    db = chromadb.PersistentClient(path=db_path)
    collections = [c.name for c in db.list_collections()]

    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # llama-index 0.14.x 的 StorageContext.from_defaults 不接受 embed_model 参数
    # embed_model 应通过 Settings.embed_model 或 VectorStoreIndex.from_vector_store 传入
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection_name in collections and chroma_collection.count() > 0:
        logger.info(f"检测到已有的持久化向量库 (共 {chroma_collection.count()} 个切片)，直接加载")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context
        )
    else:
        logger.info(f"未检测到缓存，正在解析财报: {pdf_path}")
        logger.info("提示：因为启用了防封禁限流，生成向量大概需要几分钟，请耐心等待...")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"找不到财报文件，请确保路径正确: {pdf_path}")

        loader = PyMuPDFReader()
        documents = loader.load(file_path=pdf_path)

        # 构建 page_texts: [(page_num, text), ...]
        page_texts = []
        for i, doc in enumerate(documents):
            page_num = i + 1
            page_texts.append((page_num, doc.text))
            doc.metadata["page_label"] = str(page_num)

        # 使用财报专用分块器
        logger.info("使用 FinancialReportChunker 进行财报专用分块...")
        chunk_candidates = chunk_financial_report(
            page_texts,
            max_chunk_tokens=600,
            min_chunk_tokens=100,
            overlap_tokens=80,
        )

        logger.info(f"分块完成: {len(chunk_candidates)} 个节点 "
                    f"(表格 {sum(1 for c in chunk_candidates if c.is_table)} 个, "
                    f"文本 {sum(1 for c in chunk_candidates if not c.is_table)} 个)")

        # 将 ChunkCandidate 转换为 LlamaIndex Node
        from llama_index.core.schema import TextNode
        nodes = []
        for j, cc in enumerate(chunk_candidates):
            node = TextNode(
                text=cc.text,
                metadata={
                    # 以下字段会被 ChromaDB 单独存储（用于过滤/显示）
                    "page_label": str(cc.page_start),
                    "page_start": cc.page_start,
                    "page_end": cc.page_end,
                    "is_table": cc.is_table,
                    "file_path": pdf_path,
                    "file_name": os.path.basename(pdf_path),
                    # 章节名称（如果有的话）
                    **({"section_name": cc.section_name} if cc.section_name else {}),
                },
                id_=f"node_{collection_name}_{j}",
            )
            nodes.append(node)

        index = VectorStoreIndex(
            nodes,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=True
        )
        logger.info("财报解析与向量化落盘完成")

    # 包装为混合检索引擎或标准引擎
    if use_hybrid:
        # 检查是否启用 Cross-Encoder（环境变量控制，可选功能）
        use_ce = os.environ.get("USE_CROSS_ENCODER", "0") == "1"
        engine = HybridQueryEngine(
            index=index,
            similarity_top_k=8,
            hybrid_k=15,
            vector_weight=0.6,
            bm25_weight=0.3,
            numeric_weight=0.1,
            use_cross_encoder=use_ce,
        )
        ce_status = " + Cross-Encoder 精排" if use_ce else ""
        logger.info(f"混合检索引擎已就绪 (向量60% + BM25 30% + 数字匹配10%{ce_status})")
    else:
        engine = index.as_query_engine(
            similarity_top_k=8,
            response_mode="compact"
        )
        logger.info("标准向量检索引擎已就绪")

    _QUERY_ENGINE_CACHE[collection_name] = engine
    return engine


def list_available_collections():
    """列出 ChromaDB 中已有的所有集合（即已入库的财报）"""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(current_dir, CHROMA_DB_PATH)
    db = chromadb.PersistentClient(path=db_path)
    return [{"name": c.name, "count": c.count()} for c in db.list_collections()]


def clear_cache(collection_name: str = None):
    """
    清除查询引擎缓存（用于测试或热重载）

    Args:
        collection_name: 只清除指定集合的缓存，为 None 时清除全部
    """
    global _QUERY_ENGINE_CACHE
    if collection_name is None:
        _QUERY_ENGINE_CACHE.clear()
        _setup_settings()
        _get_embedding_model()
        logger.info("全部查询引擎缓存已清除")
    elif collection_name in _QUERY_ENGINE_CACHE:
        del _QUERY_ENGINE_CACHE[collection_name]
        logger.info(f"查询引擎缓存已清除: {collection_name}")


@tool
def analyze_catl_report(query: str, collection_name: str = "catl_report") -> str:
    """
    当需要回答关于【上市公司】的财务数据、业务营收、毛利率、产能、战略规划、
    技术研发投入或具体历史年份的财报细节时，必须调用此工具。
    输入参数 query 应该是具体且清晰的查询问题。
    collection_name: 知识库集合名称，不同公司使用不同集合（默认 catl_report）。
    """
    logger.info(f"深度研究员正在翻阅财报检索 [库: {collection_name}]: '{query}'")
    try:
        engine = get_query_engine(collection_name=collection_name)

        # 兼容混合引擎和标准引擎
        if isinstance(engine, HybridQueryEngine):
            response = engine.query(query)
            source_nodes = response.source_nodes
            final_answer = _format_hybrid_response(response)
        else:
            response = engine.query(query)
            source_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
            final_answer = str(response)

        final_res = f"【结论】: {final_answer}\n\n【知识库原始证据】:\n"

        if source_nodes:
            for i, node in enumerate(source_nodes[:3]):
                file_name = node.metadata.get('file_name', node.metadata.get('file_path', '未知文件'))
                if file_name != '未知文件':
                    file_name = os.path.basename(file_name)

                page_label = node.metadata.get('page_label', node.metadata.get('source', node.metadata.get('page_num', node.metadata.get('page', '未知页码'))))
                snippet = node.get_content().replace('\n', ' ')[:200] + "..."  # 从 120 增加到 200

                score_info = ""
                if hasattr(node, 'score'):
                    score_info = f" [相关度: {node.score:.3f}]"

                final_res += f"来源 {i+1}: `{file_name}` (第 {page_label} 页){score_info}\n"
                final_res += f"原文切片: {snippet}\n\n"
        else:
            final_res += "> 未能获取到具体的底层原文切片。\n"

        return final_res

    except Exception as e:
        logger.error(f"检索财报时发生内部错误: {e}")
        return f"检索财报时发生内部错误: {str(e)}"


def _format_hybrid_response(response: HybridQueryResponse) -> str:
    """将混合检索结果格式化为可读的 LLM 上下文"""
    parts = []
    for i, node in enumerate(response.source_nodes):
        content = node.get_content()
        metadata = node.metadata
        page = metadata.get('page_label', metadata.get('source', metadata.get('page_num', '未知')))
        parts.append(f"[证据 {i+1} (第{page}页, 相关度:{node.score:.3f})]\n{content}")
    return "\n\n---\n\n".join(parts)