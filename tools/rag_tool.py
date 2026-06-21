"""
RAG 财报检索工具 (CATL)

工厂模式 + 懒加载，第一次调用 get_query_engine() 时才初始化连接。
"""
import sys
import os
import io
import time
import chromadb
from typing import List
from dotenv import load_dotenv
from langchain_core.tools import tool
from tenacity import retry, wait_exponential, stop_after_attempt

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

from config import API_KEY, BASE_URL, CHAT_MODEL, EMBED_MODEL, CHROMA_DB_PATH, DATA_PATH
from logger import logger

# UTF-8 环境防护
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 全局缓存（懒加载，第一次调用时才初始化）
# 结构: {collection_name: query_engine}
_QUERY_ENGINE_CACHE = {}


def _setup_settings():
    """配置 LlamaIndex 全局设置（幂等，可安全多次调用）"""
    if Settings.llm is not None:
        return  # 已配置，跳过
    logger.info("初始化硅基流动 (SiliconFlow) 大模型引擎...")
    Settings.llm = OpenAILike(
        model=CHAT_MODEL,
        api_key=API_KEY,
        api_base=BASE_URL,
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
    """获取 Embedding 模型（带限流保护）"""
    if Settings.embed_model is not None:
        return Settings.embed_model
    Settings.embed_model = SafeSiliconFlowEmbedding(
        model_name=EMBED_MODEL,
        api_key=API_KEY,
        api_base=BASE_URL,
        embed_batch_size=50
    )
    return Settings.embed_model


def get_query_engine(collection_name: str = "catl_report", pdf_filename: str = None):
    """
    获取查询引擎（懒加载模式）。
    首次调用时初始化，后续调用返回缓存。

    Args:
        collection_name: ChromaDB 集合名，不同公司/文档使用不同集合
        pdf_filename: PDF 文件名（位于 data/ 目录下），为 None 时默认为宁德时代2025年度报告.pdf
    """
    if collection_name in _QUERY_ENGINE_CACHE:
        return _QUERY_ENGINE_CACHE[collection_name]

    _setup_settings()
    _get_embedding_model()

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
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection_name in collections and chroma_collection.count() > 0:
        logger.info(f"检测到已有的持久化向量库 (共 {chroma_collection.count()} 个切片)，直接加载")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
    else:
        logger.info(f"未检测到缓存，正在解析财报: {pdf_path}")
        logger.info("提示：因为启用了防封禁限流，生成向量大概需要几分钟，请耐心等待...")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"找不到财报文件，请确保路径正确: {pdf_path}")

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        loader = PyMuPDFReader()
        documents = loader.load(file_path=pdf_path)

        for i, doc in enumerate(documents):
            doc.metadata["page_label"] = str(i + 1)

        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        logger.info("财报解析与向量化落盘完成")

    engine = index.as_query_engine(
        similarity_top_k=8,
        response_mode="compact"
    )
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
        response = engine.query(query)

        final_res = f"【结论】: {str(response)}\n\n【知识库原始证据】:\n"

        if hasattr(response, "source_nodes") and response.source_nodes:
            for i, node in enumerate(response.source_nodes[:3]):
                file_name = node.metadata.get('file_name', node.metadata.get('file_path', '未知文件'))
                if file_name != '未知文件':
                    file_name = os.path.basename(file_name)

                page_label = node.metadata.get('page_label', node.metadata.get('page_num', node.metadata.get('page', '未知页码')))
                snippet = node.get_content().replace('\n', ' ')[:120] + "..."

                final_res += f"来源 {i+1}: `{file_name}` (第 {page_label} 页)\n"
                final_res += f"原文切片: {snippet}\n\n"
        else:
            final_res += "> 未能获取到具体的底层原文切片。\n"

        return final_res

    except Exception as e:
        logger.error(f"检索财报时发生内部错误: {e}")
        return f"检索财报时发生内部错误: {str(e)}"
