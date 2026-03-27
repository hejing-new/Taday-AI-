import os
import time
import chromadb
from typing import List
from dotenv import load_dotenv
from langchain_core.tools import tool
from tenacity import retry, wait_exponential, stop_after_attempt

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

# ================= 1. 初始化环境与配置 =================
load_dotenv()
api_key = os.getenv("api_key")
base_url = os.getenv("base_url", "https://api.siliconflow.cn/v1")

if not api_key:
    raise ValueError("⚠️ 找不到 api_key，请检查 .env 文件！")

print("🔌 正在初始化硅基流动 (SiliconFlow) 大模型引擎...")

# 1. 配置大模型 (Qwen2.5-7B)
Settings.llm = OpenAILike(
    model="Qwen/Qwen2.5-7B-Instruct", 
    api_key=api_key, 
    api_base=base_url, 
    max_tokens=1024,
    is_chat_model=True,
    context_window=32768
)

# === 🌟 核心升级 1：企业级 API 限流与指数退避重试 ===
class SafeSiliconFlowEmbedding(OpenAIEmbedding):
    """
    使用 Tenacity 实现优雅的重试机制。
    遇到 403/429 时，休眠时间会按指数级增长 (2s, 4s, 8s...)，最高重试 5 次。
    """
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15), 
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 保持极轻微的基础限流，避免一上来就触碰 RPM 红线
        time.sleep(0.5)
        return super()._get_text_embeddings(texts)

# 2. 配置原生 Embedding 模型，并穿上我们的“防弹衣”
Settings.embed_model = SafeSiliconFlowEmbedding(
    model_name="BAAI/bge-m3", 
    api_key=api_key, 
    api_base=base_url,
    embed_batch_size=50 # 控制打包体积，防封禁
)

# ================= 2. 构建或加载本地知识库 (懒加载模式) =================

# 🌟 核心升级 2：全局缓存，防止一 import 模块就阻塞主线程
_QUERY_ENGINE_CACHE = None

def get_query_engine():
    global _QUERY_ENGINE_CACHE
    if _QUERY_ENGINE_CACHE is not None:
        return _QUERY_ENGINE_CACHE
        
    print("⏳ 正在检查或初始化本地知识库状态...")
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(current_dir, "chroma_db")
    data_path = os.path.join(current_dir, "data")
    pdf_path = os.path.join(data_path, "宁德时代2025年度报告.pdf") 

    # 1. 连接 ChromaDB
    db = chromadb.PersistentClient(path=db_path)
    collections = [c.name for c in db.list_collections()]
    collection_name = "catl_report"
    
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2. 判断是加载已有缓存，还是重新切片入库
    if collection_name in collections and chroma_collection.count() > 0:
        print(f"✅ 检测到已有的持久化向量库 (共 {chroma_collection.count()} 个切片)，直接加载！(秒开)")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
    else:
        print(f"📄 未检测到缓存，正在解析财报: {pdf_path}")
        print("💡 提示：因为启用了防封禁限流，生成向量大概需要几分钟，请耐心等待...")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"找不到财报文件，请确保路径正确: {pdf_path}")

        # 🌟 核心升级 3：使用 SentenceSplitter 精细化保护财务表格语义
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        loader = PyMuPDFReader()
        documents = loader.load(file_path=pdf_path)
        
        # 提取 Node 并入库
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context,
            show_progress=True 
        )
        print("✅ 财报解析与向量化落盘完成！")

    # 🌟 核心升级 4：优化检索策略，防噪音干扰
    _QUERY_ENGINE_CACHE = index.as_query_engine(
        similarity_top_k=8,        # 缩减 Top K，避免将无关的页眉页脚喂给大模型
        response_mode="compact"    # 针对事实数据的浓缩模式，比 tree 结构更严谨
    )
    return _QUERY_ENGINE_CACHE

report_query_engine = get_query_engine()

# ================= 3. 核心：封装为 LangGraph 工具 =================
@tool
def analyze_catl_report(query: str) -> str:
    """
    当需要回答关于【宁德时代】（CATL）的财务数据、业务营收、毛利率、产能、战略规划、
    技术研发投入或具体历史年份的财报细节时，必须调用此工具。
    输入参数 query 应该是具体且清晰的查询问题，例如“宁德时代2025年的动力电池系统毛利率是多少？”
    """
    print(f"\n[🔧 Tool invoked] 深度研究员正在翻阅宁德时代财报检索: '{query}'")
    try:
        # 只有真正调用工具时，才会去建立数据库连接和引擎
        engine = get_query_engine()
        response = engine.query(query)
        
        # 🌟 核心修复：不仅要答案，还要把底层的来源切片一起打包返回给路由中枢！
        final_res = f"【结论】: {str(response)}\n\n【📚 知识库原始证据】:\n"
        
        # 解析底层数据来源 (Source Nodes)
        if hasattr(response, "source_nodes") and response.source_nodes:
            # 只取相关度最高的 Top 3 切片展示，防止把前端撑爆
            for i, node in enumerate(response.source_nodes[:3]):
                file_name = node.metadata.get('file_name', '未知文件')
                page_label = node.metadata.get('page_label', '未知页码')
                # 截取原文摘要，去掉换行符让排版更紧凑
                snippet = node.get_content().replace('\n', ' ')[:120] + "..."
                
                final_res += f"📄 **来源 {i+1}**: `{file_name}` (第 {page_label} 页)\n"
                final_res += f"🔎 **原文切片**: {snippet}\n\n"
        else:
            final_res += "> ⚠️ 未能获取到具体的底层原文切片。\n"
            
        return final_res
        
    except Exception as e:
        return f"检索财报时发生内部错误: {str(e)}"

# ================= 4. 本地模块测试 =================
if __name__ == "__main__":
    print("\n--- 深度研究员工具独立测试 ---")
    test_query = "总结一下宁德时代在动力电池领域的营收情况和市占率。"
    
    # 模拟 LangGraph 传入参数的格式
    result = analyze_catl_report.invoke({"query": test_query})
    print("\n📝 最终检索生成的答案：\n")
    print(result)