import os
import time
import chromadb
from typing import List
from dotenv import load_dotenv
from langchain_core.tools import tool

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.openai_like import OpenAILike
# 在文件顶部引入新的 Reader
from llama_index.readers.file import PyMuPDFReader

# ================= 1. 初始化环境与配置 =================
load_dotenv()
api_key = os.getenv("api_key")
base_url = os.getenv("base_url", "https://api.siliconflow.cn/v1")

if not api_key:
    raise ValueError("⚠️ 找不到 api_key，请检查 .env 文件！")

print("🔌 正在连接硅基流动 (SiliconFlow) API...")

# 1. 配置大模型 (Qwen2.5-7B)
# 1. 配置大模型 (使用 OpenAILike 彻底绕过官方白名单校验)
Settings.llm = OpenAILike(
    model="Qwen/Qwen2.5-7B-Instruct", 
    api_key=api_key, 
    api_base=base_url, 
    max_tokens=1024,
    is_chat_model=True,   # 👈 明确告诉框架这是一个对话模型
    context_window=32768  # 👈 手动指定上下文长度，防止框架瞎猜报错
)

# === 🌟 核心破解：企业级 API 限流与自动重试拦截器 ===
class RateLimitedEmbedding(LangchainEmbedding):
    """
    拦截器：用于防止未实名的白嫖账号被硅基流动封禁。
    当遇到 403 或 RPM 限制时，自动休眠并重试，而不是让程序直接崩溃。
    """
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # 每次发请求前强制停顿 1.5 秒，极大地降低 RPM (每分钟请求数)
                time.sleep(1.5)
                return super()._get_text_embeddings(texts)
            except Exception as e:
                # 如果依然被抓住了，就惩罚性休眠更长时间
                if "RPM limit" in str(e) or "403" in str(e) or "429" in str(e):
                    wait_time = 10 * (attempt + 1)
                    print(f"\n⚠️ 触发平台并发限制，触发器已拦截！休眠 {wait_time} 秒后自动重试...")
                    time.sleep(wait_time)
                else:
                    raise e
        return super()._get_text_embeddings(texts)

# 配置底层 LangChain Embedding 模型
lc_embed_model = OpenAIEmbeddings(
    model="BAAI/bge-m3", 
    openai_api_key=api_key, 
    openai_api_base=base_url,
    check_embedding_ctx_length=False 
)

# 给模型穿上我们的“防弹衣”并注入 LlamaIndex
Settings.embed_model = RateLimitedEmbedding(lc_embed_model)
# 强制调小打包发送的体积，进一步降低被封禁的概率
Settings.embed_batch_size = 50 


# ================= 2. 构建或加载本地知识库 =================
def get_query_engine():
    print("⏳ 正在检查本地知识库状态...")
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(current_dir, "chroma_db")
    data_path = os.path.join(current_dir, "data")

    db = chromadb.PersistentClient(path=db_path)
    collections = [c.name for c in db.list_collections()]
    collection_name = "catl_report"
    
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection_name in collections and chroma_collection.count() > 0:
        print("✅ 检测到已有的持久化向量库，直接加载！(秒开)")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
    else:
        print(f"📄 未检测到缓存，正在解析 {data_path} 下的宁德时代财报...")
        print("💡 提示：因为启用了防封禁限流，生成 1300+ 个向量大概需要 3-5 分钟，请耐心等待...")
        # documents = SimpleDirectoryReader(data_path).load_data()
        pdf_path = os.path.join(data_path, "宁德时代2025年度报告.pdf") 
        loader = PyMuPDFReader()
        documents = loader.load(file_path=pdf_path)
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            show_progress=True 
        )
        print("✅ 财报解析与向量化落盘完成！")

    return index.as_query_engine(
        similarity_top_k=15, 
        response_mode="tree_summarize"
    )

report_query_engine = get_query_engine()

# ================= 3. 核心：封装为 LangGraph 工具 =================
@tool
def analyze_catl_report(query: str) -> str:
    """
    当你需要回答关于【宁德时代】（CATL）的财务数据、业务营收、毛利率、产能、战略规划、
    技术研发投入或具体历史年份的财报细节时，必须调用此工具。
    输入参数 query 应该是具体且清晰的查询问题，例如“宁德时代2025年的动力电池系统毛利率是多少？”
    """
    print(f"\n[🔧 Tool invoked] 深度研究员正在翻阅宁德时代财报检索: '{query}'")
    try:
        response = report_query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"检索财报时发生错误: {str(e)}"

# ================= 4. 本地模块测试 =================
if __name__ == "__main__":
    print("\n--- 深度研究员工具独立测试 ---")
    test_query = "总结一下宁德时代在动力电池领域的营收情况和市占率。"
    result = analyze_catl_report.invoke({"query": test_query})
    print("\n📝 最终检索生成的答案：\n")
    print(result)