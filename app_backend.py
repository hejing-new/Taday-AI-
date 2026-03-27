import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import UploadFile, File
from llama_index.core import Document, SummaryIndex
from llama_index.readers.file import PyMuPDFReader
import shutil

# 🚀 全局内存存储：存放每个 Session 的临时查询引擎
temp_engines = {}


# 把根目录加入路径，导入你的真实后端
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.rag_tool import report_query_engine

# ==========================================
# 1. 初始化 FastAPI 实例
# ==========================================
app = FastAPI(
    title="Taday 智能助手核心后端 API",
    description="提供企业级金融 RAG 检索、对话及溯源服务",
    version="1.0.0"
)

# ==========================================
# 2. 定义数据模型 (Schema) - 规范输入输出格式
# ==========================================
class ChatRequest(BaseModel):
    query: str = Field(..., description="用户的金融提问")
    session_id: Optional[str] = Field("default", description="会话ID，用于将来扩展多轮记忆")

class SourceNodeModel(BaseModel):
    chunk_id: str
    text_preview: str
    similarity_score: float

class ChatResponse(BaseModel):
    answer: str = Field(..., description="大模型生成的最终文本")
    sources: List[SourceNodeModel] = Field(default=[], description="参考的原文切片列表，用于数据溯源")

# ==========================================
# 3. 核心业务接口定义 (路由)
# ==========================================
# 1. 新增：临时文档上传接口
@app.post("/api/v1/upload_temp")
async def upload_temp_document(session_id: str, file: UploadFile = File(...)):
    try:
        # 1. 保存临时文件
        temp_dir = f"./temp_storage/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 解析 PDF 并构建内存索引 (SummaryIndex 适合单文档快速检索)
        loader = PyMuPDFReader()
        documents = loader.load_data(file_path=file_path)
        
        # 构建一个临时的内存引擎
        index = SummaryIndex.from_documents(documents)
        temp_engines[session_id] = index.as_query_engine()

        return {"status": "success", "message": f"文档 {file.filename} 已挂载至会话 {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse, summary="提交问题并获取大模型回答与溯源数据")
async def chat_endpoint(request: ChatRequest):
    """
    接收前端传来的纯文本问题，调用底层的 LlamaIndex / LangGraph 引擎进行检索和生成。
    """
    try:
        # 🚀 核心逻辑：优先检查是否有临时文档引擎
        if request.session_id in temp_engines:
            print(f"🔍 使用临时私有引擎处理会话: {request.session_id}")
            engine = temp_engines[request.session_id]
        else:
            print(f"📚 使用公共财报库处理会话: {request.session_id}")
            engine = report_query_engine

        response = engine.query(request.query)
        
        # 2. 提取大模型文本
        final_answer = str(response)
        
        # 3. 提取底层溯源数据并清洗
        sources_list = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                # 获取分数
                score = getattr(node, "score", 0.0)
                if score is None:
                    score = 0.0
                    
                # 获取切片预览
                content = node.node.get_content()
                preview = content[:250] + "..." if len(content) > 250 else content
                
                sources_list.append(SourceNodeModel(
                    chunk_id=node.node.node_id,
                    text_preview=preview,
                    similarity_score=round(float(score), 4)
                ))
                
        print("[API 处理完毕] 成功返回答案与溯源数据")
        
        # 4. 构造标准 JSON 返回给前端
        return ChatResponse(
            answer=final_answer,
            sources=sources_list
        )

    except Exception as e:
        print(f"❌ [API 异常] {str(e)}")
        # 抛出标准的 HTTP 500 错误
        raise HTTPException(status_code=500, detail=f"底层 AI 引擎处理失败: {str(e)}")


# ==========================================
# 4. 启动服务 (仅在直接运行此文件时执行)
# ==========================================
if __name__ == "__main__":
    print("🚀 正在启动 Taday 后端 API 服务 (端口: 8000)...")
    # 注意：在生产环境中通常用 gunicorn 启动
    uvicorn.run(app, host="127.0.0.1", port=8000)