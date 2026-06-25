import sys
import os
import io

# Windows 编码修复：必须在所有 import 之前设置
if sys.platform == 'win32':
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import uvicorn
import secrets
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import UploadFile, File
from llama_index.core import Document, SummaryIndex
from llama_index.readers.file import PyMuPDFReader
import shutil

from config import (
    ADMIN_USER, ADMIN_PASS, ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE,
    TEMP_STORAGE_PATH
)

# ================= 安全认证配置 =================
security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """验证管理员身份"""
    is_user_correct = secrets.compare_digest(credentials.username, ADMIN_USER)
    is_pass_correct = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (is_user_correct and is_pass_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未授权访问：管理员凭据无效",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# 🚀 全局内存存储：存放每个 Session 的临时查询引擎（LRU 淘汰，防内存泄漏）
from collections import OrderedDict
_temp_engines = OrderedDict()
MAX_TEMP_ENGINES = 50  # 最多保留 50 个会话的临时引擎
_TEMP_ENGINE_TTL = 3600  # 1 小时过期


def _get_temp_engine(session_id: str):
    """获取临时引擎，不存在返回 None"""
    return _temp_engines.get(session_id)


def _set_temp_engine(session_id: str, engine):
    """存储临时引擎，超出限制时淘汰最早的"""
    _temp_engines[session_id] = engine
    # LRU 淘汰：超出限制则移除最早的条目
    while len(_temp_engines) > MAX_TEMP_ENGINES:
        _temp_engines.popitem(last=False)


# 把根目录加入路径，导入你的真实后端
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.rag_tool import get_query_engine

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
# 上传文件限制：仅允许 PDF，最大 50MB
ALLOWED_CONTENT_TYPES = ["application/pdf"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# 1. 新增：临时文档上传接口（需认证）
@app.post("/api/v1/upload_temp")
async def upload_temp_document(session_id: str, file: UploadFile = File(...), username: str = Depends(verify_admin)):
    # 文件类型校验
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}，仅允许 PDF")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="文件扩展名必须为 .pdf")

    try:
        # 1. 保存临时文件（先读入内存检查大小）
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"文件过大（{len(content)//1024//1024}MB），最大允许 50MB")

        temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # 2. 解析 PDF 并构建内存索引 (SummaryIndex 适合单文档快速检索)
        loader = PyMuPDFReader()
        documents = loader.load_data(file_path=file_path)

        # 构建一个临时的内存引擎（带 LRU 淘汰）
        index = SummaryIndex.from_documents(documents)
        _set_temp_engine(session_id, index.as_query_engine())

        return {"status": "success", "message": f"文档 {file.filename} 已挂载至会话 {session_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse, summary="提交问题并获取大模型回答与溯源数据")
async def chat_endpoint(request: ChatRequest):
    """
    接收前端传来的纯文本问题，调用底层的 LlamaIndex / LangGraph 引擎进行检索和生成。
    """
    try:
        # 🚀 核心逻辑：优先检查是否有临时文档引擎
        engine = _get_temp_engine(request.session_id)
        if engine is not None:
            print(f"🔍 使用临时私有引擎处理会话: {request.session_id}")
        else:
            print(f"📚 使用公共财报库处理会话: {request.session_id}")
            engine = get_query_engine()

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
# 5. 对话历史管理接口
# ==========================================
from utils.conversation_store import (
    get_history, clear_history, get_all_sessions, get_session_stats
)


@app.get("/api/v1/history/{session_id}", summary="获取会话历史")
async def get_conversation_history(session_id: str, limit: int = 50):
    """获取指定会话的对话历史（用于前端恢复对话）"""
    try:
        history = get_history(session_id, limit=limit)
        return {"session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/history/{session_id}", summary="清空会话历史")
async def clear_conversation_history(session_id: str):
    """清空指定会话的对话历史"""
    try:
        clear_history(session_id)
        return {"status": "success", "message": f"会话 {session_id} 已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sessions", summary="获取所有会话列表（管理员）")
async def list_sessions(username: str = Depends(verify_admin)):
    """获取所有活跃会话列表"""
    try:
        sessions = get_all_sessions()
        stats = get_session_stats()
        return {"sessions": sessions, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 6. 启动服务 (仅在直接运行此文件时执行)
# ==========================================
if __name__ == "__main__":
    from config import PORT_API
    print(f"🚀 正在启动 Taday 后端 API 服务 (端口: {PORT_API})...")
    # 注意：在生产环境中通常用 gunicorn 启动
    uvicorn.run(app, host="127.0.0.1", port=PORT_API)