"""
文档管理相关路由（B 端）
"""
import os
import uuid
import sqlite3
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from config import DB_FILE, ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE, CHROMA_DB_PATH, DATA_PATH, TEMP_STORAGE_PATH
from logger import logger

router = APIRouter()

# 数据模型
class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    created_at: str
    chunk_count: int

class ChunkUpdateParams(BaseModel):
    new_text: str

class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    text_content: str
    chunk_index: int
    status: str


# ==========================================
# 文档解析后台任务
# ==========================================

def process_document_task(doc_id: str, temp_path: str, filename: str):
    """后台任务：PDF 解析 → 切块 → 入库"""
    logger.info(f"后台任务开始，正在切分文档: {filename}")
    try:
        loader = PyMuPDFReader()
        docs = loader.load_data(file_path=temp_path)

        parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(docs)

        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()

        for i, node in enumerate(nodes):
            chunk_id = f"chk_{uuid.uuid4().hex[:8]}"
            cursor.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, node.get_content(), i, 'active')
            )

        cursor.execute("UPDATE documents SET status = 'pending' WHERE doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()
        logger.info(f"后台任务完成，{filename} 切分完毕，共 {len(nodes)} 个切片")

    except Exception as e:
        logger.error(f"后台任务失败，{filename} 解析异常: {e}")
        conn = sqlite3.connect(DB_FILE)
        conn.execute("UPDATE documents SET status = 'failed' WHERE doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================
# 文档管理 API
# ==========================================

@router.post("/admin/api/upload", summary="极速上传 (触发异步解析)")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 文件类型校验
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}，仅允许 PDF")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="文件扩展名必须为 .pdf")

    doc_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_STORAGE_PATH, f"{doc_id}_{file.filename}")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"文件过大（{len(content)//1024//1024}MB），最大允许 50MB")

    os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
    with open(temp_path, "wb") as buffer:
        buffer.write(content)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents VALUES (?, ?, ?, ?)",
                   (doc_id, file.filename, 'processing', now_str))
    conn.commit()
    conn.close()

    background_tasks.add_task(process_document_task, doc_id, temp_path, file.filename)

    return {
        "status": "success",
        "doc_id": doc_id,
        "message": "文件已接收，后台正在拼命解析中，请稍后在看板查看状态..."
    }


@router.get("/admin/api/docs", response_model=List[DocumentResponse], summary="获取全局文档看板")
def get_all_documents():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT d.doc_id, d.filename, d.status, d.created_at, COUNT(c.chunk_id)
        FROM documents d
        LEFT JOIN chunks c ON d.doc_id = c.doc_id AND c.status = 'active'
        GROUP BY d.doc_id
        ORDER BY d.created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    return [
        DocumentResponse(doc_id=r[0], filename=r[1], status=r[2], created_at=r[3], chunk_count=r[4])
        for r in rows
    ]


@router.get("/admin/api/docs/{doc_id}/chunks", response_model=List[ChunkResponse])
def get_chunks(doc_id: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,))
    rows = cursor.fetchall()
    conn.close()
    return [ChunkResponse(chunk_id=r[0], doc_id=r[1], text_content=r[2], chunk_index=r[3], status=r[4]) for r in rows]


@router.put("/admin/api/chunks/{chunk_id}")
def update_chunk(chunk_id: str, params: ChunkUpdateParams):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE chunks SET text_content = ? WHERE chunk_id = ?", (params.new_text, chunk_id))
    conn.commit()
    conn.close()
    return {"status": "success"}


@router.delete("/admin/api/chunks/{chunk_id}")
def delete_chunk(chunk_id: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE chunks SET status = 'deleted' WHERE chunk_id = ?", (chunk_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}


@router.post("/admin/api/docs/{doc_id}/publish")
def publish_document(doc_id: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE documents SET status = 'published' WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "已成功发布入库！"}
