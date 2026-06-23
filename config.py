"""
Taday 金融智能体 — 统一配置管理

所有全局配置集中在此，其他模块统一从 config.py 导入，避免散落各处。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 🌐 API 配置
# ==========================================
BASE_URL = os.getenv("base_url", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("api_key", "")
LONGCAT_API_KEY = os.getenv("LONGCAT_API_KEY", "")
LONGCAT_BASE_URL = os.getenv("LONGCAT_BASE_URL", "https://api.longcat.chat/openai/v1")

# LLM config for rag_tool (must support LongCat-2.0-Preview)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", LONGCAT_BASE_URL)
LLM_API_KEY = os.getenv("LLM_API_KEY", LONGCAT_API_KEY)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ==========================================
# 🤖 模型配置
# ==========================================
# 对话模型（负责工具调用和日常对话）
CHAT_MODEL = os.getenv("CHAT_MODEL", "LongCat-2.0-Preview")
# 裁判/出题模型（用于 Ragas 评测和自动巡检）
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
# 修复模型（用于 Bad Case 自愈）
HEAL_MODEL = os.getenv("HEAL_MODEL", "LongCat-Flash-Thinking-2601")
# Embedding 模型
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# ==========================================
# 🔐 管理员认证
# ==========================================
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "")
if not ADMIN_PASS:
    import warnings
    warnings.warn(
        "⚠️ 安全警告: ADMIN_PASS 未设置！请在 .env 文件中配置管理员密码。"
        "启动请设置环境变量: ADMIN_PASS=your_secure_password"
    )

# ==========================================
# 🚪 服务端口
# ==========================================
PORT_CHAT = 7860          # C 端 Gradio 对话界面
PORT_ADMIN = 7861         # B 端 Gradio 管理后台
PORT_API = 8002           # FastAPI 对话后端
PORT_ADMIN_API = 8004     # FastAPI 管理后台 API

# ==========================================
# 📡 外部 API 地址
# ==========================================
# 支持 Docker Compose 服务名: 环境变量 DOCKER_HOST 存在时使用服务名
def _get_service_url(port: int) -> str:
    """获取服务 URL，Docker 环境使用服务名，本地使用 127.0.0.1"""
    if os.getenv("DOCKER_HOST"):
        # Docker Compose 网络: 使用服务名 (api / admin-api)
        service = "api" if port == PORT_API else "admin-api"
        return f"http://{service}:{port}"
    return f"http://127.0.0.1:{port}"

API_URL = _get_service_url(PORT_API)
ADMIN_API_URL = _get_service_url(PORT_ADMIN_API)

# ==========================================
# 🗄️ 数据库与存储路径
# ==========================================
DB_FILE = "knowledge_draft.db"
CHROMA_DB_PATH = "chroma_db"
DATA_PATH = "data"
TEMP_STORAGE_PATH = "temp_storage"

# ==========================================
# 📋 文件上传限制
# ==========================================
ALLOWED_CONTENT_TYPES = ["application/pdf"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# ==========================================
# 📊 BI 看板
# ==========================================
ANALYTICS_PAGE_SIZE = 50  # 日志分页大小

# ==========================================
# 🔧 Ragas 评测配置
# ==========================================
RAGAS_DEFAULT_QUESTIONS = 3  # 默认出题数
