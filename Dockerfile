# ==========================================
# Taday 金融智能体 — 多阶段构建 Dockerfile
# ==========================================

# --- 阶段 1: 依赖缓存 ---
FROM python:3.11-slim AS deps

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirement.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir --prefix=/install -r requirement.txt

# --- 阶段 2: 运行时镜像 ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# 系统依赖（运行时需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制 Python 依赖
COPY --from=deps /install /usr/local

# 复制项目文件
COPY . .

# 创建数据目录
RUN mkdir -p /app/chroma_db /app/data /app/temp_storage /app/logs

# 环境变量
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT_API:-8002}/docs || exit 1

# 默认启动命令（会被 docker-compose 覆盖）
CMD ["python", "app_backend.py"]
