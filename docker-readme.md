# Docker 部署指南

## 快速开始

### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并填写 API Key：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 2. 构建并启动

```bash
# 构建镜像并启动所有服务
docker-compose up -d --build

# 只启动 API 服务
docker-compose up -d --build api admin-api

# 只启动前端界面
docker-compose up -d --build chat admin
```

### 3. 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| C 端对话界面 | http://localhost:7860 | 用户对话 |
| B 端管理后台 | http://localhost:7861 | 管理控制台 |
| 对话 API | http://localhost:8002/docs | API 文档 |
| 管理 API | http://localhost:8004/docs | API 文档 |

### 4. 常用命令

```bash
# 查看日志
docker-compose logs -f api
docker-compose logs -f chat

# 停止服务
docker-compose down

# 停止并删除数据卷（谨慎！）
docker-compose down -v

# 重启服务
docker-compose restart

# 查看服务状态
docker-compose ps
```

### 5. 数据持久化

以下数据通过 Docker Volume 持久化：

| Volume | 内容 | 容器内路径 |
|--------|------|-----------|
| `chroma_data` | ChromaDB 向量库 | `/app/chroma_db` |
| `app_data` | 上传的 PDF 文件 | `/app/data` |
| `temp_data` | 临时文件 | `/app/temp_storage` |
|  bind mount | 对话历史 SQLite | `/app/conversations.db` |

### 6. 生产部署建议

1. **使用环境变量文件**：不要将 `.env` 提交到 Git
2. **配置反向代理**：使用 Nginx/Caddy 配置 HTTPS
3. **设置强密码**：修改 `ADMIN_PASS`
4. **配置外部 API**：确保 `BASE_URL`、`LONGCAT_API_KEY` 等配置正确
5. **资源限制**：在 `docker-compose.yml` 中添加 `deploy.resources.limits`

### 7. 故障排查

```bash
# 检查容器日志
docker-compose logs api

# 进入容器调试
docker-compose exec api bash

# 检查网络连通性
docker-compose exec chat curl http://api:8002/docs
```
