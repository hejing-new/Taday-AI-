"""
对话记忆持久化存储

使用 SQLite 存储会话历史，支持：
- 服务重启后恢复对话
- 浏览器刷新后恢复对话
- 多会话隔离（不同 session_id）
- 自动清理过期会话（默认保留 7 天）
"""
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# 数据库路径
DB_FILE = "conversations.db"

# 线程锁
_lock = threading.Lock()

# 保留天数
RETENTION_DAYS = 7


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（线程安全）"""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # 更好的并发支持
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db():
    """初始化数据库表"""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations (session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations (timestamp)")
            # 自动清理过期数据
            cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).isoformat()
            conn.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()


# 启动时初始化
_init_db()


def store_message(session_id: str, role: str, content: str):
    """存储一条消息"""
    if not session_id or not content:
        return
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content[:8000])  # 限制单条消息长度
            )
            conn.commit()
        finally:
            conn.close()


def store_exchange(session_id: str, user_msg: str, assistant_msg: str):
    """存储一轮对话（用户问题 + AI 回答）"""
    if not session_id or not user_msg:
        return
    with _lock:
        conn = _get_conn()
        try:
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, 'user', ?, ?)",
                (session_id, user_msg[:8000], now)
            )
            if assistant_msg:
                conn.execute(
                    "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, 'assistant', ?, ?)",
                    (session_id, assistant_msg[:8000], now)
                )
            conn.commit()
        finally:
            conn.close()


def get_history(session_id: str, limit: int = 50) -> List[Dict[str, str]]:
    """获取会话历史（按时间正序）

    Args:
        session_id: 会话 ID
        limit: 最多返回多少条消息

    Returns:
        [{"role": "user", "content": "..."}, ...]
    """
    if not session_id:
        return []
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT role, content FROM conversations "
            "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]
    finally:
        conn.close()


def get_history_as_messages(session_id: str, limit: int = 50) -> List[Dict[str, str]]:
    """获取会话历史（别名，与 get_history 相同）"""
    return get_history(session_id, limit)


def clear_history(session_id: str):
    """清空会话历史"""
    if not session_id:
        return
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()


def get_all_sessions() -> List[Dict[str, any]]:
    """获取所有会话列表（用于管理后台）

    Returns:
        [{"session_id": "...", "message_count": 10, "last_active": "2026-06-23T21:00:00"}, ...]
    """
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT session_id, COUNT(*) as msg_count, MAX(timestamp) as last_active "
            "FROM conversations "
            "GROUP BY session_id "
            "ORDER BY last_active DESC "
            "LIMIT 100"
        )
        rows = cursor.fetchall()
        return [
            {"session_id": r[0], "message_count": r[1], "last_active": r[2]}
            for r in rows
        ]
    finally:
        conn.close()


def get_session_stats() -> Dict[str, int]:
    """获取会话统计"""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT session_id), COUNT(*) FROM conversations"
        )
        row = cursor.fetchone()
        return {
            "total_sessions": row[0],
            "total_messages": row[1],
        }
    finally:
        conn.close()
