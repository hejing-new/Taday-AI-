"""
共享 JSON 文件存储工具

统一管理 bad_cases_staging.json 和 dynamic_cases_archive.json 的读写，
避免多个路由模块重复实现相同逻辑。
"""
import os
import json
import threading
from typing import List, Dict, Any

# 文件路径
JSON_LOG_FILE = "bad_cases_staging.json"
DYNAMIC_JSON_FILE = "dynamic_cases_archive.json"

# 线程锁，防止并发写冲突
_lock = threading.Lock()


def ensure_json_files():
    """确保 JSON 文件存在，不存在则创建空文件"""
    for filepath in [JSON_LOG_FILE, DYNAMIC_JSON_FILE]:
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([], f)


def load_json(filepath: str) -> List[Dict[str, Any]]:
    """安全加载 JSON 文件，返回空列表如果文件不存在或损坏"""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def save_json(filepath: str, data: List[Dict[str, Any]]):
    """安全写入 JSON 文件（线程安全）"""
    with _lock:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def append_to_json(filepath: str, item: Dict[str, Any]):
    """追加一条记录到 JSON 文件（线程安全）"""
    with _lock:
        data = load_json(filepath)
        data.append(item)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_bad_cases() -> List[Dict[str, Any]]:
    """加载 Bad Cases"""
    ensure_json_files()
    return load_json(JSON_LOG_FILE)


def save_bad_cases(cases: List[Dict[str, Any]]):
    """保存 Bad Cases"""
    save_json(JSON_LOG_FILE, cases)


def load_dynamic_archive() -> List[Dict[str, Any]]:
    """加载动态归档"""
    ensure_json_files()
    return load_json(DYNAMIC_JSON_FILE)


def save_dynamic_archive(archive: List[Dict[str, Any]]):
    """保存动态归档"""
    save_json(DYNAMIC_JSON_FILE, archive)
