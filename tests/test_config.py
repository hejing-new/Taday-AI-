"""配置模块测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ADMIN_USER, ADMIN_PASS, DB_FILE, CHROMA_DB_PATH, DATA_PATH,
    API_KEY, BASE_URL, CHAT_MODEL, JUDGE_MODEL, HEAL_MODEL, EMBED_MODEL,
    PORT_CHAT, PORT_ADMIN, PORT_API, PORT_ADMIN_API,
    ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE, ANALYTICS_PAGE_SIZE
)


def test_admin_credentials():
    """管理员凭据应该有值"""
    assert ADMIN_USER, "ADMIN_USER 不应为空"
    assert ADMIN_PASS, "ADMIN_PASS 不应为空"


def test_model_names():
    """模型名应该正确设置"""
    assert "Qwen" in CHAT_MODEL or "qwen" in CHAT_MODEL.lower()
    assert "Qwen" in JUDGE_MODEL or "qwen" in JUDGE_MODEL.lower()
    assert HEAL_MODEL, "HEAL_MODEL 不应为空"
    assert "bge" in EMBED_MODEL.lower(), "EMBED_MODEL 应为 BAAI/bge-m3"


def test_ports():
    """端口应该为正整数且不冲突"""
    ports = [PORT_CHAT, PORT_ADMIN, PORT_API, PORT_ADMIN_API]
    assert all(isinstance(p, int) and p > 0 for p in ports)
    assert len(set(ports)) == len(ports), "端口不能重复"


def test_paths():
    """路径常量应该非空"""
    assert DB_FILE
    assert CHROMA_DB_PATH
    assert DATA_PATH


def test_upload_limits():
    """上传限制应该合理"""
    assert "application/pdf" in ALLOWED_CONTENT_TYPES
    assert MAX_FILE_SIZE > 0
    assert MAX_FILE_SIZE <= 100 * 1024 * 1024  # 不超过 100MB


def test_analytics_page_size():
    """分页大小应该为正数"""
    assert ANALYTICS_PAGE_SIZE > 0


if __name__ == "__main__":
    tests = [
        test_admin_credentials,
        test_model_names,
        test_ports,
        test_paths,
        test_upload_limits,
        test_analytics_page_size,
    ]
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
        except Exception as e:
            print(f"⚠️ {t.__name__}: {e}")
