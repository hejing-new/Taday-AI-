"""SQL 工具安全测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.sql_tool import _validate_sql


def test_select_allowed():
    """正常的 SELECT 应该通过"""
    assert _validate_sql("SELECT * FROM catl_finance")
    assert _validate_sql("SELECT year, revenue_bn FROM catl_finance WHERE year > 2022")
    assert _validate_sql("  SELECT COUNT(*) FROM catl_finance")


def test_drop_blocked():
    """DROP 应该被拦截"""
    assert not _validate_sql("DROP TABLE catl_finance")


def test_delete_blocked():
    """DELETE 应该被拦截"""
    assert not _validate_sql("DELETE FROM catl_finance WHERE year = 2024")


def test_insert_blocked():
    """INSERT 应该被拦截"""
    assert not _validate_sql("INSERT INTO catl_finance VALUES (2025, 5000, 500, 25)")


def test_update_blocked():
    """UPDATE 应该被拦截"""
    assert not _validate_sql("UPDATE catl_finance SET revenue_bn = 5000 WHERE year = 2024")


def test_union_blocked():
    """UNION 注入应该被拦截"""
    assert not _validate_sql("SELECT * FROM catl_finance UNION SELECT * FROM users")


def test_alter_blocked():
    """ALTER 应该被拦截"""
    assert not _validate_sql("ALTER TABLE catl_finance ADD COLUMN test TEXT")


def test_non_sql_blocked():
    """非 SQL 语句应该被拦截"""
    assert not _validate_sql("hello world")
    assert not _validate_sql("")
    assert not _validate_sql("12345")


if __name__ == "__main__":
    tests = [
        test_select_allowed,
        test_drop_blocked,
        test_delete_blocked,
        test_insert_blocked,
        test_update_blocked,
        test_union_blocked,
        test_alter_blocked,
        test_non_sql_blocked,
    ]
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
        except Exception as e:
            print(f"⚠️ {t.__name__}: {e}")
