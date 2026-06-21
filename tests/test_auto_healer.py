"""自动巡检模块测试"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置 UTF-8 输出，避免 Windows GBK 终端 emoji 报错
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from auto_healer import classify_error_type


def test_classify_dynamic():
    """包含'新闻''今天''股价'的提问应归类为 DYNAMIC"""
    assert classify_error_type("新能源最新新闻", "some response") == "DYNAMIC"
    assert classify_error_type("今天股价是多少", "some response") == "DYNAMIC"
    assert classify_error_type("当前股价", "some response") == "DYNAMIC"


def test_classify_static():
    """关于财报/战略的提问应归类为 STATIC"""
    assert classify_error_type("宁德时代2025年营收是多少？", "some response") == "STATIC"
    assert classify_error_type("动力电池毛利率", "some response") == "STATIC"


def test_empty_input():
    """空输入应默认归类为 STATIC"""
    assert classify_error_type("", "") == "STATIC"


if __name__ == "__main__":
    tests = [
        test_classify_dynamic,
        test_classify_static,
        test_empty_input,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR: {t.__name__}: {e}")
