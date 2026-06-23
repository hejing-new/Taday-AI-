"""
financial_chunker 边界测试
验证 FinancialReportChunker 在各种输入下的鲁棒性。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONUTF8'] = '1'

from tools.financial_chunker import (
    chunk_financial_report,
    _is_section_header,
    _is_table_line,
    _estimate_tokens,
)
from tools.rag_tool import _tokenize as chunker_tokenize


class TestSectionDetection:
    """测试章节标题识别"""

    def test_chinese_brackets(self):
        assert _is_section_header('（一）公司基本情况') is True

    def test_chinese_dot(self):
        assert _is_section_header('一、公司基本情况') is True

    def test_arabic_numeral(self):
        assert _is_section_header('3.1 财务数据') is True

    def test_regular_text(self):
        assert _is_section_header('这是一段普通文本') is False

    def test_empty_string(self):
        assert _is_section_header('') is False


class TestTableDetection:
    """测试表格行检测"""

    def test_pure_numeric_line(self):
        assert _is_table_line('5,978,096') is True

    def test_separator_line(self):
        """含表格分隔符的行应识别为表格行"""
        assert _is_table_line('| 营业收入 | 3165.06 |') is True
        assert _is_table_line('│\t净利润\t│\t500.12\t│') is True

    def test_indicator_with_number(self):
        assert _is_table_line('营业收入 3165.06') is True

    def test_regular_text(self):
        assert _is_table_line('这是一段普通文本描述') is False


class TestTokenEstimation:
    """测试 token 估算"""

    def test_pure_chinese(self):
        tokens = _estimate_tokens('宁德时代')
        assert tokens > 0, "纯中文应产生正 token 数"

    def test_pure_english(self):
        tokens = _estimate_tokens('Hello World')
        assert tokens > 0, "纯英文应产生正 token 数"

    def test_mixed(self):
        tokens = _estimate_tokens('宁德时代 CATL 2025')
        assert tokens > 0, "混合文本应产生正 token 数"

    def test_empty(self):
        assert _estimate_tokens('') == 0, "空文本应为 0"


class TestChunking:
    """测试分块器端到端"""

    def test_empty_input(self):
        """空输入不应崩溃"""
        chunks = chunk_financial_report([])
        assert isinstance(chunks, list)

    def test_single_page(self):
        """单页输入应产生至少一个 chunk（降低 min_chunk_tokens 以适应短文本）"""
        long_text = '宁德时代2025年营业收入3165亿元，同比增长15.2%，净利润500.12亿元，同比增长18.5%。' * 5
        pages = [(1, long_text)]
        chunks = chunk_financial_report(pages, min_chunk_tokens=10)
        assert len(chunks) >= 1, "单页应产生至少一个 chunk"

    def test_table_preservation(self):
        """表格应被保留为独立 chunk"""
        pages = [
            (1, '财务数据\n营业收入 3165.06\n净利润 500.12\n营业成本 2000.50\n毛利率 18.5%'),
        ]
        chunks = chunk_financial_report(pages, min_chunk_tokens=10)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) >= 1, "应检测到表格 chunk"

    def test_section_cutting(self):
        """章节标题应触发切割"""
        pages = [
            (1, '（一）公司基本情况\n宁德时代成立于2011年，总部位于福建省宁德市，是一家专注于新能源汽车动力电池的研发、生产和销售的高新技术企业。公司产品广泛应用于新能源商用车、乘用车、储能等领域。\n（二）财务数据\n营业收入3165亿元，净利润500.12亿元'),
        ]
        chunks = chunk_financial_report(pages, min_chunk_tokens=10)
        # 至少应产生 2 个 chunk（两个章节）
        assert len(chunks) >= 2, "章节标题应触发分段"


# 运行测试
if __name__ == '__main__':
    import traceback
    test_classes = [TestSectionDetection, TestTableDetection, TestTokenEstimation, TestChunking]
    passed = 0
    failed = 0
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f'  [PASS] {cls.__name__}.{method_name}')
                    passed += 1
                except Exception as e:
                    print(f'  [FAIL] {cls.__name__}.{method_name}: {e}')
                    traceback.print_exc()
                    failed += 1
    print(f'\n{"="*50}')
    print(f'Result: {passed} passed, {failed} failed')
