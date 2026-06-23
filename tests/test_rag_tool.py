"""
BM25 / RRF / 分词 单元测试
验证 rag_tool.py 中核心计算逻辑的正确性。
"""
import sys
import os
import math
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONUTF8'] = '1'

from tools.rag_tool import _tokenize, _compute_bm25, _compute_numeric_boost, _expand_query


class TestTokenize:
    """测试 _tokenize 分词器"""

    def test_chinese_tokens(self):
        tokens = _tokenize('宁德时代2025年营业收入')
        assert len(tokens) > 0, "应产生至少一个 token"
        # 应包含中文词组
        assert any('宁德' in t or '时代' in t for t in tokens), "应识别中文词组"

    def test_english_tokens(self):
        tokens = _tokenize('Apple Inc revenue 2024')
        assert any(t == 'apple' for t in tokens), "应提取英文单词并转小写"
        assert any(t == 'revenue' for t in tokens), "应提取 revenue"

    def test_numeric_tokens(self):
        tokens = _tokenize('营收3165.06亿元')
        assert any('3165.06' in t for t in tokens), "应提取数字+单位"

    def test_percentage_tokens(self):
        tokens = _tokenize('毛利率18.5%')
        assert any('18.5%' in t for t in tokens), "应提取百分比"

    def test_empty_input(self):
        assert _tokenize('') == [], "空输入应返回空列表"
        assert _tokenize('   ') == [], "空白输入应返回空列表"


class TestBM25:
    """测试 _compute_bm25 打分"""

    def test_identical_documents(self):
        """查询与文档完全相同时，BM25 分数应 > 0"""
        tokens = _tokenize('宁德时代营业收入')
        score = _compute_bm25(tokens, tokens, 200, 100)
        assert score > 0, "相同文档的 BM25 分数应 > 0"

    def test_unrelated_documents(self):
        """查询与文档无关时，BM25 分数应为 0"""
        q_tokens = _tokenize('宁德时代营业收入')
        d_tokens = _tokenize('今天天气很好')
        score = _compute_bm25(q_tokens, d_tokens, 200, 100)
        assert score == 0, "无关文档的 BM25 分数应为 0"

    def test_empty_query(self):
        """空查询应返回 0"""
        d_tokens = _tokenize('宁德时代营业收入')
        score = _compute_bm25([], d_tokens, 200, 100)
        assert score == 0, "空查询应返回 0"

    def test_empty_document(self):
        """空文档应返回 0"""
        q_tokens = _tokenize('宁德时代营业收入')
        score = _compute_bm25(q_tokens, [], 200, 100)
        assert score == 0, "空文档应返回 0"

    def test_more_matching_terms_scores_higher(self):
        """同一查询下，包含更多匹配词项的文档分数应更高"""
        # 使用英文单词确保分词结果可预测（中文正则贪婪匹配可能导致词边界不对齐）
        q_tokens = _tokenize('apple revenue profit')
        short_doc = _tokenize('apple revenue')  # 匹配 2 个词
        long_doc = _tokenize('apple revenue profit income')  # 匹配 3 个词
        short_score = _compute_bm25(q_tokens, short_doc, 200, 100)
        long_score = _compute_bm25(q_tokens, long_doc, 200, 100)
        assert long_score > short_score, "匹配更多查询词的文档分数应更高"

    def test_doc_freq_affects_score(self):
        """语料库级 DF 应影响分数：高频词 IDF 低，低频词 IDF 高"""
        # 使用英文单词确保分词结果可预测
        q_tokens = _tokenize('revenue')
        d_tokens = _tokenize('revenue growth')
        # DF=1（低频）应比 DF=50（高频）得分更高
        df_low = Counter({'revenue': 1})
        df_high = Counter({'revenue': 50})
        score_low_df = _compute_bm25(q_tokens, d_tokens, 200, 100, df_low)
        score_high_df = _compute_bm25(q_tokens, d_tokens, 200, 100, df_high)
        assert score_low_df > score_high_df, "低频词应产生更高的 IDF 分数"


class TestNumericBoost:
    """测试 _compute_numeric_boost"""

    def test_matching_numbers(self):
        """文档中包含查询中的数字，应获得 boost"""
        score = _compute_numeric_boost('宁德时代2025年营收', '宁德时代2025年营收3165亿元')
        assert score > 0, "匹配数字应产生正分"

    def test_no_matching_numbers(self):
        """文档中不包含查询中的数字，boost 应为 0"""
        score = _compute_numeric_boost('宁德时代2025年营收', '宁德时代2024年营收3000亿元')
        assert score == 0, "不匹配数字的 boost 应为 0"

    def test_query_without_numbers(self):
        """查询中没有数字，boost 应为 0"""
        score = _compute_numeric_boost('宁德时代营收', '宁德时代营收3165亿元')
        assert score == 0, "查询无数字则 boost 为 0"


class TestQueryExpansion:
    """测试 _expand_query"""

    def test_returns_original_query(self):
        """扩展结果应始终包含原始查询"""
        results = _expand_query('宁德时代2025年营业收入')
        assert '宁德时代2025年营业收入' in results, "应包含原始查询"

    def test_synonym_expansion(self):
        """同义词替换应产生扩展查询"""
        results = _expand_query('宁德时代营收')
        # 营收 → 营业收入 / 销售收入 等
        assert len(results) > 1, "应产生至少一个扩展查询"

    def test_no_expansion_for_generic_query(self):
        """不含金融术语的查询扩展数量应有限"""
        results = _expand_query('你好世界')
        assert len(results) <= 2, "通用查询扩展应有限"


class TestSQLValidation:
    """测试 SQL 注入防护"""

    def test_valid_select(self):
        from tools.sql_tool import _validate_sql
        assert _validate_sql('SELECT * FROM catl_finance') is True

    def test_reject_drop(self):
        from tools.sql_tool import _validate_sql
        assert _validate_sql('DROP TABLE catl_finance') is False

    def test_reject_multi_statement(self):
        from tools.sql_tool import _validate_sql
        assert _validate_sql('SELECT * FROM catl_finance; DROP TABLE catl_finance') is False

    def test_reject_union(self):
        from tools.sql_tool import _validate_sql
        assert _validate_sql('SELECT * FROM catl_finance UNION SELECT * FROM sqlite_master') is False


# 运行测试
if __name__ == '__main__':
    import traceback
    test_classes = [TestTokenize, TestBM25, TestNumericBoost, TestQueryExpansion, TestSQLValidation]
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
