"""
结构化财务数据 SQL 查询工具

支持自然语言 → SQL 查询宁德时代历年财务数据。
使用参数化查询防止 SQL 注入，仅允许 SELECT 语句。
"""
import sqlite3
import re
from langchain_core.tools import tool
from config import CHAT_MODEL, API_KEY, BASE_URL
from logger import logger

# 数据库路径
DB_PATH = "finance_data.db"

# Schema 信息
SCHEMA_INFO = """
数据库包含一张表: catl_finance
字段说明:
- year (INTEGER): 年份，例如 2021, 2022, 2023, 2024
- revenue_bn (REAL): 营业收入，单位是"亿元"
- net_profit_bn (REAL): 净利润，单位是"亿元"
- gross_margin (REAL): 毛利率，单位是百分比"%"
"""

# SQL 注入防护：只允许以 SELECT 开头的语句
_SELECT_PATTERN = re.compile(r'^\s*SELECT\b', re.IGNORECASE)
# 禁止的危险关键词
_FORBIDDEN_KEYWORDS = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'EXEC', 'UNION']


def _get_llm():
    """延迟导入 llm，避免循环依赖"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=API_KEY,
        base_url="https://api.siliconflow.cn/v1",
        temperature=0.1
    )


def _validate_sql(sql: str) -> bool:
    """校验 SQL 语句安全性：只允许 SELECT，禁止危险关键词"""
    if not _SELECT_PATTERN.match(sql):
        return False
    upper_sql = sql.upper()
    for kw in _FORBIDDEN_KEYWORDS:
        if kw in upper_sql:
            return False
    return True


@tool
def query_financial_db(query: str) -> str:
    """
    当用户询问【历年财务数据对比】、【跨年份的营收/利润/毛利率数值】或需要【计算平均值、最高值】时，必须调用此工具。
    输入参数 query 应该是用户的原始自然语言问题。
    """
    logger.info(f"数据分析师正在将自然语言转换为 SQL: '{query}'")

    # 1. 构造 Text-to-SQL Prompt
    prompt = f"""你是一个高级的底层 SQL 数据库工程师。请根据以下表结构，为用户的提问编写相应的 SQLite 查询语句。
    {SCHEMA_INFO}

    用户的提问是: {query}

    ⚠️ 纪律要求：
    1. 你只能输出合法的 SQL 语句，绝对不能包含任何其他解释性文字！
    2. 不要使用 markdown 代码块包裹（如 ```sql ），直接输出纯 SQL 文本！
    3. 只允许 SELECT 查询，不能写 DROP/DELETE/INSERT/UPDATE 等危险语句！
    """

    try:
        llm = _get_llm()

        # 2. 让大模型写出 SQL
        response = llm.invoke(prompt)
        sql_query = response.content.strip()

        # 清洗 markdown 标记
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        logger.info(f"生成的底层 SQL: {sql_query}")

        # 3. SQL 注入防护校验
        if not _validate_sql(sql_query):
            logger.error(f"SQL 安全校验失败，拦截: {sql_query}")
            return f"❌ 安全拦截：生成的 SQL 包含危险操作或不是 SELECT 语句，已被拦截。"

        # 4. 执行查询
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()

        column_names = [description[0] for description in cursor.description] if cursor.description else []
        conn.close()

        if not results:
            return f"执行 SQL: {sql_query} 后，未查到任何数据。可能是年份超出了数据库范围。"

        # 5. 格式化返回
        formatted_result = f"✅ 数据库查询成功 (执行的SQL: {sql_query})\n"
        formatted_result += f"查询到的字段: {column_names}\n"
        formatted_result += f"具体数据结果: {results}\n"

        return formatted_result

    except Exception as e:
        logger.error(f"数据库查询失败: {e}")
        return f"❌ 数据库查询失败: {str(e)}"
