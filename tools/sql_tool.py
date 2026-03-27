import sqlite3
from langchain_core.tools import tool
# 这里引入你的大模型实例，请替换为你实际项目中初始化 llm 的方式
# 例如: from core.llm_config import llm (假设你有一个全局 llm)
# 这里为了演示，假设你已经有了一个名为 sql_llm 的对象

@tool
def query_financial_db(query: str) -> str:
    """
    当用户询问【历年财务数据对比】、【跨年份的营收/利润/毛利率数值】或需要【计算平均值、最高值】时，必须调用此工具。
    输入参数 query 应该是用户的原始自然语言问题。
    """
    print(f"\n[📊 Tool invoked] 数据分析师正在将自然语言转换为 SQL: '{query}'")
    
    # 1. 把数据库的 Schema（表结构）告诉大模型
    schema_info = """
    数据库包含一张表: catl_finance
    字段说明:
    - year (INTEGER): 年份，例如 2021, 2022, 2023, 2024
    - revenue_bn (REAL): 营业收入，单位是“亿元”
    - net_profit_bn (REAL): 净利润，单位是“亿元”
    - gross_margin (REAL): 毛利率，单位是百分比“%”
    """
    
    # 2. 构造极其严格的 Text-to-SQL Prompt
    prompt = f"""你是一个高级的底层 SQL 数据库工程师。请根据以下表结构，为用户的提问编写相应的 SQLite 查询语句。
    {schema_info}
    
    用户的提问是: {query}
    
    ⚠️ 纪律要求：
    1. 你只能输出合法的 SQL 语句，绝对不能包含任何其他解释性文字！
    2. 不要使用 markdown 代码块包裹（如 ```sql ），直接输出纯 SQL 文本！
    """
    
    try:
        # 这里请确保 sql_llm 是你项目中可用的 LangChain ChatModel 实例
        # 注意：你需要根据你的项目实际情况导入这个 llm
        from core.graph import llm  # 假设你可以从这里引入你的大模型
        
        # 3. 让大模型写出 SQL
        response = llm.invoke(prompt)
        sql_query = response.content.strip()
        
        # 简单清洗，防止小模型不听话加了 markdown 标记
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        print(f"  -> 🧠 生成的底层 SQL: {sql_query}")
        
        # 4. 在真实的数据库中执行这句 SQL
        conn = sqlite3.connect("finance_data.db")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # 获取表头字段名
        column_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not results:
            return f"执行 SQL: {sql_query} 后，未查到任何数据。可能是年份超出了数据库范围。"
            
        # 5. 把“冰冷的元组数据”包装成带表头的字符串返回给路由大脑
        formatted_result = f"✅ 数据库查询成功 (执行的SQL: {sql_query})\n"
        formatted_result += f"查询到的字段: {column_names}\n"
        formatted_result += f"具体数据结果: {results}\n"
        
        return formatted_result
        
    except Exception as e:
        return f"❌ 数据库查询失败，生成的 SQL 可能有语法错误: {str(e)}"