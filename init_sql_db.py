import sqlite3

def setup_financial_db():
    # 连接到本地 SQLite 数据库（如果不存在会自动创建）
    conn = sqlite3.connect("finance_data.db")
    cursor = conn.cursor()
    
    # 创建宁德时代历年财务核心指标表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS catl_finance (
            year INTEGER PRIMARY KEY,
            revenue_bn REAL,     -- 营业收入 (亿元)
            net_profit_bn REAL,  -- 净利润 (亿元)
            gross_margin REAL    -- 毛利率 (%)
        )
    ''')
    
    # 插入一些模拟的真实历史财务数据
    data = [
        (2021, 1303.56, 159.31, 26.28),
        (2022, 3285.94, 307.29, 20.25),
        (2023, 4009.17, 441.21, 22.91),
        (2024, 4500.00, 480.00, 24.50)
    ]
    
    # 忽略重复插入
    cursor.executemany('INSERT OR IGNORE INTO catl_finance VALUES (?, ?, ?, ?)', data)
    conn.commit()
    conn.close()
    print("✅ 结构化财务数据库 finance_data.db 初始化完成！")

if __name__ == "__main__":
    setup_financial_db()