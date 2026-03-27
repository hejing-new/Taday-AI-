import os
# ================= 强行注入网络代理 =================
# 请确保你的梯子/代理软件处于打开状态，并核对端口号
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

import yfinance as yf
from langchain_core.tools import tool

@tool
def get_stock_price(ticker: str) -> str:
    """
    当用户询问某只股票的【实时价格】、【最新股价】或【当前行情】时，必须调用此工具。
    输入参数 ticker 必须是标准的雅虎财经股票代码。
    例如：
    - 宁德时代 (A股)："300750.SZ"
    - 贵州茅台 (A股)："600519.SS"
    - 特斯拉 (美股)："TSLA"
    - 苹果 (美股)："AAPL"
    """
    print(f"\n[📈 Tool invoked] 行情交易员正在查询股票 {ticker} 的实时价格...")
    try:
        # 获取股票对象
        stock = yf.Ticker(ticker)
        
        # 获取最新行情数据 (只取当天的)
        todays_data = stock.history(period='1d')
        
        if todays_data.empty:
            return f"未能找到代码为 {ticker} 的股票数据，请检查代码是否正确。"
        
        # 提取收盘价 (如果是交易时间，就是最新价)
        current_price = todays_data['Close'].iloc[0]
        
        # 获取货币单位
        currency = stock.info.get('currency', '未知货币')
        
        return f"股票 {ticker} 的最新价格是：{current_price:.2f} {currency}。"
        
    except Exception as e:
        return f"查询股票 {ticker} 时发生网络或解析错误: {str(e)}"


# ================= 本地模块测试 =================
if __name__ == "__main__":
    print("\n--- 行情交易员工具独立测试 ---")
    
    # 测试 1: 查询宁德时代 (A股深圳)
    print("\n测试 1：查询宁德时代 (300750.SZ)")
    result_catl = get_stock_price.invoke({"ticker": "300750.SZ"})
    print(result_catl)
    
    # 测试 2: 查询特斯拉 (美股)
    print("\n测试 2：查询特斯拉 (TSLA)")
    result_tsla = get_stock_price.invoke({"ticker": "TSLA"})
    print(result_tsla)