from langchain_core.tools import tool
from duckduckgo_search import DDGS

# ==========================================
# 🌐 宏观与舆情雷达：全网搜索工具
# ==========================================

@tool
def web_search_tool(query: str) -> str:
    """
    当用户询问：
    1. 最新新闻、突发事件（如“今天宁德时代为什么大跌？”）
    2. 实时宏观经济政策、行业补贴（如“最近出台了什么新能源政策？”）
    3. 其他在企业财报中无法找到的最新全网信息时...
    必须调用此工具。
    输入参数 query 应该是精炼的搜索引擎查询词，例如 "宁德时代 最新 新闻 关税"。
    """
    print(f"\n[🌐 Tool invoked] 联网雷达正在扫描全网: '{query}'")
    
    try:
        results = []
        # 使用 DDGS 发起搜索，限制 max_results=3 以防止撑爆大模型的上下文窗口
        with DDGS() as ddgs:
            # region='cn-zh' 可以优先搜索中文资讯
            # 可选参数：timelimit: 'd' (一天), 'w' (一周), 'm' (一个月), 'y' (一年)
            search_generator = ddgs.text(query, region='cn-zh', safesearch='moderate', max_results=3, timelimit='m')
            
            for r in search_generator:
                # 将抓取到的标题、摘要和链接拼接成大模型易于阅读的格式
                result_str = (
                    f"【标题】: {r.get('title', '无标题')}\n"
                    f"【摘要】: {r.get('body', '无摘要')}\n"
                    f"【来源】: {r.get('href', '无链接')}\n"
                )
                results.append(result_str)

        if not results:
            return "未找到相关的网络搜索结果，请尝试更换关键词。"

        # 组合最终的返回字符串
        final_output = "以下是为你抓取到的全网最新资讯：\n\n" + "\n---\n".join(results)
        return final_output

    except Exception as e:
        # 企业级容错：搜索失败时不崩溃，而是告诉大模型搜索受阻
        return f"联网抓取数据时发生网络异常: {str(e)}"

# ==========================================
# 🧪 本地独立测试
# ==========================================
if __name__ == "__main__":
    print("\n--- 联网雷达工具独立测试 ---")
    test_query = "宁德时代 固态电池 最新进展 2026"
    
    # 模拟大模型传入参数
    result = web_search_tool.invoke({"query": test_query})
    
    print("\n📝 最终抓取到的外网数据：\n")
    print(result)