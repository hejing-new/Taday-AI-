# ==========================================
# 🚀 终极 UTF-8 防护罩 MAX 版 (务必放在第一行)
# ==========================================
import sys
import os
import io

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==========================================
# 🌐 宏观与舆情雷达：Tavily 企业级全网搜索工具
# ==========================================
from langchain_core.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
from logger import logger

# 加载环境变量
load_dotenv()

@tool
def web_search_tool(query: str) -> str:
    """
    当用户询问最新新闻、突发事件、宏观政策等实时全网信息时调用。
    """
    logger.info(f"Tavily Search: 搜索词='{query}'")
    
    try:
        # 每次调用动态获取 Key，防呆设计
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "系统未配置 Tavily API Key，联网搜索暂时不可用。请在 .env 中配置 TAVILY_API_KEY。"

        # 初始化 Tavily 客户端
        client = TavilyClient(api_key=api_key)

        # 执行深度搜索
        # search_depth="advanced" 会深入网页内部抓取高质量正文，专为大模型设计
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            include_answer=True  # 保留 Tavily 摘要，为 LLM 提供更多上下文
        )

        results = response.get("results", [])

        if not results:
            return f"Tavily 搜索引擎未能找到关于 '{query}' 的结果。这可能是因为搜索词太偏。"

        import re

        formatted_results = []
        for i, r in enumerate(results):
            raw_title = r.get('title', '无标题')
            url = r.get('url', '#')
            content = r.get('content', '无摘要')
            
            # 🚀 1. 洗标题：去掉开头的数字、标点和乱码
            # 改用更强力的正则：去掉开头所有非字母数字的东西，直到遇到第一个真正的文字
            clean_title = re.sub(r'^[^\u4e00-\u9fa5a-zA-Z]+', '', raw_title)
            
            # 🚀 2. 洗内容：把 Tavily 抓回来的 ## 或 ### 全部替换成空格
            # 这样就能彻底解决“字大”的问题，让它变回普通正文
            clean_content = re.sub(r'#+\s?', '', content)
            # 顺便把内容里的多余换行也洗了，让排版更紧凑
            clean_content = clean_content.replace('\n', ' ').strip()
            
            # 🚀 3. 重新组装：使用粗体代替标题，链接放在最后
            # 取前1000字，为 LLM 提供足够上下文
            result_str = (
                f"**🔗 来源 {i+1}: [{clean_title}]({url})**\n\n"
                f"{clean_content[:1000]}\n"
                f"[阅读全文]({url})\n"
            )
            formatted_results.append(result_str)

        # 最终输出，顶部标题也稍微控制一下字号
        final_output = "#### 🌐 Tavily 实时情报库\n\n" + "\n---\n".join(formatted_results)

        # 如果有 Tavily 摘要答案，附加在末尾
        tavily_answer = response.get("answer", "")
        if tavily_answer:
            final_output += f"\n\n**📋 Tavily 摘要:** {tavily_answer}"

        return final_output

    except Exception as e:
        # 企业级容错
        return f"调用 Tavily API 抓取数据时发生异常: {str(e)}"

# ==========================================
# 🧪 本地独立测试
# ==========================================
if __name__ == "__main__":
    print("\n--- Tavily 联网雷达工具独立测试 ---")
    
    # 模拟大模型生成的干练搜索词
    test_query = "新能源政策 最新"
    
    result = web_search_tool.invoke({"query": test_query})
    
    print("\n📝 最终抓取到的外网数据：\n")
    print(result)