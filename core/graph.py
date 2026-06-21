import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from datetime import datetime # 确保文件顶部有这个引

# 导入我们千辛万苦打磨好的三个“得力干将”
from tools.rag_tool import analyze_catl_report
from tools.price_tool import get_stock_price
from tools.web_search_tool import web_search_tool

# ================= 1. 初始化环境与大模型 =================
from config import API_KEY, BASE_URL, CHAT_MODEL

# 使用 LangChain 的标准大模型接口连接硅基流动
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1 # 金融场景，温度设低一点，保证严谨性
)

# 🚀 四个工具打包，赋予大模型超能力
tools = [get_stock_price, analyze_catl_report, web_search_tool, query_financial_db]
llm_with_tools = llm.bind_tools(tools)

# ================= 2. 定义状态 (State) =================
# Agent 的记忆库，保存着所有的聊天记录
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ================= 3. 定义节点 (Nodes) =================
# 节点 A：大脑思考节点
def call_model(state: AgentState):
    print("\n[🧠 Brain] 大模型正在思考策略与分配任务...")
    messages = state["messages"]
    
    if not isinstance(messages[0], SystemMessage):
        # 获取当前真实的系统日期
        current_date = datetime.now().strftime("%Y年%m月%d日")
        
        system_msg = SystemMessage(
            content=f"""你是一位顶级的首席金融分析师。你的名字叫“Taday 金融大脑”。
            【当前系统时间】：{current_date}
            
            【工具使用守则】：
            1. 股票行情：询问当前【实时价格、收盘价、涨跌幅】，调用 `get_stock_price`。
            2. 深度财报：询问宁德时代【具体的业务板块营收数据】、【详细财报文字细节、产能规划、战略方向等内部信息】，必须调用 `analyze_catl_report`。
            3. 全网搜索：询问【最新新闻、突发事件、宏观政策】时，调用 `web_search_tool`。
            4. 历年数据：询问【跨年份营收/利润/毛利率对比】、【多年数据趋势】时，调用 `query_financial_db`。
            
            ⚠️【搜索关键词纪律】⚠️
            当调用 `web_search_tool` 时，请像人类使用百度一样精炼关键词：
            1. 动态判断主体：如果用户的问题明确包含或暗示了某家公司（如宁德时代），必须带上公司全称；如果用户问的是宏观行业（如新能源政策），则只搜行业关键词，绝不强塞公司名！
            2. 动态时间定语：如果询问最新动态，请务必在搜索词中加入“{current_date}”以保证时效性。
            3. 搜索词必须极简！例如用户问“最近出台了什么新能源政策”，你的 query 应该是："新能源政策 最新 2026年" ，而不是一整句长对话。
            
            ⚠️【输出纪律与规范】⚠️
            1. 强制中文：必须使用流畅的中文（简体）进行回复！
            2. 拒绝幻觉：只根据工具返回的内容作答，如果没有搜到，直接说“未找到相关信息”。绝对不能自己捏造数据。
            3. 格式严谨：调用工具时参数必须是合法 JSON，全英文半角双引号。
            4. 灵活变通：如果调用某个工具没有查到数据，思考是否应该换另一个工具重试（例如查新闻没查到具体营收，请去调用 analyze_catl_report 查财报）。
            """
        )
        messages = [system_msg] + messages
        
    # 🚀 核心修复：为 7B 小模型加上“防弹衣”（异常重试机制）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"\n[⚠️ 容错拦截] 第 {attempt+1} 次尝试时，拦截到小模型的 JSON 格式幻觉。强制大脑重试...")
            if attempt == max_retries - 1:
                # 兜底：如果重试 3 次还是疯言疯语，就优雅地返回文本，绝不让程序报错红屏
                from langchain_core.messages import AIMessage
                fallback_msg = "对不起，我在同时调用多个外部工具汇总庞大数据时，底层的格式处理出现了小故障。请您尝试将问题拆分开来，例如：先问股价，然后再问新闻原因。"
                return {"messages": [AIMessage(content=fallback_msg)]}

# 节点 B：工具执行节点 (LangGraph 自带了现成的封装，直接传入 tools 列表即可)
tool_node = ToolNode(tools)

# ================= 4. 定义路由条件 (Edges) =================
# 决定到底是去调用工具，还是直接把话回给用户
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果大模型的回复里包含 tool_calls，说明它想用工具
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_names = [t['name'] for t in last_message.tool_calls]
        print(f"\n[🔀 Router] 路由中枢决定派发任务给: {tool_names}")
        return "tools"
    
    # 否则，说明它思考完毕，直接结束流程
    print("\n[🔀 Router] 大脑已得出最终结论，准备输出。")
    return "__end__"

# ================= 5. 编排工作流并编译成图 =================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 画线连接
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "agent") # 工具执行完，必须把结果还给大脑重新思考

# 编译生成最终的智能体
app_graph = workflow.compile()

# ================= 6. 终端终极连调测试 =================
if __name__ == "__main__":
    print("🚀 启动 Taday 金融大脑 (LangGraph Router) 测试...")
    
    # 模拟那个最刁钻的混合问题
    user_input = "查一下宁德时代今天的股价，并结合全网最新新闻，分析一下今天股价波动的原因。"
    print(f"\n👤 客户提问: {user_input}")
    print("-" * 50)

    # 构造初始记忆库
    initial_state = {"messages": [HumanMessage(content=user_input)]}

    # 让 LangGraph 开始运转！(这里我们不用 stream，直接 invoke 看最终结果)
    result = app_graph.invoke(initial_state)

    print("\n" + "=" * 50)
    print("🏁 最终生成的研报级回答：\n")
    # 提取最后一条消息（也就是大模型汇总所有工具结果后生成的最终回答）
    print(result["messages"][-1].content)