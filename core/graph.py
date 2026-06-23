import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

from tools.rag_tool import analyze_catl_report
from tools.price_tool import get_stock_price
from tools.web_search_tool import web_search_tool
from tools.sql_tool import query_financial_db

from config import API_KEY, BASE_URL, CHAT_MODEL

llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1
)

tools = [get_stock_price, analyze_catl_report, web_search_tool, query_financial_db]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str = "default"


def call_model(state: AgentState):
    print("\n[Brain] Thinking...")
    messages = state["messages"]

    if not isinstance(messages[0], SystemMessage):
        current_date = datetime.now().strftime("%Y-%m-%d")

        system_msg = SystemMessage(
            content=f"""You are a senior financial analyst named "Taday Brain".
Current date: {current_date}

Tool rules:
1. Stock price: call get_stock_price
2. Financial report details for CATL: call analyze_catl_report
3. Latest news/policy: call web_search_tool
4. Multi-year financial data comparison: call query_financial_db

Output rules:
- Reply in Simplified Chinese only
- Only use tool results, never fabricate data
- Valid JSON for tool calls only"""
        )
        messages = [system_msg] + messages

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)

            # Detect corrupted text-based tool call and retry
            if hasattr(response, 'content') and response.content:
                content = response.content.strip()
                is_corrupt_tool_call = (
                    content.startswith('{') and
                    '"function"' in content and
                    ('analyze_catl' in content or 'catl_report' in content)
                )
                if is_corrupt_tool_call:
                    print(f"\n[WARN] Format hallucination (attempt {attempt+1}), retrying...")
                    if attempt == max_retries - 1:
                        # Fallback: try to extract query and call tool directly
                        query = None
                        try:
                            parsed = json.loads(content)
                            args = parsed.get("arguments", {})
                            query = args.get("query") or args.get("ticker")
                        except Exception:
                            pass
                        if not query:
                            # Last resort: extract user's original question
                            for msg in reversed(messages):
                                if isinstance(msg, HumanMessage):
                                    query = msg.content
                                    break
                        if query:
                            print(f"\n[FALLBACK] Calling tool directly with: {query}")
                            try:
                                from tools.rag_tool import get_query_engine
                                engine = get_query_engine()
                                result = engine.query(query)
                                result_str = str(result)
                                if len(result_str) > 50 and not result_str.count('{') > 5:
                                    return {"messages": [AIMessage(content=f"Knowledge base result:\n{result_str}")]}
                            except Exception as e:
                                print(f"\n[FALLBACK] Tool call failed: {e}")
                        fallback = "Sorry, tool scheduling format error. Please try rephrasing."
                        return {"messages": [AIMessage(content=fallback)]}
                    continue

            return {"messages": [response]}
        except Exception as e:
            print(f"\n[WARN] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                fallback_msg = "Sorry, an error occurred. Please try again."
                return {"messages": [AIMessage(content=fallback_msg)]}


tool_node = ToolNode(tools)


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_names = [t['name'] for t in last_message.tool_calls]
        print(f"\n[Router] Dispatching to: {tool_names}")
        return "tools"

    print("\n[Router] Final answer ready.")
    return "__end__"


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "agent")

memory_saver = MemorySaver()
app_graph = workflow.compile(checkpointer=memory_saver)

if __name__ == "__main__":
    print("Taday Brain starting...")
    user_input = "Check CATL stock price today and analyze reasons."
    initial_state = {"messages": [HumanMessage(content=user_input)], "thread_id": "test"}
    config = {"configurable": {"thread_id": "test"}}
    result = app_graph.invoke(initial_state, config=config)
    print("\n" + "=" * 50)
    print(result["messages"][-1].content)
