"""
ReAct Engine for Taday Financial Brain

ReAct pattern: Reason -> Act -> Observe loop
Model outputs text instructions, engine parses and executes tool calls.
"""
import json
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import CHAT_MODEL, LONGCAT_API_KEY, LONGCAT_BASE_URL
from logger import logger

# Tool registry: maps tool name -> (function, required_args)
from tools.rag_tool import get_query_engine
from tools.price_tool import get_stock_price
from tools.web_search_tool import web_search_tool
from tools.sql_tool import query_financial_db

# LLM 单例：全局复用，避免每次调用都创建新客户端
_llm_instance = None


def _get_llm():
    """获取全局 LLM 单例，首次调用时创建"""
    global _llm_instance
    if _llm_instance is None:
        from langchain_openai import ChatOpenAI
        _llm_instance = ChatOpenAI(
            model_name=CHAT_MODEL,
            temperature=0.1,
            openai_api_key=LONGCAT_API_KEY,
            openai_api_base=LONGCAT_BASE_URL,
            max_retries=2,
        )
    return _llm_instance

def _rag_query(query, **_):
    """适配混合引擎和标准引擎的查询"""
    engine = get_query_engine()
    return engine.query(query)

TOOL_REGISTRY = {
    "analyze_catl_report": {
        "fn": _rag_query,
        "args": ["query"],
        "description": "Search financial reports for revenue, profit, margins, strategy, etc."
    },
    "get_stock_price": {
        "fn": lambda ticker, **_: get_stock_price.invoke({"ticker": ticker}),
        "args": ["ticker"],
        "description": "Get real-time stock price (e.g., ticker=300750.SZ)"
    },
    "web_search_tool": {
        "fn": lambda query, **_: web_search_tool.invoke({"query": query}),
        "args": ["query"],
        "description": "Search latest news, policy, market events"
    },
    "query_financial_db": {
        "fn": lambda query, **_: query_financial_db.invoke({"query": query}),
        "args": ["query"],
        "description": "SQL query for multi-year financial data comparison"
    },
}

SYSTEM_PROMPT = """You are Taday Brain, a senior financial analyst.

You have access to these tools:
{tool_descriptions}

CRITICAL INSTRUCTIONS:
1. When you need data, output EXACTLY one tool call per response using this format:
   TOOL: <tool_name>
   ARGS: <json_args>

2. After receiving tool results, analyze them and either:
   - Call another tool if more data is needed
   - Provide a final answer with ALL the data

3. For final answers, output:
   FINAL: <your complete answer in Simplified Chinese>

4. NEVER fabricate data. Only use tool results.
5. NEVER output tool calls as plain text examples - always use the TOOL/ARGS format.
6. ARGS must be valid JSON with double quotes."""


def _build_tool_descriptions():
    descs = []
    for name, info in TOOL_REGISTRY.items():
        args_str = ", ".join(info["args"])
        descs.append(f"- {name}({args_str}): {info['description']}")
    return "\n".join(descs)


def _parse_tool_call(text: str):
    """Parse TOOL/ARGS format from model output."""
    text = text.strip()
    tool_match = re.search(r'TOOL:\s*(\w+)', text)
    args_match = re.search(r'ARGS:\s*(\{[^}]+\})', text, re.DOTALL)

    if not tool_match:
        return None

    tool_name = tool_match.group(1)
    if tool_name not in TOOL_REGISTRY:
        return None

    if args_match:
        try:
            args = json.loads(args_match.group(1))
            return tool_name, args
        except json.JSONDecodeError:
            pass

    # Try to extract args from surrounding text
    return tool_name, {}


def _execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool and return the result."""
    tool_info = TOOL_REGISTRY[tool_name]
    required = tool_info["args"]

    # Fill missing args with empty strings
    for arg in required:
        if arg not in args:
            args[arg] = ""

    try:
        result = tool_info["fn"](**args)
        return str(result)
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e}")
        return f"[Tool Error: {e}]"


def run_react(user_question: str, thread_id: str = "default", history_messages: list = None):
    """
    Run the ReAct loop.

    Args:
        user_question: The user's question
        thread_id: Session ID for memory
        history_messages: Previous messages as list of dicts with 'role' and 'content'

    Yields:
        tuples of (answer_text, status, source_cards)
    """
    # 延迟导入避免循环依赖
    from utils.conversation_store import store_exchange, get_history

    llm = _get_llm()

    current_date = datetime.now().strftime("%Y-%m-%d")
    system_content = SYSTEM_PROMPT.format(tool_descriptions=_build_tool_descriptions())
    system_content += f"\n\nCurrent date: {current_date}"

    messages = [SystemMessage(content=system_content)]

    # Add history — 优先使用传入的 history_messages，否则从数据库恢复
    if not history_messages:
        history_messages = get_history(thread_id, limit=50)
    for msg in history_messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Add the new question
    messages.append(HumanMessage(content=user_question))

    source_cards = ""
    final_answer = ""

    for iteration in range(5):  # Max 5 tool calls per question
        logger.info(f"[ReAct] Iteration {iteration + 1}, thinking...")
        yield "", f"[ReAct] Thinking... (step {iteration + 1})", source_cards

        try:
            response = llm.invoke(messages)
            content = response.content.strip()
            logger.info(f"[ReAct] Model output: {content[:200]}...")
        except Exception as e:
            logger.error(f"[ReAct] LLM call failed: {e}")
            yield f"Error: {e}", "Error", source_cards
            return

        # Check if it's a tool call
        parsed = _parse_tool_call(content)

        if parsed:
            tool_name, args = parsed
            logger.info(f"[ReAct] Calling tool: {tool_name}({args})")
            yield "", f"Calling {tool_name}...", source_cards

            result = _execute_tool(tool_name, args)
            logger.info(f"[ReAct] Tool result: {result[:200]}...")

            # Add tool call and result to messages
            messages.append(AIMessage(content=f"TOOL: {tool_name}\nARGS: {json.dumps(args, ensure_ascii=False)}"))
            messages.append(HumanMessage(content=f"Tool result: {result}"))

            # Collect source evidence（限制长度，防止无限制增长）
            _MAX_SOURCE_TOTAL = 2000   # source_cards 总长度上限
            _MAX_PER_TOOL = 800        # 单条工具结果最大长度
            if tool_name == "analyze_catl_report":
                addition = f"\n#### Tool: {tool_name}\n> {result[:_MAX_PER_TOOL]}\n"
                if len(source_cards) + len(addition) <= _MAX_SOURCE_TOTAL:
                    source_cards += addition
                logger.info(f"[ReAct] source_cards updated (analyze_catl_report), length: {len(source_cards)}")
            elif tool_name == "web_search_tool":
                addition = f"\n#### Web Search\n> {result[:_MAX_PER_TOOL]}\n"
                if len(source_cards) + len(addition) <= _MAX_SOURCE_TOTAL:
                    source_cards += addition
                logger.info(f"[ReAct] source_cards updated (web_search_tool), length: {len(source_cards)}")

        else:
            # It's a final answer
            # Extract from FINAL: format or use the whole content
            final_match = re.search(r'FINAL:\s*(.+)', content, re.DOTALL)
            if final_match:
                final_answer = final_match.group(1).strip()
            else:
                final_answer = content
            logger.info(f"[ReAct] Final answer: {final_answer[:200]}...")
            break

        # Add observation prompt
        messages.append(HumanMessage(content="Based on this result, provide the final answer or call another tool if needed."))

    if not final_answer:
        final_answer = "Sorry, unable to generate a complete answer. Please try rephrasing."

    # 持久化对话到 SQLite（仅在非空会话时存储）
    if thread_id and thread_id != "default":
        try:
            store_exchange(thread_id, user_question, final_answer)
        except Exception as e:
            logger.warning(f"Failed to persist conversation: {e}")

    logger.info(f"[ReAct] FINAL yield: source_cards length={len(source_cards)}, content preview: {source_cards[:100]}...")
    yield final_answer, "Done", source_cards
