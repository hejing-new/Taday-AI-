import sys
import os

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import gradio as gr
import time
import requests
import json
import uuid

# ReAct engine (replaces LangGraph)
from core.react_engine import run_react

CURRENT_SESSION_ID = f"sess_{uuid.uuid4().hex[:8]}"

from config import API_URL, ADMIN_API_URL, CHAT_MODEL
from logger import logger


def real_rag_backend(user_message, history, s_id):
    """Call ReAct engine: Reason -> Act -> Observe loop"""
    yield "", "Taday Brain is thinking...", ""

    query_str = user_message[0].get("text", str(user_message)) if isinstance(user_message, list) else str(user_message)

    try:
        thinking_log = "### Taday Brain Reasoning Process:\n\n"
        final_answer = ""
        source_cards = ""

        # Convert history to simple dict list
        history_msgs = []
        for msg in history:
            history_msgs.append({"role": msg["role"], "content": msg["content"]})

        for answer, status, sources in run_react(query_str, thread_id=s_id, history_messages=history_msgs):
            if answer:
                final_answer = answer
            if sources:
                source_cards = sources
            logger.info(f"[DEBUG real_rag_backend] yield: status={status[:40]}, source_cards_len={len(source_cards)}, answer_len={len(answer)}")
            yield thinking_log, status, source_cards

        if not final_answer:
            final_answer = "Sorry, unable to generate an answer. Please try again."

        logger.info(f"[DEBUG real_rag_backend] FINAL yield: source_cards_len={len(source_cards)}, source_cards_preview={source_cards[:150]}")
        yield final_answer, "Done", source_cards

    except Exception as e:
        logger.error(f"ReAct engine error: {e}")
        yield f"Error: {str(e)}", "Error", ""
        
# ==========================================
# 🎨 前端 UI 布局
# ==========================================
with gr.Blocks(title="Taday 智能助手") as demo:
    # 🚀 核心修复：将 session_id 存入 gr.State，确保每个浏览器窗口独立
    session_id_state = gr.State(lambda: str(uuid.uuid4()))
    panel_is_visible = gr.State(False)
    downvoted_records = gr.State(set())  # 🌟 新增这行：用一个集合(Set)来当备忘录
    
    with gr.Row():
        # ---------------- 1. 左侧栏 ----------------
        with gr.Column(scale=1, min_width=200):
            new_chat_btn = gr.Button("➕ 新建对话", variant="primary")

            # 🚀 核心新增：文件上传组件
            upload_comp = gr.File(label="📎 上传临时 PDF (仅限本轮对话)", file_types=[".pdf"])
            upload_status = gr.Markdown("状态: 🌐 连接公共库")

            gr.Markdown("---")
            gr.Markdown("### 📜 快捷提问")
            btn_q1 = gr.Button("💬 营收查询", size="sm")
            btn_q2 = gr.Button("💬 宁德时代2025", size="sm")
            gr.Markdown("---")
            gr.Dropdown(choices=[CHAT_MODEL], value=CHAT_MODEL, label="当前模型")

            # 🚀 核心新增：底部后台入口
            gr.Markdown("---")
            gr.Markdown("### 🏢 运营管理")
            admin_btn = gr.Button("⚙️ 进入知识库后台", variant="secondary")
            # 🚀 核心新增：底部后台入口, 新增一个占位符，用于显示跳转链接
            admin_link_area = gr.Markdown("")
        
        # ---------------- 2. 中间栏 ----------------
        with gr.Column(scale=3):
            # chatbot 已在"对话历史恢复"区块定义，支持自动恢复
            with gr.Row():
                thinking_status = gr.Markdown("*💡 等待提问...*")
                toggle_btn = gr.Button("⬅️ 展开溯源面板", size="sm", scale=0)
            with gr.Row():
                msg = gr.Textbox(placeholder="请输入问题...", show_label=False, scale=5, lines=2)
                submit_btn = gr.Button("发送 🚀", variant="primary", scale=1)
                
        # ---------------- 3. 右侧栏 (默认隐藏) ----------------
        with gr.Column(scale=2, visible=False) as right_column:
            gr.Markdown("### 🔍 数据溯源阅读器")
            source_panel = gr.Markdown(value="> 待检索...", elem_id="source_panel")

    # ==========================================
    # 对话历史恢复
    # ==========================================
    def restore_history(s_id):
        """页面加载时从后端恢复对话历史"""
        try:
            resp = requests.get(f"{API_URL}/api/v1/history/{s_id}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                history = data.get("messages", [])
                # 转换为 Gradio Chatbot 格式: [{"role": "user", "content": "..."}, ...]
                return history
        except Exception as e:
            logger.warning(f"恢复历史失败: {e}")
        return []

    def clear_and_new_session(s_id):
        """清空当前会话并创建新的"""
        try:
            requests.delete(f"{API_URL}/api/v1/history/{s_id}", timeout=5)
        except Exception as e:
            logger.warning(f"清空历史失败: {e}")
        return [], "*💡 等待提问...*", "待检索..."

    # 页面加载时恢复历史
    chatbot = gr.Chatbot(
        height=550, label="智能对话区",
        value=lambda: restore_history(CURRENT_SESSION_ID),
    )

    # ==========================================
    # 交互逻辑：文件上传处理
    # ==========================================
   # 修改 handle_upload 函数，接收 session_id
    def handle_upload(file, s_id):
        if file is None: return "状态: 🌐 连接公共库"
        try:
            with open(file.name, "rb") as f:
                files = {"file": (os.path.basename(file.name), f, "application/pdf")}
                # 🚀 使用传入的独立 s_id
                resp = requests.post(f"{API_URL}/api/v1/upload_temp?session_id={s_id}", files=files, timeout=60)
            return f"✅ 已挂载: {os.path.basename(file.name)}"
        except Exception as e:
            return f"❌ 错误: {str(e)}"

    # 绑定上传事件时传入 session_id_state
    upload_comp.change(handle_upload, inputs=[upload_comp, session_id_state], outputs=[upload_status])

    def user_action(user_message, history):
        # 🚀 修正 1：history 追加字典而非列表
        history.append({"role": "user", "content": user_message})
        return "", history

    # 修改 bot_action 函数，从状态中获取 session_id
    def bot_action(history, s_id):
        user_message = history[-1]["content"]
        history.append({"role": "assistant", "content": ""})

        # 🌟 埋点第一步：掐表计时
        start_time = time.time()

        # 流式返回大模型的思考与结果
        source_text_final = ""
        for partial_text, status_text, source_text in real_rag_backend(user_message, history[:-2], s_id):
            history[-1]["content"] = partial_text
            if source_text:
                source_text_final = source_text
            yield history, f"*💡 {status_text}*", source_text_final

        # ==========================================
        # 🌟 埋点第二步：大模型回答完毕，停止计时
        # ==========================================
        latency = time.time() - start_time
        
        # 强力清洗外壳
        clean_query = str(user_message).strip()
        if clean_query.startswith("[{") and clean_query.endswith("}]"):
            try:
                import ast
                pq = ast.literal_eval(clean_query)
                clean_query = pq[0]['text'] if isinstance(pq, list) and len(pq)>0 else clean_query
            except:
                pass

        # ==========================================
        # 🌟 埋点第三步：强制发送请求，并加上显式打印
        # ==========================================
        logger.info(f"BI 准备发送 | 提问: {clean_query[:10]}... | 耗时: {latency:.2f}s")

        try:
            resp = requests.post(
                f"{ADMIN_API_URL}/log_search",
                json={
                    "user_query": clean_query,
                    "session_id": str(s_id),
                    "latency": round(latency, 2)
                },
                timeout=3
            )
            logger.info(f"BI 发送成功 | 后端响应码: {resp.status_code}")
        except Exception as e:
            logger.error(f"BI 发送失败: {e}")

        # 🌟 极其关键的一步：在发送完数据后，再 yield 最后一次！
        # 这样能强迫 Gradio 耐心等待上面那段 requests.post 执行完毕，绝对不会中途掐断！
        logger.info(f"[DEBUG bot_action] FINAL yield: source_text_final length={len(source_text_final)}")
        yield history, "✅ 回答完毕", source_text_final

    # 事件绑定记得加上 session_id_state
    msg.submit(user_action, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_action, [chatbot, session_id_state], [chatbot, thinking_status, source_panel]
    )
    
    submit_btn.click(user_action, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_action, [chatbot, session_id_state], [chatbot, thinking_status, source_panel]
    )

    # 展开/收起逻辑
    def toggle_panel(current_state):
        new_state = not current_state
        return gr.update(visible=new_state), gr.update(value="➡️ 收起" if new_state else "⬅️ 展开"), new_state

    toggle_btn.click(toggle_panel, [panel_is_visible], [right_column, toggle_btn, panel_is_visible])

    # 快捷键与清空
    btn_q1.click(lambda: "宁德时代2025年动力电池系统的营业收入是多少？", None, msg)
    btn_q2.click(lambda: "总结一下宁德时代2025年的发展趋势", None, msg)

    # 清空时同步删除后端历史，并创建新会话
    new_chat_btn.click(
        lambda s_id: clear_and_new_session(s_id),
        inputs=[session_id_state],
        outputs=[chatbot, thinking_status, source_panel],
    )

    # ==========================================
    # 🚀 修改：生成带安全提示的后台跳转链接
    # ==========================================
    def goto_admin():
        # 返回一段带有超链接的 Markdown 文本
        return (
            "👉 **[点击此处进入知识库控制台 (将在新标签页打开)](http://127.0.0.1:7861)**\n\n"
            "*🔒 提示：此区域仅限管理员访问。*"
        )
        
    # 将生成的 Markdown 渲染到刚才留的占位符里
    admin_btn.click(goto_admin, inputs=None, outputs=[admin_link_area])

    # ==========================================
    # 🚀 真正接入后端的 RLHF 反馈机制 (带状态记忆的撤销版)
    # ==========================================
    # 🌟 注意这里多了一个参数 downvoted_set
    def handle_vote(vote: gr.LikeData, current_history, downvoted_set):
        logger.info("捕捉到点踩/点赞动作，开始精准提取")
        
        # 1. 深度清洗 AI 回答的外壳
        clean_val = str(vote.value).strip()
        if clean_val.startswith("['") and clean_val.endswith("']"):
            ai_response_text = clean_val[2:-2]
        elif clean_val.startswith('["') and clean_val.endswith('"]'):
            ai_response_text = clean_val[2:-2]
        else:
            ai_response_text = clean_val

        # 2. 提取原问题（反向遍历找最后一条 user 消息，更可靠）
        user_query_text = "【系统异常】前端上下文提取失败"
        try:
            if hasattr(vote, 'index') and vote.index is not None:
                idx = vote.index[0] if isinstance(vote.index, (list, tuple)) else vote.index
                # 从 idx 往前找最后一条 role=user 的消息
                for i in range(idx - 1, -1, -1):
                    msg = current_history[i]
                    role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                    if role == "user":
                        if isinstance(msg, dict):
                            user_query_text = msg.get("content", "")
                        elif hasattr(msg, 'content'):
                            user_query_text = msg.content
                        elif isinstance(msg, (list, tuple)):
                            user_query_text = msg[0]
                        break
        except Exception as e:
            logger.warning(f"索引提取异常: {e}")

        # 3. 强力清洗多模态外壳
        user_query_str = str(user_query_text).strip()
        if user_query_str.startswith("[{") and user_query_str.endswith("}]"):
            try:
                import ast
                parsed_query = ast.literal_eval(user_query_str)
                if isinstance(parsed_query, list) and len(parsed_query) > 0 and 'text' in parsed_query[0]:
                    user_query_str = parsed_query[0]['text']
            except:
                pass
        
        payload = {"user_query": user_query_str, "ai_response": str(ai_response_text).strip()}
        
        # 🌟 核心 UX 升级：为这对问答生成一个唯一的身份证号 (Hash)
        qa_key = f"{user_query_str}_{ai_response_text}"

        if vote.liked:
            gr.Info("👍 感谢您的肯定！")
            try:
                requests.post(f"{ADMIN_API_URL}/api/v1/feedback/cancel", json=payload, timeout=3)
                # 如果点赞了，把它从点踩备忘录里划掉
                if qa_key in downvoted_set:
                    downvoted_set.remove(qa_key)
            except: pass
            
        else:
            # 🌟 点踩逻辑分支：检查备忘录
            if qa_key in downvoted_set:
                # 情况 A：如果备忘录里已经有它了，说明这是用户第二次点击 👎！代表【撤销】！
                try:
                    requests.post(f"{ADMIN_API_URL}/api/v1/feedback/cancel", json=payload, timeout=3)
                    gr.Info("🔄 已撤销点踩反馈！")
                    downvoted_set.remove(qa_key) # 从备忘录里划掉
                except Exception as e:
                    logger.error(f"撤销点踩异常: {e}")
            else:
                # 情况 B：备忘录里没有它，说明这是第一次正常的点踩 👎！代表【写入】！
                try:
                    resp = requests.post(f"{ADMIN_API_URL}/api/v1/feedback", json=payload, timeout=5)
                    if resp.status_code == 200:
                        gr.Info("👎 反馈已同步至高管质检草稿箱！")
                        downvoted_set.add(qa_key) # 记入备忘录
                except Exception as e:
                    logger.error(f"发送反馈异常: {e}")

        # 返回更新后的备忘录给前端 State 保存
        return downvoted_set

    chatbot.like(
        handle_vote, 
        inputs=[chatbot, downvoted_records], # 把记忆体传进函数
        outputs=[downvoted_records]          # 把更新后的记忆体存回来
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())