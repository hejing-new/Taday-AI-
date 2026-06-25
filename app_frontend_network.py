import sys, os, io, html as html_mod
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import gradio as gr
import time, requests, json, uuid
from core.react_engine import run_react
from config import API_URL, ADMIN_API_URL, CHAT_MODEL, PORT_CHAT
from logger import logger

# ── Backend ──

def real_rag_backend(user_message, history, s_id):
    yield "", "思考中...", "", 0
    query_str = user_message[0].get("text", str(user_message)) if isinstance(user_message, list) else str(user_message)
    try:
        final_answer, source_cards, source_count = "", "", 0
        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        for answer, status, sources in run_react(query_str, thread_id=s_id, history_messages=history_msgs):
            if answer:
                final_answer = answer
            if sources:
                source_cards = sources
                source_count = max(source_count, sources.count("来源"))
            yield final_answer, status, source_cards, source_count
        if not final_answer:
            final_answer = "抱歉，无法生成回答，请重试。"
        yield final_answer, "Done", source_cards, source_count
    except requests.exceptions.ConnectionError:
        yield "⚠️ 无法连接到后端服务，请检查 API 是否启动。", "Error", "", 0
    except requests.exceptions.Timeout:
        yield "⚠️ 请求超时，请稍后重试。", "Error", "", 0
    except Exception as e:
        logger.error(f"ReAct error: {e}")
        yield f"⚠️ 异常: {str(e)}", "Error", "", 0

def restore_history(s_id):
    try:
        resp = requests.get(f"{API_URL}/api/v1/history/{s_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("messages", [])
    except: pass
    return []

def clear_and_new_session(s_id):
    try:
        requests.delete(f"{API_URL}/api/v1/history/{s_id}", timeout=5)
    except Exception:
        pass
    empty_src = "<div class='source-empty'>回答生成后，相关证据将显示在这里</div>"
    return str(uuid.uuid4()), [], "", empty_src

# ── Source formatter ──

def format_sources(raw):
    if not raw or raw == "待检索...":
        return "<div class='source-empty'>暂无溯源数据</div>"
    sections = [s.strip() for s in raw.split("---") if s.strip()]
    cards = []
    for i, sec in enumerate(sections):
        lines = sec.split("\n")
        src = score = ""
        content_lines = []
        for l in lines:
            if l.startswith("来源"): src = l
            elif "相关度" in l: score = l.strip().strip("[]")
            else: content_lines.append(l)
        content = "\n".join(content_lines).strip()
        if content.endswith("..."): content = content[:-3]
        badge = f'<span style="font-size:11px;background:#e8f0fe;color:#1a73e8;padding:3px 10px;border-radius:999px;">{html_mod.escape(score)}</span>' if score else ''
        cards.append(f'''<div style="background:#fff;border:1px solid #e3e8ef;border-radius:12px;padding:14px 16px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;gap:8px;">
    <span style="font-size:13px;font-weight:500;color:#1a73e8;">📄 {html_mod.escape(src) if src else f"证据 {i+1}"}</span>{badge}</div>
  <div style="font-size:13px;color:#444;line-height:1.7;">{html_mod.escape(content[:300])}{"..." if len(content)>300 else ""}</div></div>''')
    return "\n".join(cards) if cards else f"<div class='source-empty'>{html_mod.escape(raw[:500])}</div>"

WELCOME = """
<div class="welcome-inner">
  <h1 class="welcome-greeting">你好，有什么可以帮你？</h1>
  <p class="welcome-desc">财报检索 · 财务 SQL · 实时资讯 · 数据溯源</p>
  <div class="suggest-chips">
    <span class="chip">📊 查询营收与利润</span>
    <span class="chip">🔋 宁德时代 2025 年报</span>
    <span class="chip">📈 毛利率趋势分析</span>
    <span class="chip">🌐 最新行业动态</span>
  </div>
</div>"""

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,wght@6..144,400;6..144,500;6..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
"""

CSS = """
*,*::before,*::after{box-sizing:border-box}
html,body{font-family:'Google Sans Flex','Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;height:100%!important;overflow:hidden!important;margin:0!important;padding:0!important}
.gradio-container{height:100vh!important;max-height:100vh!important;max-width:100vw!important;overflow:hidden!important;margin:0!important;padding:0!important;background:#fff!important}
.contain,.wrap,.panel-wrap{padding:0!important;gap:0!important;max-width:100%!important}
.main,.app{height:100vh!important;overflow:hidden!important;padding:0!important}
.gradio-container footer,.footer{display:none!important}
.block,.form{background:transparent!important;border:none!important;box-shadow:none!important;padding:0!important;margin:0!important}
.layout-row{display:flex!important;flex-direction:row!important;height:100vh!important;width:100%!important;overflow:hidden!important;gap:0!important;margin:0!important;padding:0!important}

/* ── 左侧栏 25% ── */
.sidebar{min-width:220px!important;max-width:300px!important;height:100vh!important;overflow-y:auto!important;overflow-x:hidden!important;background:#f0f4f9!important;padding:12px 12px 16px!important;flex-shrink:0!important;border-right:1px solid #e3e8ef!important;display:flex!important;flex-direction:column!important}
.sidebar>.block,.sidebar .form{gap:2px!important}
.sidebar .brand{display:flex;align-items:center;gap:10px;padding:8px 8px 16px}
.sidebar .brand-icon{font-size:22px;line-height:1}
.sidebar .brand-text{flex:1}
.sidebar .brand-name{font-size:16px;font-weight:600;color:#1f1f1f;letter-spacing:-.01em;line-height:1.2}
.sidebar .brand-sub{font-size:11px;color:#5f6368;margin-top:1px}
.sidebar .section-label{color:#5f6368!important;font-size:11px!important;font-weight:500!important;margin:14px 8px 6px!important;padding:0!important}
.sidebar .divider{height:1px!important;background:#e3e8ef!important;margin:8px 0!important;border:none!important}
.sidebar .gr-button-primary{background:#dde3ea!important;border:none!important;color:#1f1f1f!important;font-weight:500!important;border-radius:999px!important;padding:10px 16px!important;margin:0 0 4px!important;width:100%!important;font-size:14px!important;box-shadow:none!important;transition:background .15s!important;justify-content:flex-start!important}
.sidebar .gr-button-primary:hover{background:#d3dbe4!important;transform:none!important;box-shadow:none!important}
.sidebar .gr-button-secondary{background:transparent!important;border:none!important;color:#3c4043!important;border-radius:999px!important;font-size:13px!important;padding:9px 14px!important;margin:1px 0!important;width:100%!important;text-align:left!important;justify-content:flex-start!important;transition:background .12s!important;font-weight:400!important}
.sidebar .gr-button-secondary:hover{background:#e2e7ee!important;color:#1f1f1f!important;border:none!important}
.sidebar .nav-btn .gr-button-secondary{font-size:13px!important}
.sidebar .gr-file{border:1px dashed #c4c7c5!important;border-radius:12px!important;background:#fff!important;padding:8px!important;min-height:auto!important}
.sidebar .gr-file label,.sidebar .gr-dropdown label{color:#5f6368!important;font-size:11px!important;font-weight:500!important}
.sidebar .gr-dropdown{background:#fff!important;border:1px solid #e3e8ef!important;border-radius:10px!important}
.sidebar .upload-hint{color:#80868b!important;font-size:11px!important;line-height:1.5!important;margin:4px 8px 0!important}
.sidebar .admin-link{margin:4px 8px 0!important}
.sidebar .admin-link a{color:#1a73e8!important;font-size:12px!important;text-decoration:none!important}
.sidebar .admin-link a:hover{text-decoration:underline!important}
.sidebar-footer{margin-top:auto!important;padding-top:8px!important}

/* ── 中间主区域 ── */
.main-center{flex:1!important;min-width:0!important;height:100vh!important;display:flex!important;flex-direction:column!important;overflow:hidden!important;background:radial-gradient(ellipse 80% 60% at 50% 0%,#dce8ff 0%,#eef3fc 35%,#f8fafc 70%,#fff 100%)!important;padding:0!important;position:relative!important}
.main-topbar{flex-shrink:0!important;padding:12px 20px!important;display:flex!important;align-items:center!important;justify-content:flex-end!important;gap:8px!important;min-height:48px!important;background:transparent!important;margin:0!important}
.main-topbar .gr-button-secondary,.main-topbar .gr-button-primary{border-radius:999px!important;font-size:12px!important;padding:7px 16px!important;border:1px solid #e3e8ef!important;background:rgba(255,255,255,.7)!important;color:#3c4043!important;font-weight:500!important;backdrop-filter:blur(8px)!important;box-shadow:0 1px 2px rgba(0,0,0,.04)!important}
.main-topbar .gr-button-primary{background:#1a73e8!important;border-color:#1a73e8!important;color:#fff!important}
.thinking-bar{flex-shrink:0!important;padding:0 24px 8px!important;text-align:center!important;min-height:0!important}
.thinking-bar .gr-markdown p{margin:0!important}
.chat-scroll{flex:1!important;min-height:0!important;overflow:hidden!important;display:flex!important;flex-direction:column!important;padding:0!important}
.welcome-box{flex:1!important;min-height:0!important;overflow-y:auto!important;display:flex!important;align-items:center!important;justify-content:center!important;padding:0 24px 24px!important}
.welcome-inner{text-align:center;max-width:720px;width:100%}
.welcome-greeting{font-size:clamp(28px,4vw,44px)!important;font-weight:400!important;color:#1f1f1f!important;margin:0 0 12px!important;letter-spacing:-.02em!important;line-height:1.2!important;font-family:'Google Sans Flex','Inter',sans-serif!important}
.welcome-desc{font-size:15px;color:#5f6368;margin:0 0 28px;line-height:1.5}
.suggest-chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center}
.chip{display:inline-flex;align-items:center;padding:10px 18px;background:rgba(255,255,255,.85);border:1px solid #e3e8ef;border-radius:999px;font-size:13px;color:#3c4043;cursor:default;transition:all .15s;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.chip:hover{background:#fff;border-color:#c4c7c5;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.chatbot-inner{flex:1!important;min-height:0!important;overflow-y:auto!important;padding:12px 24px 8px!important;border:none!important;background:transparent!important;max-width:820px!important;width:100%!important;margin:0 auto!important;height:auto!important;min-height:unset!important}
.chatbot-inner>.wrap,.chatbot-inner .bubble-wrap,.chatbot-inner .component-wrap{height:auto!important;min-height:unset!important;max-height:100%!important}
.chatbot-inner .message{border-radius:18px!important;padding:10px 16px!important;margin:4px 0!important;max-width:85%!important;font-size:15px!important;line-height:1.6!important}
.chatbot-inner .message.user{background:#e8f0fe!important;color:#1f1f1f!important;border-bottom-right-radius:6px!important;margin-left:auto!important;box-shadow:none!important;border:1px solid #d2e3fc!important}
.chatbot-inner .message.bot,.chatbot-inner .message.assistant{background:rgba(255,255,255,.92)!important;color:#1f1f1f!important;border-bottom-left-radius:6px!important;box-shadow:0 1px 3px rgba(0,0,0,.06)!important;border:1px solid #e8eaed!important}
.answer-meta{display:flex;align-items:center;justify-content:center;gap:10px;padding:2px 0;font-size:11px;color:#80868b}
.am-badge{display:inline-flex;align-items:center;gap:4px;padding:4px 12px;border-radius:999px;font-size:11px;font-weight:500;background:rgba(255,255,255,.8);border:1px solid #e3e8ef}
.am-time{color:#5f6368}.am-src{color:#137333}
@keyframes tp{0%,100%{opacity:.3}50%{opacity:1}}
.td{display:inline-block;width:5px;height:5px;border-radius:50%;background:#1a73e8;margin:0 2px;animation:tp 1.4s infinite}
.td:nth-child(2){animation-delay:.2s}.td:nth-child(3){animation-delay:.4s}

/* ── 底部输入框 Gemini 胶囊 ── */
.input-area{flex-shrink:0!important;padding:12px 24px 28px!important;background:transparent!important;border:none!important;margin:0!important;display:flex!important;justify-content:center!important;align-items:center!important;gap:0!important}
.input-wrap{max-width:720px!important;width:100%!important;display:flex!important;align-items:flex-end!important;gap:8px!important;background:#fff!important;border:1px solid #e3e8ef!important;border-radius:28px!important;padding:6px 6px 6px 4px!important;box-shadow:0 2px 12px rgba(0,0,0,.08)!important;transition:box-shadow .2s,border-color .2s!important}
.input-wrap:focus-within{border-color:#c4c7c5!important;box-shadow:0 4px 20px rgba(0,0,0,.1)!important}
.input-box{flex:1!important;border:none!important;background:transparent!important;box-shadow:none!important;border-radius:0!important}
.input-box:focus-within{border:none!important;box-shadow:none!important}
.input-box textarea{background:transparent!important;font-size:15px!important;padding:12px 8px 12px 16px!important;color:#1f1f1f!important;line-height:1.5!important;border:none!important;min-height:24px!important}
.input-box .wrap{border:none!important;box-shadow:none!important;background:transparent!important}
.input-area .send-btn{background:#1a73e8!important;border:none!important;border-radius:999px!important;min-width:44px!important;width:44px!important;height:44px!important;font-weight:600!important;font-size:13px!important;box-shadow:none!important;flex-shrink:0!important;padding:0!important;color:#fff!important}
.input-area .send-btn:hover{background:#1765cc!important;box-shadow:0 2px 8px rgba(26,115,232,.3)!important}

/* ── 右侧溯源面板 26% ── */
.source-col{min-width:260px!important;max-width:360px!important;height:100vh!important;overflow:hidden!important;border-left:1px solid #e3e8ef!important;padding:0!important;background:#f8fafd!important;flex-shrink:0!important;display:flex!important;flex-direction:column!important}
.source-header{flex-shrink:0!important;padding:14px 16px!important;border-bottom:1px solid #e3e8ef!important;background:#fff!important;display:flex!important;align-items:center!important;justify-content:space-between!important;gap:12px!important;margin:0!important;width:100%!important}
.source-header .gr-markdown{flex:1!important;min-width:0!important}
.source-header .gr-markdown p{margin:0!important;font-size:15px!important;font-weight:600!important;color:#1f1f1f!important}
.source-header .close-panel-btn{border-radius:8px!important;font-size:13px!important;padding:6px 14px!important;border:1px solid #e3e8ef!important;background:#fff!important;color:#3c4043!important;min-width:56px!important;flex-shrink:0!important;cursor:pointer!important}
.source-header .close-panel-btn:hover{background:#f1f3f4!important;border-color:#dadce0!important}
.source-body{flex:1!important;overflow-y:auto!important;padding:14px 16px!important}
.source-panel{font-size:13px!important;line-height:1.7!important;color:#3c4043!important}
.source-empty{color:#80868b!important;font-size:13px!important;text-align:center!important;padding:40px 16px!important}

::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-thumb{background:#dadce0;border-radius:10px}::-webkit-scrollbar-track{background:transparent}
.hist-item{padding:9px 14px;border-radius:999px;cursor:pointer;transition:background .12s;color:#3c4043;font-size:13px;margin:2px 0}
.hist-item:hover{background:#e2e7ee}
.hist-item.active{background:#d3e3fd;color:#1a73e8;font-weight:500}
@media(max-width:960px){
  .sidebar{width:240px!important;min-width:240px!important;max-width:240px!important}
  .source-col{width:280px!important;min-width:280px!important;max-width:280px!important}
  .welcome-greeting{font-size:28px!important}
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#1a73e8",
    button_primary_background_fill_hover="#1765cc",
    button_primary_text_color="white",
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_hover="#e2e7ee",
    block_background_fill="#ffffff",
    body_background_fill="#ffffff",
    input_background_fill="#ffffff",
    border_color_primary="#e3e8ef",
)

with gr.Blocks(title="Taday 智能助手", fill_height=True) as demo:
    session_id_state = gr.State(lambda: str(uuid.uuid4()))
    panel_is_visible = gr.State(False)
    downvoted_records = gr.State(set())
    history_list = gr.State([])
    welcome_visible = gr.State(True)

    with gr.Row(elem_classes=["layout-row"]):
        # ── 左侧栏 ~26% ──
        with gr.Column(scale=2, min_width=220, elem_classes=["sidebar"]):
            gr.Markdown(
                "<div class='brand'>"
                "<span class='brand-icon'>✦</span>"
                "<div class='brand-text'>"
                "<div class='brand-name'>Taday</div>"
                "<div class='brand-sub'>金融智能助手</div>"
                "</div></div>"
            )
            new_chat_btn = gr.Button("＋ 新建对话", variant="primary", size="sm")
            gr.Markdown("<div class='section-label'>快捷提问</div>")
            with gr.Column(elem_classes=["nav-btn"]):
                btn_q1 = gr.Button("📊 营收与利润查询", variant="secondary", size="sm")
                btn_q2 = gr.Button("🔋 宁德时代 2025 年报", variant="secondary", size="sm")
                btn_q3 = gr.Button("📈 毛利率趋势分析", variant="secondary", size="sm")
                btn_q4 = gr.Button("🌐 最新行业动态", variant="secondary", size="sm")
            gr.Markdown("<div class='divider'></div>")
            gr.Markdown("<div class='section-label'>历史会话</div>")
            history_container = gr.Column()
            gr.Markdown("<div class='divider'></div>")
            gr.Markdown("<div class='section-label'>上传财报</div>")
            upload_comp = gr.File(label="PDF 文件", file_types=[".pdf"], height=48)
            upload_status = gr.Markdown("<div class='upload-hint'>🌐 当前连接公共知识库</div>")
            with gr.Column(elem_classes=["sidebar-footer"]):
                gr.Markdown("<div class='divider'></div>")
                gr.Dropdown(choices=[CHAT_MODEL], value=CHAT_MODEL, label="模型", min_width=100)
                admin_btn = gr.Button("⚙ 管理后台", variant="secondary", size="sm")
                admin_link_area = gr.Markdown("", elem_classes=["admin-link"])

        # ── 中间主区域 ──
        with gr.Column(scale=8, elem_classes=["main-center"]):
            with gr.Row(elem_classes=["main-topbar"]):
                toggle_btn = gr.Button("📑 数据溯源", size="sm", variant="secondary", scale=0)

            with gr.Column(elem_classes=["chat-scroll"]):
                welcome_box = gr.HTML(WELCOME, visible=True, elem_classes=["welcome-box"])
                chatbot = gr.Chatbot(
                    value=[], elem_classes=["chatbot-inner"], show_label=False,
                    height=420, visible=False, autoscroll=True, layout="bubble",
                )

            with gr.Row(elem_classes=["thinking-bar"]):
                thinking_status = gr.Markdown("")

            with gr.Row(elem_classes=["input-area"]):
                with gr.Row(elem_classes=["input-wrap"]):
                    msg = gr.Textbox(
                        placeholder="向 Taday 提问…",
                        show_label=False, scale=1, lines=1,
                        elem_classes=["input-box"], max_lines=6,
                    )
                    submit_btn = gr.Button("↑", variant="primary", scale=0, elem_classes=["send-btn"])

        # ── 右侧溯源面板 ~26% ──
        with gr.Column(scale=2, visible=False, elem_classes=["source-col"]) as right_column:
            with gr.Row(elem_classes=["source-header"]):
                gr.Markdown("**🔍 数据溯源**")
                close_panel_btn = gr.Button("收起", size="sm", variant="secondary", scale=0, elem_classes=["close-panel-btn"])
            with gr.Column(elem_classes=["source-body"]):
                source_panel = gr.HTML(
                    value="<div class='source-empty'>回答生成后，相关证据将显示在这里</div>",
                    elem_classes=["source-panel"],
                )

    # ── Interactions ──

    def handle_upload(file, s_id):
        if file is None: return "<div class='upload-hint'>🌐 当前连接公共知识库</div>"
        try:
            with open(file.name, "rb") as f:
                files = {"file": (os.path.basename(file.name), f, "application/pdf")}
                resp = requests.post(f"{API_URL}/api/v1/upload_temp?session_id={s_id}", files=files, timeout=60)
            return f"<div class='upload-hint'>✅ 已挂载: {html_mod.escape(os.path.basename(file.name))}</div>"
        except Exception as e: return f"<div class='upload-hint'>❌ 错误: {html_mod.escape(str(e))}</div>"

    upload_comp.change(handle_upload, inputs=[upload_comp, session_id_state], outputs=[upload_status])

    def user_action(user_message, history):
        history.append({"role": "user", "content": user_message})
        return "", history

    def bot_action(history, s_id):
        user_message = history[-1]["content"]
        history.append({"role": "assistant", "content": ""})
        start_time = time.time()
        source_text_final = ""
        source_count_final = 0

        for partial_text, status_text, source_text, source_count in real_rag_backend(user_message, history[:-2], s_id):
            history[-1]["content"] = partial_text
            if source_text:
                source_text_final = source_text
                source_count_final = source_count
            if status_text and status_text not in ("Done", "Error"):
                dots = '<span class="td"></span><span class="td"></span><span class="td"></span>'
                status_display = f'<span style="color:#1a73e8;font-size:13px;">{dots} {html_mod.escape(status_text)}</span>'
            elif status_text == "Done":
                latency = time.time() - start_time
                meta = f'<div class="answer-meta"><span class="am-badge am-time">⏱ {latency:.1f}s</span>'
                if source_count_final > 0:
                    meta += f'<span class="am-badge am-src">📄 {source_count_final} 条证据</span>'
                meta += '</div>'
                status_display = meta
            elif status_text == "Error":
                status_display = f'<span style="color:#d93025;font-size:13px;">⚠️ {html_mod.escape(partial_text or "请求失败")}</span>'
            else:
                status_display = ""
            history[-1]["content"] = partial_text if partial_text else ("正在思考…" if status_text not in ("Done", "Error") else partial_text)
            yield history, status_display, format_sources(source_text_final)

        # BI tracking
        latency = time.time() - start_time
        clean_query = str(user_message).strip()
        if clean_query.startswith("[{") and clean_query.endswith("}]"):
            try:
                import ast
                pq = ast.literal_eval(clean_query)
                clean_query = pq[0]['text'] if isinstance(pq, list) and len(pq) > 0 else clean_query
            except: pass
        try:
            requests.post(f"{ADMIN_API_URL}/log_search", json={"user_query": clean_query[:200], "session_id": str(s_id), "latency": round(latency, 2)}, timeout=3)
        except: pass

        yield history, '<div class="answer-meta"><span class="am-badge am-time">⏱ {:.1f}s</span></div>'.format(latency), format_sources(source_text_final)

    def toggle_panel(current_state):
        new_state = not current_state
        label = "收起溯源" if new_state else "📑 数据溯源"
        return (
            gr.update(visible=new_state),
            gr.update(value=label, variant="primary" if new_state else "secondary"),
            new_state,
        )

    def close_panel(_current_state):
        return gr.update(visible=False), gr.update(value="📑 数据溯源", variant="secondary"), False

    def show_welcome():
        return gr.update(visible=True), gr.update(visible=False)

    def show_chat():
        return gr.update(visible=False), gr.update(visible=True)

    def add_to_history(s_id, history):
        """Track conversation in sidebar"""
        if len(history) >= 2:  # At least one exchange
            query = history[-2]["content"][:30] if len(history) >= 2 else ""
            hist = [{"id": s_id, "query": query, "time": time.strftime("%H:%M")}]
        else:
            hist = []
        return hist

    # Event bindings
    msg.submit(user_action, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_action, [chatbot, session_id_state], [chatbot, thinking_status, source_panel]
    ).then(show_chat, outputs=[welcome_box, chatbot])

    submit_btn.click(user_action, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_action, [chatbot, session_id_state], [chatbot, thinking_status, source_panel]
    ).then(show_chat, outputs=[welcome_box, chatbot])

    toggle_btn.click(toggle_panel, [panel_is_visible], [right_column, toggle_btn, panel_is_visible])
    close_panel_btn.click(close_panel, [panel_is_visible], [right_column, toggle_btn, panel_is_visible])

    btn_q1.click(lambda: "宁德时代2025年动力电池系统的营业收入是多少？", None, msg)
    btn_q2.click(lambda: "总结一下宁德时代2025年的发展趋势", None, msg)
    btn_q3.click(lambda: "宁德时代2021-2024年的毛利率变化趋势如何？", None, msg)
    btn_q4.click(lambda: "搜索最新的新能源行业政策新闻", None, msg)

    new_chat_btn.click(
        clear_and_new_session,
        [session_id_state],
        [session_id_state, chatbot, thinking_status, source_panel],
    ).then(
        close_panel, [panel_is_visible], [right_column, toggle_btn, panel_is_visible]
    ).then(
        show_welcome, outputs=[welcome_box, chatbot]
    )

    admin_btn.click(
        lambda: gr.update(value=f"<a href='http://127.0.0.1:7861' target='_blank'>打开管理后台 →</a>"),
        outputs=[admin_link_area]
    )

    # Load history on demo load
    demo.load(
        restore_history,
        [session_id_state],
        [chatbot]
    ).then(
        lambda hist: (gr.update(visible=False), gr.update(visible=True)) if hist else (gr.update(visible=True), gr.update(visible=False)),
        [chatbot],
        [welcome_box, chatbot]
    )

# ==========================================
# 启动服务
# ==========================================
if __name__ == "__main__":
    print(f"🚀 正在启动 Tady 前端界面 (端口: {PORT_CHAT})...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=PORT_CHAT,
        show_error=True,
        share=False,
        css=CSS,
        theme=THEME,
        head=HEAD,
    )
