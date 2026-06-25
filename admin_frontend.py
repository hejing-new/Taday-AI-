import gradio as gr
import requests
import pandas as pd
import time
from config import ADMIN_API_URL, ADMIN_USER, ADMIN_PASS
from logger import logger

ADMIN_AUTH = (ADMIN_USER, ADMIN_PASS)

STATUS_MAP = {
    "processing": "⏳ 解析切分中",
    "pending": "🟠 待人工审核",
    "published": "🟢 已发布入库",
    "failed": "❌ 解析失败",
}

CSS = """
*,*::before,*::after{box-sizing:border-box}
html,body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;height:100%!important;overflow:hidden!important;margin:0!important;padding:0!important}
.gradio-container{height:100vh!important;max-height:100vh!important;max-width:100vw!important;overflow:hidden!important;margin:0!important;padding:0!important;background:#f0f4f9!important}
.contain,.wrap,.panel-wrap{padding:0!important;max-width:100%!important}
.main,.app{height:100vh!important;overflow:hidden!important;padding:0!important}
.gradio-container footer,.footer{display:none!important}
.block,.form{background:transparent!important;border:none!important;box-shadow:none!important}
.admin-shell{display:flex!important;flex-direction:column!important;height:100vh!important;overflow:hidden!important}

/* ── 顶栏 ── */
.header-bar{background:#fff!important;padding:0 24px!important;min-height:60px!important;display:flex!important;align-items:center!important;justify-content:space-between!important;border-bottom:1px solid #e3e8ef!important;flex-shrink:0!important;margin:0!important}
.header-bar .brand-wrap{display:flex;align-items:center;gap:12px}
.header-bar .brand-icon{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#1a73e8,#4285f4);display:flex;align-items:center;justify-content:center;color:#fff;font-size:18px;font-weight:700}
.header-bar .title{font-size:18px;font-weight:600;color:#1f1f1f;line-height:1.2;margin:0}
.header-bar .subtitle{font-size:12px;color:#5f6368;margin-top:2px}
.header-bar .badge{display:inline-flex;align-items:center;padding:4px 10px;border-radius:999px;background:#e8f0fe;color:#1a73e8;font-size:11px;font-weight:500}

/* ── 标签页 ── */
.admin-tabs{flex:1!important;min-height:0!important;overflow:hidden!important;display:flex!important;flex-direction:column!important}
.admin-tabs>.tab-nav{border-bottom:1px solid #e3e8ef!important;background:#fff!important;padding:0 16px!important;flex-shrink:0!important}
.admin-tabs button{border-radius:8px 8px 0 0!important;font-size:13px!important;font-weight:500!important;padding:12px 18px!important;color:#5f6368!important;border:none!important;background:transparent!important}
.admin-tabs button.selected{color:#1a73e8!important;background:#f0f4f9!important;border-bottom:2px solid #1a73e8!important}
.admin-tabs .tabitem{padding:20px 24px 24px!important;overflow-y:auto!important;height:100%!important;background:#f0f4f9!important}

/* ── 卡片 ── */
.card{background:#fff!important;border-radius:14px!important;padding:20px 22px!important;box-shadow:0 1px 2px rgba(60,64,67,.06)!important;border:1px solid #e3e8ef!important;margin-bottom:16px!important}
.card-title{font-size:15px;font-weight:600;color:#1f1f1f;margin-bottom:4px;display:flex;align-items:center;gap:8px}
.card-hint{color:#80868b!important;font-size:12px!important;line-height:1.5!important;margin:0 0 14px!important}
.card-divider{height:1px;background:#e3e8ef;margin:16px 0;border:none}

/* ── 指标卡 ── */
.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:16px}
.metric-card{background:#fff;border:1px solid #e3e8ef;border-radius:12px;padding:16px 18px;box-shadow:0 1px 2px rgba(60,64,67,.04)}
.metric-label{font-size:12px;color:#5f6368;margin-bottom:6px}
.metric-value{font-size:28px;font-weight:600;color:#1f1f1f;line-height:1.1}
.metric-unit{font-size:14px;font-weight:500;color:#80868b;margin-left:4px}

/* ── 表格 ── */
.gr-dataframe{border-radius:10px!important;overflow:hidden!important;border:1px solid #e3e8ef!important;background:#fff!important}
.gr-dataframe th{background:#f8fafd!important;font-weight:600!important;font-size:12px!important;color:#3c4043!important;padding:10px 14px!important;border-bottom:1px solid #e3e8ef!important}
.gr-dataframe td{padding:10px 14px!important;font-size:13px!important;color:#3c4043!important;border-bottom:1px solid #f1f3f4!important}

/* ── 按钮 ── */
.gr-button-primary{background:#1a73e8!important;border:none!important;border-radius:8px!important;font-weight:500!important;padding:9px 18px!important;font-size:13px!important;box-shadow:none!important;transition:background .15s!important}
.gr-button-primary:hover{background:#1765cc!important;transform:none!important;box-shadow:0 1px 4px rgba(26,115,232,.25)!important}
.gr-button-secondary{background:#fff!important;border:1px solid #dadce0!important;border-radius:8px!important;padding:9px 18px!important;font-size:13px!important;color:#3c4043!important;font-weight:500!important}
.gr-button-secondary:hover{background:#f8f9fa!important;border-color:#bdc1c6!important}
.gr-button-stop{background:#fff!important;border:1px solid #f28b82!important;color:#c5221f!important;border-radius:8px!important;padding:9px 18px!important;font-size:13px!important;font-weight:500!important}
.gr-button-stop:hover{background:#fce8e6!important}

/* ── 输入框 ── */
.gr-textbox,.gr-file{border-radius:10px!important;border:1px solid #dadce0!important;background:#fff!important}
.gr-textbox:focus-within,.gr-file:focus-within{border-color:#1a73e8!important;box-shadow:0 0 0 2px rgba(26,115,232,.12)!important}
.gr-textbox label,.gr-file label{font-size:12px!important;font-weight:500!important;color:#5f6368!important}
.gr-file{border-style:dashed!important;padding:14px!important}

/* ── 操作区 ── */
.action-panel{background:#f8fafd!important;border:1px solid #e3e8ef!important;border-radius:12px!important;padding:16px!important}
.action-panel .card-title{font-size:14px!important;margin-bottom:12px!important}
.toolbar{display:flex!important;align-items:center!important;justify-content:space-between!important;gap:12px!important;margin-bottom:12px!important;flex-wrap:wrap!important}
.toolbar-left{color:#80868b;font-size:12px;flex:1;min-width:200px}

::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-thumb{background:#dadce0;border-radius:10px}
::-webkit-scrollbar-track{background:transparent}
@media(max-width:900px){.metric-row{grid-template-columns:1fr 1fr}.admin-tabs .tabitem{padding:16px!important}}
"""

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
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
    block_background_fill="#f0f4f9",
    body_background_fill="#f0f4f9",
    input_background_fill="#ffffff",
    border_color_primary="#e3e8ef",
)


def _metrics_html(total_searches, avg_latency):
    return f"""
<div class="metric-row">
  <div class="metric-card"><div class="metric-label">全站总提问次数</div><div class="metric-value">{total_searches}<span class="metric-unit">次</span></div></div>
  <div class="metric-card"><div class="metric-label">平均响应延迟</div><div class="metric-value">{avg_latency}<span class="metric-unit">秒</span></div></div>
  <div class="metric-card"><div class="metric-label">监控状态</div><div class="metric-value" style="font-size:18px;color:#137333">● 实时</div></div>
</div>"""


# ==========================================
# 🧠 接口通信
# ==========================================

def fetch_dashboard():
    try:
        resp = requests.get(f"{ADMIN_API_URL}/docs", timeout=10, auth=ADMIN_AUTH)
        if resp.status_code == 200:
            docs = resp.json()
            table_data = []
            for d in docs:
                status_icon = STATUS_MAP.get(d['status'], "❓ 未知")
                table_data.append([d['filename'], status_icon, d['chunk_count'], d['created_at'], d['doc_id']])
            return pd.DataFrame(table_data, columns=["文档名称", "当前状态", "切片数量", "上传时间", "文档ID"])
        return pd.DataFrame(columns=["文档名称", "当前状态", "切片数量", "上传时间", "文档ID"])
    except Exception as e:
        logger.error(f"看板获取失败: {e}")
    return pd.DataFrame(columns=["文档名称", "当前状态", "切片数量", "上传时间", "文档ID"])


def handle_upload(file):
    if file is None:
        return fetch_dashboard(), "⚠️ 请选择文件"
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name.split("/")[-1], f, "application/pdf")}
            resp = requests.post(f"{ADMIN_API_URL}/upload", files=files, timeout=60, auth=ADMIN_AUTH)
        if resp.status_code == 200:
            msg = resp.json()["message"]
            return fetch_dashboard(), f"✅ {msg}"
        else:
            return fetch_dashboard(), f"❌ 上传失败: {resp.text}"
    except Exception as e:
        return fetch_dashboard(), f"❌ 接口异常: {str(e)}"


def on_select_doc(evt: gr.SelectData, df: pd.DataFrame):
    row_index = evt.index[0]
    doc_id = df.iloc[row_index]["文档ID"]
    status = df.iloc[row_index]["当前状态"]
    filename = df.iloc[row_index]["文档名称"]

    if "解析切分中" in status:
        title = f"<div class='card-title'>🔍 切片审核工作台</div><p class='card-hint'>⚠️ 《{filename}》正在后台切片中，请稍后再试或点击刷新</p>"
        return doc_id, title, pd.DataFrame(), [], ""

    try:
        resp = requests.get(f"{ADMIN_API_URL}/docs/{doc_id}/chunks", auth=ADMIN_AUTH)
        if resp.status_code == 200:
            chunks = resp.json()
            table_data = []
            for c in chunks:
                preview = c['text_content'][:50].replace("\n", "") + "..."
                s_icon = "🟢 正常" if c['status'] == "active" else "🔴 废弃"
                table_data.append([c['chunk_index'], c['chunk_id'], s_icon, preview, c['text_content']])
            chunk_df = pd.DataFrame(table_data, columns=["序号", "切片ID", "状态", "内容预览", "完整内容"])
            title = f"<div class='card-title'>🔍 切片审核工作台</div><p class='card-hint'>✅ 正在审核：《{filename}》</p>"
            return doc_id, title, chunk_df, chunks, ""
    except Exception as e:
        title = f"<div class='card-title'>🔍 切片审核工作台</div><p class='card-hint' style='color:#c5221f'>❌ 切片拉取失败: {e}</p>"
        return doc_id, title, pd.DataFrame(), [], ""


def on_select_chunk(evt: gr.SelectData, df: pd.DataFrame):
    row_index = evt.index[0]
    chunk_id = df.iloc[row_index]["切片ID"]
    full_text = df.iloc[row_index]["完整内容"]
    return chunk_id, full_text


def _refresh_chunks(doc_id):
    try:
        resp = requests.get(f"{ADMIN_API_URL}/docs/{doc_id}/chunks", auth=ADMIN_AUTH)
        if resp.status_code == 200:
            chunks = resp.json()
            table_data = []
            for c in chunks:
                preview = c['text_content'][:50].replace("\n", "") + "..."
                s_icon = "🟢 正常" if c['status'] == "active" else "🔴 废弃"
                table_data.append([c['chunk_index'], c['chunk_id'], s_icon, preview, c['text_content']])
            return pd.DataFrame(table_data, columns=["序号", "切片ID", "状态", "内容预览", "完整内容"]), chunks
    except Exception as e:
        logger.error(f"刷新切片列表失败: {e}")
    return pd.DataFrame(), []


def save_chunk(chunk_id, new_text, doc_id):
    if not chunk_id:
        return "⚠️ 未选中切片", pd.DataFrame(), []
    requests.put(f"{ADMIN_API_URL}/chunks/{chunk_id}", json={"new_text": new_text}, auth=ADMIN_AUTH)
    chunk_df, raw_chunks = _refresh_chunks(doc_id)
    return f"✅ 切片 {chunk_id} 已保存", chunk_df, raw_chunks


def delete_chunk(chunk_id, doc_id):
    if not chunk_id:
        return "⚠️ 未选中切片", pd.DataFrame(), []
    requests.delete(f"{ADMIN_API_URL}/chunks/{chunk_id}", auth=ADMIN_AUTH)
    chunk_df, raw_chunks = _refresh_chunks(doc_id)
    return f"🗑️ 切片 {chunk_id} 已废弃", chunk_df, raw_chunks


def publish_doc(doc_id):
    if not doc_id:
        return "⚠️ 请先在上方看板选中一个待审核的文档！", fetch_dashboard()
    resp = requests.post(f"{ADMIN_API_URL}/docs/{doc_id}/publish", auth=ADMIN_AUTH, timeout=30)
    if resp.status_code == 200:
        return f"🚀 {resp.json()['message']}", fetch_dashboard()
    return f"❌ 发布失败: {resp.text}", fetch_dashboard()


def fetch_bad_cases():
    try:
        resp = requests.get(f"{ADMIN_API_URL}/bad_cases", timeout=10, auth=ADMIN_AUTH)
        if resp.status_code == 200:
            cases = resp.json()
            table_data = []
            for c in cases:
                s = c.get('status', 'pending')
                if s == 'pending': s_icon = "🔴 待修复"
                elif s == 'manual_review': s_icon = "⚠️ 需人工审核"
                elif s == 'fixed': s_icon = "🟢 人工已修复"
                elif s == 'auto_fixed': s_icon = "🤖 AI已修复"
                elif s == 'ignored': s_icon = "⚪ 垃圾废弃"
                elif s == 'ignored_dynamic': s_icon = "⏳ 动态时效(已归档)"
                else: s_icon = "❓ 未知"

                display_result = c.get('corrected_answer', c.get('admin_note', '-'))
                table_data.append([c['case_id'], s_icon, c['user_query'], c['ai_response'], display_result, c['created_at']])
            return pd.DataFrame(table_data, columns=["案例ID", "处理状态", "用户提问", "AI 翻车回答", "处理结果/修复答案", "发生时间"])
    except Exception as e:
        logger.error(f"拉取草稿箱数据失败: {e}")
    return pd.DataFrame(columns=["案例ID", "处理状态", "用户提问", "AI 翻车回答", "处理结果/修复答案", "发生时间"])


def fetch_analytics():
    try:
        resp = requests.get(f"{ADMIN_API_URL}/analytics", timeout=10, auth=ADMIN_AUTH)
        if resp.status_code == 200:
            data = resp.json()
            metrics = data["metrics"]
            logs = data["logs"]
            metrics_md = _metrics_html(metrics['total_searches'], metrics['avg_latency'])
            table_data = [[l['time'], l['session'], l['query'], f"{l['latency']}s"] for l in logs]
            df = pd.DataFrame(table_data, columns=["搜索时间", "独立访客 ID (Session)", "用户提问内容", "响应耗时"])
            return metrics_md, df
    except Exception as e:
        logger.error(f"拉取 BI 大盘数据失败: {e}")
    return "<div class='card-hint' style='color:#c5221f;padding:12px'>⚠️ 数据大盘拉取失败，请检查管理后台 API 是否启动</div>", pd.DataFrame()


# ==========================================
# 🎨 前端 UI
# ==========================================

with gr.Blocks(title="Taday 管理控制台", fill_height=True) as demo:

    with gr.Column(elem_classes=["admin-shell"]):
        with gr.Row(elem_classes=["header-bar"]):
            gr.Markdown(
                "<div class='brand-wrap'>"
                "<div class='brand-icon'>T</div>"
                "<div><div class='title'>Taday Console</div>"
                "<div class='subtitle'>知识库资产 · 问答质检 · 运营监控</div></div>"
                "</div>"
            )
            gr.Markdown("<span class='badge'>管理后台</span>")

        with gr.Tabs(elem_classes=["admin-tabs"]):

            # ═══ Tab 1: 知识资产审核 ═══
            with gr.TabItem("📚 知识资产"):
                current_doc_id = gr.State("")
                raw_chunks_data = gr.State([])

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown("<div class='card-title'>📊 文档资产大盘</div>")
                    gr.Markdown("<p class='card-hint'>上传 PDF 财报后自动解析切片，点击表格行进入审核工作台</p>")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=260):
                            upload_file = gr.File(label="上传新财报 (PDF)", file_types=[".pdf"])
                            upload_status = gr.Textbox(label="上传反馈", interactive=False, value="等待上传…")
                        with gr.Column(scale=3):
                            with gr.Row(elem_classes=["toolbar"]):
                                gr.Markdown("<div class='toolbar-left'>👇 选择文档行查看切片详情</div>")
                                refresh_btn = gr.Button("刷新", size="sm", variant="secondary", scale=0)
                            doc_table = gr.Dataframe(
                                headers=["文档名称", "当前状态", "切片数量", "上传时间", "文档ID"],
                                interactive=False, wrap=True,
                            )

                with gr.Group(elem_classes=["card"]):
                    review_title = gr.Markdown("<div class='card-title'>🔍 切片审核工作台</div><p class='card-hint'>选中切片后可编辑原文，确认无误后发布入库</p>")

                    with gr.Row():
                        with gr.Column(scale=2):
                            chunk_table = gr.Dataframe(
                                headers=["序号", "切片ID", "状态", "内容预览", "完整内容"],
                                interactive=False, wrap=True,
                            )
                        with gr.Column(scale=1, elem_classes=["action-panel"]):
                            gr.Markdown("<div class='card-title'>✏️ 人工精修</div>")
                            edit_chunk_id = gr.Textbox(label="目标切片 ID", interactive=False)
                            edit_textarea = gr.Textbox(label="切片原文", lines=10, placeholder="点击左侧切片行后，原文将在此显示…")

                            with gr.Row():
                                save_btn = gr.Button("保存修改", variant="primary")
                                delete_btn = gr.Button("废弃段落", variant="stop")

                            edit_status = gr.Textbox(label="操作日志", interactive=False, value="等待操作…")

                            gr.Markdown("<div class='card-divider'></div>")
                            publish_btn = gr.Button("审核无误，发布入库", size="lg", variant="primary")

            # ═══ Tab 2: 问答质检 ═══
            with gr.TabItem("🎯 问答质检"):
                with gr.Group(elem_classes=["card"]):
                    gr.Markdown("<div class='card-title'>🚨 待质检草稿箱</div>")
                    gr.Markdown("<p class='card-hint'>先运行 AI 自动巡检过滤动态数据；AI 无法修复的会标记为「需人工审核」</p>")

                    with gr.Row():
                        refresh_bc_btn = gr.Button("刷新列表", size="sm", variant="secondary")
                        auto_heal_btn = gr.Button("启动 AI 自动巡检", size="sm", variant="primary")

                    bad_case_table = gr.Dataframe(
                        headers=["案例ID", "处理状态", "用户提问", "AI 翻车回答", "处理结果/修复答案", "发生时间"],
                        interactive=False, wrap=True,
                    )

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown("<div class='card-title'>📝 人工纠偏台</div><p class='card-hint'>选中案例后填写标准答案，或标记为误报忽略</p>")
                    with gr.Row():
                        selected_case_id = gr.Textbox(label="当前案例 ID", interactive=False, scale=1)
                        selected_query = gr.Textbox(label="用户原问题", interactive=False, scale=3, lines=3)

                    manual_correct_answer = gr.Textbox(
                        label="标准正确答案",
                        lines=5,
                        placeholder="人工核实后的标准答案…",
                    )

                    with gr.Row():
                        submit_fix_btn = gr.Button("确认修正并入库", variant="primary")
                        ignore_btn = gr.Button("误报作废", variant="stop")

                    bc_action_log = gr.Textbox(label="系统日志", interactive=False, value="等待操作…")

            # ═══ Tab 3: BI 大盘 ═══
            with gr.TabItem("📈 运营大盘"):
                with gr.Group(elem_classes=["card"]):
                    gr.Markdown("<div class='card-title'>🌐 检索流量监控</div>")
                    gr.Markdown("<p class='card-hint'>分析用户搜索行为与响应延迟，指导 RAG 知识库优化</p>")

                    with gr.Row(elem_classes=["toolbar"]):
                        gr.Markdown("<div class='toolbar-left'></div>")
                        refresh_bi_btn = gr.Button("刷新数据", size="sm", variant="secondary", scale=0)

                    bi_metrics_display = gr.HTML("<div class='card-hint'>数据加载中…</div>")
                    bi_log_table = gr.Dataframe(
                        headers=["搜索时间", "独立访客 ID (Session)", "用户提问内容", "响应耗时"],
                        interactive=False, wrap=True,
                    )

    # ==========================================
    # 事件绑定
    # ==========================================

    # 页面加载
    demo.load(fetch_dashboard, inputs=None, outputs=[doc_table])
    demo.load(fetch_bad_cases, inputs=None, outputs=[bad_case_table])
    demo.load(fetch_analytics, inputs=None, outputs=[bi_metrics_display, bi_log_table])

    # Tab 1: 知识资产
    refresh_btn.click(fetch_dashboard, inputs=None, outputs=[doc_table])
    upload_file.upload(handle_upload, inputs=[upload_file], outputs=[doc_table, upload_status])
    doc_table.select(on_select_doc, inputs=[doc_table], outputs=[current_doc_id, review_title, chunk_table, raw_chunks_data, edit_textarea])
    chunk_table.select(on_select_chunk, inputs=[chunk_table], outputs=[edit_chunk_id, edit_textarea])
    save_btn.click(save_chunk, inputs=[edit_chunk_id, edit_textarea, current_doc_id], outputs=[edit_status, chunk_table, raw_chunks_data])
    delete_btn.click(delete_chunk, inputs=[edit_chunk_id, current_doc_id], outputs=[edit_status, chunk_table, raw_chunks_data])
    publish_btn.click(publish_doc, inputs=[current_doc_id], outputs=[edit_status, doc_table])

    # Tab 2: 问答质检
    refresh_bc_btn.click(fetch_bad_cases, outputs=[bad_case_table])

    def on_select_case_for_qa(evt: gr.SelectData, df: pd.DataFrame):
        row_index = evt.index[0]
        case_id = df.iloc[row_index]["案例ID"]
        query = df.iloc[row_index]["用户提问"]
        fixed_ans = str(df.iloc[row_index]["处理结果/修复答案"])
        fill_ans = fixed_ans if "【LongCat" in fixed_ans or "自动纠偏" in fixed_ans else ""
        return case_id, query, fill_ans

    bad_case_table.select(
        on_select_case_for_qa,
        inputs=[bad_case_table],
        outputs=[selected_case_id, selected_query, manual_correct_answer]
    )

    def handle_submit_fix(case_id, correct_ans):
        if not case_id:
            return "⚠️ 请先选中一条待处理的案例", fetch_bad_cases()
        if not correct_ans.strip():
            return "⚠️ 请输入标准正确答案后再提交", fetch_bad_cases()
        try:
            resp = requests.post(
                f"{ADMIN_API_URL}/bad_cases/{case_id}/fix",
                json={"correct_answer": correct_ans},
                auth=ADMIN_AUTH, timeout=30,
            )
            if resp.status_code == 200:
                return f"✅ {resp.json()['message']}", fetch_bad_cases()
            return f"❌ 提交失败: {resp.text}", fetch_bad_cases()
        except Exception as e:
            return f"❌ 接口异常: {e}", fetch_bad_cases()

    submit_fix_btn.click(handle_submit_fix, inputs=[selected_case_id, manual_correct_answer], outputs=[bc_action_log, bad_case_table])

    # AI 一键巡检
    def handle_auto_heal(progress=gr.Progress()):
        progress(0, desc="🚀 正在唤醒 LongCat 大模型...")
        yield "⏳ 巡检指令已发送！大模型正在后台深度推演，请耐心等待 (约 10-60 秒)...", gr.update()

        try:
            progress(0.4, desc="🧠 大模型正在阅读报错日志与底层数据...")
            resp = requests.post(f"{ADMIN_API_URL}/bad_cases/auto_heal", timeout=300, auth=ADMIN_AUTH)
            progress(0.9, desc="💾 正在进行数据写入...")

            if resp.status_code == 200:
                data = resp.json()
                progress(1.0, desc="✅ 巡检圆满完成！")
                yield f"✅ {data['message']}", fetch_bad_cases()
            else:
                yield f"❌ 巡检失败: {resp.text}", fetch_bad_cases()
        except Exception as e:
            yield f"❌ 接口请求异常 (可能是大模型思考超时): {e}", fetch_bad_cases()

    auto_heal_btn.click(handle_auto_heal, inputs=[], outputs=[bc_action_log, bad_case_table])

    def handle_ignore(case_id):
        if not case_id:
            return "⚠️ 请先选中案例", fetch_bad_cases()
        requests.post(f"{ADMIN_API_URL}/bad_cases/{case_id}/ignore", auth=ADMIN_AUTH, timeout=30)
        return "🗑️ 已将该误报移入废弃站", fetch_bad_cases()

    ignore_btn.click(handle_ignore, inputs=[selected_case_id], outputs=[bc_action_log, bad_case_table])

    # Tab 3: BI
    refresh_bi_btn.click(fetch_analytics, inputs=None, outputs=[bi_metrics_display, bi_log_table])


if __name__ == "__main__":
    logger.info("知识库看板系统启动，端口 7861")
    import os
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASS", "Taday2026!")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        auth=(admin_user, admin_pass),
        css=CSS,
        theme=THEME,
        head=HEAD,
    )
