import gradio as gr
import requests
import pandas as pd
import time

# ==========================================
# 📡 后端 API 配置 (指向你刚启动的 8001 端口)
# ==========================================
ADMIN_API_URL = "http://127.0.0.1:8001/admin/api"

# 状态字典映射
STATUS_MAP = {
    "processing": "⏳ 解析切分中",
    "pending": "🟠 待人工审核",
    "published": "🟢 已发布入库",
    "failed": "❌ 解析失败"
}

# ==========================================
# 🧠 接口通信与业务逻辑
# ==========================================

def fetch_dashboard():
    """拉取全局文档看板数据"""
    try:
        resp = requests.get(f"{ADMIN_API_URL}/docs", timeout=10)
        if resp.status_code == 200:
            docs = resp.json()
            table_data = []
            for d in docs:
                status_icon = STATUS_MAP.get(d['status'], "❓ 未知")
                table_data.append([
                    d['filename'], 
                    status_icon, 
                    d['chunk_count'], 
                    d['created_at'], 
                    d['doc_id'] 
                ])
            df = pd.DataFrame(table_data, columns=["文档名称", "当前状态", "切片数量", "上传时间", "文档ID"])
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"看板获取失败: {e}")
        return pd.DataFrame()

def handle_upload(file):
    """异步上传：秒传文件，并立刻刷新看板看 processing 状态"""
    if file is None:
        return fetch_dashboard(), "⚠️ 请选择文件"
    
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name.split("/")[-1], f, "application/pdf")}
            resp = requests.post(f"{ADMIN_API_URL}/upload", files=files, timeout=60)
            
        if resp.status_code == 200:
            msg = resp.json()["message"]
            return fetch_dashboard(), f"✅ {msg}"
        else:
            return fetch_dashboard(), f"❌ 上传失败: {resp.text}"
    except Exception as e:
        return fetch_dashboard(), f"❌ 接口异常: {str(e)}"

def on_select_doc(evt: gr.SelectData, df: pd.DataFrame):
    """看板行点击事件：选中某个文档，去拉取它的切片"""
    row_index = evt.index[0]
    doc_id = df.iloc[row_index]["文档ID"]
    status = df.iloc[row_index]["当前状态"]
    filename = df.iloc[row_index]["文档名称"]
    
    if "解析切分中" in status:
        return doc_id, f"⚠️ 《{filename}》正在后台切片中，请稍后再试或点击刷新。", pd.DataFrame(), [], ""
    
    try:
        resp = requests.get(f"{ADMIN_API_URL}/docs/{doc_id}/chunks")
        if resp.status_code == 200:
            chunks = resp.json()
            table_data = []
            for c in chunks:
                preview = c['text_content'][:40].replace("\n", "") + "..."
                s_icon = "🟢 正常" if c['status'] == "active" else "🔴 废弃"
                table_data.append([c['chunk_index'], c['chunk_id'], s_icon, preview, c['text_content']])
            
            chunk_df = pd.DataFrame(table_data, columns=["序号", "切片ID", "状态", "内容预览", "完整内容"])
            return doc_id, f"✅ 正在审核：《{filename}》", chunk_df, chunks, ""
    except Exception as e:
        return doc_id, f"❌ 切片拉取失败: {e}", pd.DataFrame(), [], ""

def on_select_chunk(evt: gr.SelectData, df: pd.DataFrame):
    """切片行点击事件：加载内容到编辑框"""
    row_index = evt.index[0]
    chunk_id = df.iloc[row_index]["切片ID"]
    full_text = df.iloc[row_index]["完整内容"]
    return chunk_id, full_text

def save_chunk(chunk_id, new_text, doc_id):
    if not chunk_id: return "⚠️ 未选中切片", pd.DataFrame(), []
    requests.put(f"{ADMIN_API_URL}/chunks/{chunk_id}", json={"new_text": new_text})
    
    _, _, chunk_df, raw_chunks, _ = on_select_doc(gr.SelectData(target=None, index=[0,0], value=None), pd.DataFrame([{"文档ID": doc_id, "当前状态": "审核", "文档名称": ""}]))
    return f"✅ 切片 {chunk_id} 已保存", chunk_df, raw_chunks

def delete_chunk(chunk_id, doc_id):
    if not chunk_id: return "⚠️ 未选中切片", pd.DataFrame(), []
    requests.delete(f"{ADMIN_API_URL}/chunks/{chunk_id}")
    
    _, _, chunk_df, raw_chunks, _ = on_select_doc(gr.SelectData(target=None, index=[0,0], value=None), pd.DataFrame([{"文档ID": doc_id, "当前状态": "审核", "文档名称": ""}]))
    return f"🗑️ 切片 {chunk_id} 已废弃", chunk_df, raw_chunks

def publish_doc(doc_id):
    if not doc_id: return "⚠️ 请先在上方看板选中一个待审核的文档！", fetch_dashboard()
    resp = requests.post(f"{ADMIN_API_URL}/docs/{doc_id}/publish")
    if resp.status_code == 200:
        return f"🚀 {resp.json()['message']}", fetch_dashboard()
    return f"❌ 发布失败: {resp.text}", fetch_dashboard()

def fetch_bad_cases():
    """获取用户点踩的 Bad Case 列表"""
    try:
        resp = requests.get(f"{ADMIN_API_URL}/bad_cases", timeout=10)
        if resp.status_code == 200:
            cases = resp.json()
            table_data = []
            for c in cases:
                # 状态判断分支
                s = c.get('status', 'pending')
                if s == 'pending': s_icon = "🔴 待修复"
                elif s == 'manual_review': s_icon = "⚠️ 需人工审核"
                elif s == 'fixed': s_icon = "🟢 人工已修复"
                elif s == 'auto_fixed': s_icon = "🤖 AI已修复"
                elif s == 'ignored': s_icon = "⚪ 垃圾废弃"
                elif s == 'ignored_dynamic': s_icon = "⏳ 动态时效(已归档)"
                else: s_icon = "❓ 未知"

                # 提取结果或答案
                display_result = c.get('corrected_answer', c.get('admin_note', '-'))
                
                # 🌟 核心修复 1：严格将 6 列数据写入 table_data，删掉旧的冗余代码
                table_data.append([
                    c['case_id'], 
                    s_icon, 
                    c['user_query'], 
                    c['ai_response'], 
                    display_result, 
                    c['created_at']
                ])
                
            # 🌟 核心修复 2：确保列名与 Dataframe 组件的 headers 100% 对应
            return pd.DataFrame(table_data, columns=["案例ID", "处理状态", "用户提问", "AI 翻车回答", "处理结果/修复答案", "发生时间"])
    except Exception as e:
        # 🌟 核心修复 3：拒绝静默崩溃！把错误打在终端里！
        print(f"❌ 拉取草稿箱数据失败: {e}")
        
    return pd.DataFrame(columns=["案例ID", "处理状态", "用户提问", "AI 翻车回答", "处理结果/修复答案", "发生时间"])

def fetch_analytics():
    """获取 BI 大盘数据"""
    try:
        resp = requests.get(f"{ADMIN_API_URL}/analytics", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            metrics = data["metrics"]
            logs = data["logs"]
            
            # 构建顶部核心指标文本
            metrics_md = (
                f"### 📈 全站总提问次数: **{metrics['total_searches']}** 次 | "
                f"⏱️ 模型平均响应延迟: **{metrics['avg_latency']}** 秒"
            )
            
            # 构建表格数据
            table_data = [[l['time'], l['session'], l['query'], f"{l['latency']}s"] for l in logs]
            df = pd.DataFrame(table_data, columns=["搜索时间", "独立访客 ID (Session)", "用户提问内容", "响应耗时"])
            return metrics_md, df
    except:
        pass
    return "### ⚠️ 数据大盘拉取失败", pd.DataFrame()

# ==========================================
# 🎨 前端 UI 布局 (Dashboard + Tabs 架构)
# ==========================================
with gr.Blocks(title="知识库控制台", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🏢 Taday 知识库资产与质检控制台")
    
    with gr.Tabs():
        
        # ==========================================
        # 标签页 1：知识资产大盘 (你的原始代码重构入 Tab)
        # ==========================================
        with gr.TabItem("📚 知识资产审核与发布"):
            current_doc_id = gr.State("")
            raw_chunks_data = gr.State([])
            
            # ---------------- 模块 A：全局看板区 ----------------
            with gr.Group():
                gr.Markdown("### 📊 第一步：文档资产大盘 (Dashboard)")
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_file = gr.File(label="📥 拖拽上传新财报", file_types=[".pdf"])
                        upload_status = gr.Textbox(label="上传反馈", interactive=False)
                    with gr.Column(scale=3):
                        with gr.Row():
                            gr.Markdown("👇 **点击下方表格中的某一行**，即可进入该文档的切片审核模式。")
                            refresh_btn = gr.Button("🔄 手动刷新大盘", size="sm", scale=0)
                        
                        doc_table = gr.Dataframe(
                            headers=["文档名称", "当前状态", "切片数量", "上传时间", "文档ID"],
                            interactive=False,
                            wrap=True
                        )
            
            gr.Markdown("---")
            
            # ---------------- 模块 B：切片审核区 ----------------
            with gr.Group():
                review_title = gr.Markdown("### 🔍 第二步：切片审核工作台 (请先在上方选择文档)")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        chunk_table = gr.Dataframe(
                            headers=["序号", "切片ID", "状态", "内容预览", "完整内容"],
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### ✏️ 人工精修与发布")
                        edit_chunk_id = gr.Textbox(label="目标切片 ID", interactive=False)
                        edit_textarea = gr.Textbox(label="切片原文", lines=6)
                        
                        with gr.Row():
                            save_btn = gr.Button("💾 保存修改", variant="primary")
                            delete_btn = gr.Button("🗑️ 废弃段落", variant="stop")
                            
                        edit_status = gr.Textbox(label="操作日志", interactive=False)
                        
                        gr.Markdown("---")
                        publish_btn = gr.Button("✅ 审核无误，全量发布入库！", size="lg", variant="primary")
        
        # ==========================================
        # 标签页 2：RLHF 问答质检区 (真实业务流转版)
        # ==========================================
        with gr.TabItem("🎯 问答质检与人工纠偏"):
            gr.Markdown("### 🚨 待质检草稿箱 (人机协同流水线)")
            gr.Markdown("> 先点击【一键 AI 自动巡检】让大模型过滤动态数据并修复简单问题。AI 搞不定的会标记为【需人工审核】。")
            
            with gr.Row():
                refresh_bc_btn = gr.Button("🔄 刷新 JSON 列表", size="sm", variant="secondary")
                # 🌟 新增：一键启动大模型巡检按钮
                auto_heal_btn = gr.Button("🚀 启动一键 AI 自动巡检并修复", size="sm", variant="primary")
            
            # 🌟 表头加上“处理结果/修复答案”，并开启 wrap=True 防止文字被隐藏
            bad_case_table = gr.Dataframe(
                headers=["案例ID", "处理状态", "用户提问", "AI 翻车回答", "处理结果/修复答案", "发生时间"],
                interactive=False,
                wrap=True,
                scale=1
            )
            
            # 🌟 新增：人工质检与覆写工作台
            with gr.Group():
                gr.Markdown("#### 📝 人工质检纠偏台")
                with gr.Row():
                    selected_case_id = gr.Textbox(label="当前选中的案例", interactive=False, scale=1)
                    selected_query = gr.Textbox(label="用户原问题", interactive=False, scale=3, lines=4)
                
                # 运营人员在这里输入他们查到的标准正确答案
                manual_correct_answer = gr.Textbox(label="✍️ 在此输入正确的标准答案", lines=3, placeholder="人工核实后的标准答案...")
                
                with gr.Row():
                    submit_fix_btn = gr.Button("✅ 确认修正并存入数据库 (DB)", variant="primary")
                    # 🌟 新增：忽略按钮
                    ignore_btn = gr.Button("🗑️ 误报作废 (忽略)", variant="stop")
                
                bc_action_log = gr.Textbox(label="系统日志", interactive=False)

        
        # ==========================================
        # 标签页 3：BI 运营数据大盘 (新增)
        # ==========================================
        with gr.TabItem("📈 运营数据大盘 (BI)"):
            gr.Markdown("### 🌐 全局检索流量监控")
            gr.Markdown("> 实时监控用户的搜索行为，分析高频提问意图，优化底层 RAG 知识架构。")
            
            with gr.Row():
                refresh_bi_btn = gr.Button("🔄 刷新最新流量", size="sm", variant="secondary")
            
            # 核心指标看板
            bi_metrics_display = gr.Markdown("### 📈 数据加载中...")
            
            # 流量明细表格
            bi_log_table = gr.Dataframe(
                headers=["搜索时间", "独立访客 ID (Session)", "用户提问内容", "响应耗时"],
                interactive=False,
                wrap=True,
                # height=500
            )

    # ==========================================
    # ⚡ 事件绑定与生命周期
    # ==========================================
    # 0. 页面加载时自动拉取大盘和质检数据
    demo.load(fetch_dashboard, inputs=None, outputs=[doc_table])
    demo.load(fetch_bad_cases, inputs=None, outputs=[bad_case_table])
    
    refresh_btn.click(fetch_dashboard, inputs=None, outputs=[doc_table])
    refresh_bc_btn.click(fetch_bad_cases, inputs=None, outputs=[bad_case_table])

    # 数据看板事件
    demo.load(fetch_analytics, inputs=None, outputs=[bi_metrics_display, bi_log_table])
    refresh_bi_btn.click(fetch_analytics, inputs=None, outputs=[bi_metrics_display, bi_log_table])
    
    # 1. 上传文件
    upload_file.upload(
        handle_upload, inputs=[upload_file], outputs=[doc_table, upload_status]
    )
    
    # 2. 选中大盘文档 -> 加载切片
    doc_table.select(
        on_select_doc, 
        inputs=[doc_table], 
        outputs=[current_doc_id, review_title, chunk_table, raw_chunks_data, edit_textarea]
    )
    
    # 3. 选中切片 -> 加载编辑器
    chunk_table.select(
        on_select_chunk, inputs=[chunk_table], outputs=[edit_chunk_id, edit_textarea]
    )
    
    # 4. 修改与废弃
    save_btn.click(
        save_chunk, inputs=[edit_chunk_id, edit_textarea, current_doc_id], outputs=[edit_status, chunk_table, raw_chunks_data]
    )
    delete_btn.click(
        delete_chunk, inputs=[edit_chunk_id, current_doc_id], outputs=[edit_status, chunk_table, raw_chunks_data]
    )
    
    # 5. 发布入库
    publish_btn.click(
        publish_doc, inputs=[current_doc_id], outputs=[edit_status, doc_table]
    )

    # ----------------------------------------------------
    # 🚀 异步质检：选中与提交入库逻辑
    # ----------------------------------------------------
    # ----------------------------------------------------
    # 🚀 异步质检：选中与提交入库逻辑
    # ----------------------------------------------------
    # 1. 选中行，将原问题和AI修复的答案提取到操作台上
    def on_select_case_for_qa(evt: gr.SelectData, df: pd.DataFrame):
        row_index = evt.index[0]
        case_id = df.iloc[row_index]["案例ID"]
        query = df.iloc[row_index]["用户提问"]
        
        # 🌟 提取刚刚在表头新增的第5列数据
        fixed_ans = str(df.iloc[row_index]["处理结果/修复答案"])
        
        # 🌟 如果里面包含了AI修复的标志，就把答案填进框里让你审阅；否则就清空，让你完全手写
        fill_ans = fixed_ans if "【LongCat" in fixed_ans or "自动纠偏" in fixed_ans else ""
        
        return case_id, query, fill_ans

    bad_case_table.select(
        on_select_case_for_qa, 
        inputs=[bad_case_table], 
        # 🌟 核心：多加一个 output，把提取出来的答案塞进 manual_correct_answer 这个输入框里
        outputs=[selected_case_id, selected_query, manual_correct_answer] 
    )

    # 2. 提交人工答案，调用 8001 进行 DB 入库并更新 JSON
    def handle_submit_fix(case_id, correct_ans):
        if not case_id:
            return "⚠️ 请先选中一条待处理的案列", fetch_bad_cases()
        if not correct_ans.strip():
            return "⚠️ 请输入标准正确答案后再提交", fetch_bad_cases()
            
        try:
            resp = requests.post(
                f"{ADMIN_API_URL}/bad_cases/{case_id}/fix", 
                json={"correct_answer": correct_ans}
            )
            if resp.status_code == 200:
                return f"✅ {resp.json()['message']}", fetch_bad_cases()
            return f"❌ 提交失败: {resp.text}", fetch_bad_cases()
        except Exception as e:
            return f"❌ 接口异常: {e}", fetch_bad_cases()

    submit_fix_btn.click(
        handle_submit_fix,
        inputs=[selected_case_id, manual_correct_answer],
        outputs=[bc_action_log, bad_case_table]
    )
    
    # 刷新按钮绑定
    refresh_bc_btn.click(fetch_bad_cases, outputs=[bad_case_table])

    # ----------------------------------------------------
    # 🚀 AI 一键巡检逻辑 (带实时交互与进度反馈版)
    # ----------------------------------------------------
    # 引入 progress 对象来召唤 Gradio 的悬浮进度条
    def handle_auto_heal(progress=gr.Progress()):
        # ==========================================
        # 💡 交互第一步：立即响应！(点击瞬间执行)
        # ==========================================
        progress(0, desc="🚀 正在唤醒 LongCat 560B 大模型...")
        # 使用 yield 瞬间把日志框的文字改掉，告诉用户系统没有卡死
        yield "⏳ 巡检指令已发送！大模型（老中医+修复师）正在后台进行高强度深度推演，请耐心等待 (约需 10-60 秒)...", gr.update()

        try:
            # ==========================================
            # 💡 交互第二步：状态流转 (安抚用户情绪)
            # ==========================================
            progress(0.4, desc="🧠 大模型正在疯狂阅读报错日志与底层数据...")
            
            # 🌟 核心修复：把 timeout 从 60 改成 300（5分钟），给足大模型批量思考的时间！
            resp = requests.post(f"{ADMIN_API_URL}/bad_cases/auto_heal", timeout=300)
            
            progress(0.9, desc="💾 正在进行物理隔离与数据写入...")

            # ==========================================
            # 💡 交互第三步：最终裁决 (刷新大盘)
            # ==========================================
            if resp.status_code == 200:
                data = resp.json()
                progress(1.0, desc="✅ 巡检圆满完成！")
                # 再次 yield，用最终结果覆盖掉之前的提示语，并刷新表格数据
                yield f"✅ {data['message']}", fetch_bad_cases()
            else:
                yield f"❌ 巡检失败: {resp.text}", fetch_bad_cases()
                
        except Exception as e:
            yield f"❌ 接口请求异常 (可能是大模型思考超时): {e}", fetch_bad_cases()

    # 绑定事件保持不变
    auto_heal_btn.click(
        handle_auto_heal,
        inputs=[],
        outputs=[bc_action_log, bad_case_table]
    )

    def handle_ignore(case_id):
        if not case_id: return "⚠️ 请先选中案例", fetch_bad_cases()
        # 调一个后端接口把它标为废弃 (后端只需写一个很简单的接口更新 status 即可)
        requests.post(f"{ADMIN_API_URL}/bad_cases/{case_id}/ignore")
        return "🗑️ 已将该误报移入废弃站", fetch_bad_cases()

    ignore_btn.click(handle_ignore, inputs=[selected_case_id], outputs=[bc_action_log, bad_case_table])

if __name__ == "__main__":
    print("🚀 知识库看板系统启动！带 Basic Auth 安全防护。")
    # 预设管理员账号：admin / Taday2026!
    demo.launch(server_name="127.0.0.1", server_port=7861, auth=("admin", "Taday2026!"))