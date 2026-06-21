"""
Taday 金融智能体 — 一键启动脚本

启动顺序：
  1. 初始化数据库（init_sql_db）
  2. 启动 FastAPI 对话后端 (8000)
  3. 启动 FastAPI 管理后台 API (8001)
  4. 启动 C 端 Gradio 对话界面 (7860)
  5. 启动 B 端 Gradio 管理后台 (7861)

使用方式：
    python start.py                # 启动全部服务
    python start.py --api-only     # 只启动两个 API 服务
    python start.py --gui-only     # 只启动两个 Gradio 界面
    python start.py --init-db      # 只初始化数据库，不启动服务
"""
import sys
import os
import time
import subprocess
import argparse

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def init_db():
    """初始化财务数据库"""
    print("=" * 50)
    print("📊 正在初始化财务数据库...")
    from init_sql_db import setup_financial_db
    setup_financial_db()
    print("✅ 数据库初始化完成")
    print("=" * 50)


def start_api():
    """启动 FastAPI 后端 API"""
    print("🚀 启动 FastAPI 对话后端 (端口 8000)...")
    p = subprocess.Popen(
        [sys.executable, "app_backend.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    print(f"   PID: {p.pid}")
    return p


def start_admin_api():
    """启动 FastAPI 管理后台 API"""
    print("🚀 启动 FastAPI 管理后台 API (端口 8001)...")
    p = subprocess.Popen(
        [sys.executable, "admin_backend.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    print(f"   PID: {p.pid}")
    return p


def start_chat_ui():
    """启动 C 端 Gradio 对话界面"""
    print("🚀 启动 C 端对话界面 (端口 7860)...")
    p = subprocess.Popen(
        [sys.executable, "app_frontend_network.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    print(f"   PID: {p.pid}")
    return p


def start_admin_ui():
    """启动 B 端 Gradio 管理后台"""
    print("🚀 启动 B 端管理后台 (端口 7861)...")
    p = subprocess.Popen(
        [sys.executable, "admin_frontend.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    print(f"   PID: {p.pid}")
    return p


def main():
    parser = argparse.ArgumentParser(description="Taday 一键启动")
    parser.add_argument("--api-only", action="store_true", help="只启动 API 服务")
    parser.add_argument("--gui-only", action="store_true", help="只启动 Gradio 界面")
    parser.add_argument("--init-db", action="store_true", help="只初始化数据库")
    args = parser.parse_args()

    print("=" * 50)
    print("🚀 Taday 企业级多智能体金融分析系统")
    print("=" * 50)

    # 初始化数据库
    init_db()
    if args.init_db:
        return

    processes = []

    if not args.gui_only:
        processes.append(("API-8000", start_api()))
        time.sleep(2)
        processes.append(("AdminAPI-8001", start_admin_api()))
        time.sleep(2)

    if not args.api_only:
        processes.append(("ChatUI-7860", start_chat_ui()))
        time.sleep(2)
        processes.append(("AdminUI-7861", start_admin_ui()))

    print()
    print("=" * 50)
    print("✅ 所有服务已启动！")
    print("=" * 50)
    print(f"  C 端对话界面: http://127.0.0.1:7860")
    print(f"  B 端管理后台: http://127.0.0.1:7861")
    print(f"  对话 API:     http://127.0.0.1:8000")
    print(f"  管理 API:     http://127.0.0.1:8001")
    print()
    print("按 Ctrl+C 停止所有服务")

    try:
        # 等待所有进程
        while True:
            time.sleep(1)
            for name, p in processes:
                if p.poll() is not None:
                    print(f"⚠️ {name} (PID {p.pid}) 已退出，退出码: {p.returncode}")
    except KeyboardInterrupt:
        print("\n\n🛑 正在停止所有服务...")
        for name, p in processes:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
            print(f"   已停止: {name}")
        print("✅ 所有服务已停止")


if __name__ == "__main__":
    main()
