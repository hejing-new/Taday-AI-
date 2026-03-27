from core.graph import app
from langchain_core.messages import HumanMessage

print("==================================================")
print("🚀 欢迎启动 Taday 的企业级多智能体金融分析系统 🚀")
print("==================================================")
print("输入 'quit' 退出系统。\n")

# 初始化一个空的对话记录字典
config = {"configurable": {"thread_id": "1"}} 

while True:
    user_input = input("\n👨‍💻 你: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("拜拜！系统已关闭。")
        break
    
    # 把用户的问题包装成标准的 Message 格式
    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    # 启动 LangGraph 引擎，开始流转！
    for output in app.stream(inputs, config=config, stream_mode="values"):
        # 实时打印最新的消息
        message = output["messages"][-1]
        
        # 过滤掉用户的输入和工具运行的中间代码，只打印 AI 说的最终人话
        if message.type == "ai" and not message.tool_calls:
            print(f"\n🤖 Agent: {message.content}\n")
            print("-" * 50)