from openai import OpenAI

# 创建 API 客户端
client = OpenAI(api_key="sk-859a43e9221143708edcd3cf2dc00787", base_url="https://api.deepseek.com")

def deepseek(messages):
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=False
    )
    reply = response.choices[0].message.content
    print(reply + '\n')
    return reply  # 返回助手回复以维护上下文

# 初始化消息列表（仅包含系统消息）
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    print("请输入你的问题（输入 'q' 退出）:")
    usr_question = input().strip()
    
    if usr_question.lower() == 'q':
        break
    
    # 创建新的用户消息字典（避免引用污染）
    user_msg = {"role": "user", "content": usr_question}  # 直接使用输入内容
    
    # 添加用户消息到列表
    messages.append(user_msg)
    
    # 调用API并获取助手回复
    assistant_reply = deepseek(messages)
    
    # 添加助手回复到消息列表以维护上下文
    messages.append({"role": "assistant", "content": assistant_reply})