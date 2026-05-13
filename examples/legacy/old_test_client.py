import requests

url = "http://127.0.0.1:8000/chat_with_memory"


def send_message(session_id, msg):
    print(f"\n用户：{msg}")
    print("AI回复：", end="", flush=True)
    payload = {"session_id": session_id, "user_message": msg}

    with requests.post(url, json=payload, stream=True) as response:
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
    print("\n" + "-" * 30)


# 第一轮：告诉它你的身份
send_message("user_999", "你好，我叫王大锤，我最喜欢吃红烧肉。")

# 第二轮：测试它到底记没记住！
send_message("user_999", "我是谁？我最喜欢吃什么？")

# 新版测试↑

# 旧版流式请求，测试脚本

# import requests  # 如果没有，在终端 pip install requests
# import json
#
# url = "http://127.0.0.1:8001/chat_stream"
# payload = {"user_message": "给我讲个程序员修电脑的笑话，长一点"}
#
# print("用户：", payload["user_message"])
# print("AI猫咪回复：", end="", flush=True)
#
# # 核心：使用 stream=True 开启流式接收
# with requests.post(url, json=payload, stream=True) as response:
#     # 站在流水线的末端，源源不断地接收字块
#     for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
#         if chunk:
#             # 收到一个字，就立刻打印在屏幕上，并且不换行（end=""）
#             print(chunk, end="", flush=True)
#
# print("\n[接收完毕]")