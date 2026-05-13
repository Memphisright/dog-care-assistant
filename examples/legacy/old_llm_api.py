import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import redis.asyncio as redis  # 🌟 导入异步版的 redis

app = FastAPI()

client = AsyncOpenAI(
    api_key="",  # 填入你的真实 API Key
    base_url="https://api.deepseek.com"
)

# 🌟 1. 建立与 Docker 中 Redis 的连接
# decode_responses=True 意味着取出来的数据直接是字符串，不用再手动 decode 解码
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)


# 🌟 2. 升级防弹衣：新增 session_id 字段来区分不同用户
class ChatRequest(BaseModel):
    session_id: str  # 必须传：比如 "user_001"
    user_message: str  # 用户说的话


# 核心流水线
async def get_llm_stream(session_id: str, user_message: str):
    # 🌟 3. 去 Redis 里捞取历史记忆
    history_key = f"chat_history:{session_id}"
    history_str = await redis_client.get(history_key)

    if history_str:
        # 如果有记忆，把字符串反序列化为 Python 的列表
        messages = json.loads(history_str)
    else:
        # 如果是新用户第一次来，初始化人设
        messages = [{"role": "system", "content": "你是一只聪明且拥有记忆的AI猫咪。"}]

    # 把用户最新的话追加进去
    messages.append({"role": "user", "content": user_message})

    # 调大模型接口
    response_stream = await client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    # 🌟 4. 准备一个空水桶，把流式吐出来的字收集起来，为了最后存入 Redis
    full_ai_response = ""

    async for chunk in response_stream:
        text_chunk = chunk.choices[0].delta.content
        if text_chunk is not None:
            full_ai_response += text_chunk  # 把字接进水桶
            yield text_chunk  # 同时把字推给前端

    # 🌟 5. 流水线走完后，把 AI 的完整回答也加入记忆列表
    messages.append({"role": "assistant", "content": full_ai_response})

    # 🌟 6. 把更新后的记忆存回 Redis
    # setex 的意思是 Set with Expiration (带过期时间的存入)
    # 3600 代表 3600秒(1小时)后自动销毁记忆，防止 Redis 内存爆炸
    await redis_client.setex(history_key, 3600, json.dumps(messages, ensure_ascii=False))


@app.post("/chat_with_memory")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        get_llm_stream(request.session_id, request.user_message),
        media_type="text/event-stream"
    )


#新版给模型加入记忆↑

# 旧版学习流式非流式输出↓

# from fastapi import FastAPI
# from pydantic import BaseModel
# from openai import AsyncOpenAI  # 注意：我们导入的是 AsyncOpenAI，为了配合 FastAPI 的异步！
# from fastapi.responses import StreamingResponse  # ！！！新增导入这个流式响应模块！！！
#
# app = FastAPI()
#
# # ==========================================
# # 1. 初始化大模型客户端 (这里以 DeepSeek 为例，如果你用其他模型，改 base_url 和 api_key 即可)
# # ==========================================
# client = AsyncOpenAI(
#     api_key="",  # ！！！把这里换成你刚才申请的 API Key！！！
#     base_url="https://api.deepseek.com"  # 这是 DeepSeek 的官方接口地址
# )
#
#
# # ==========================================
# # 2. 定义 Pydantic 防弹衣：规定前端必须传什么格式过来
# # ==========================================
# class ChatRequest(BaseModel):
#     user_message: str
#
#
# # ==========================================
# # 3. 核心大模型对话接口
# # ==========================================
# @app.post("/chat")
# async def chat_with_pet(request: ChatRequest):
#     print(f"收到用户提问: {request.user_message}")
#
#     # 构造给大模型的 Prompt (系统提示词，这里是为你量身定制的宠物 Agent 设定)
#     messages = [
#         {"role": "system",
#          "content": "你现在是一只傲娇的小橘猫，名字叫'小胖'。回答问题时要带点猫咪的语气，比如经常加'喵~'。"},
#         {"role": "user", "content": request.user_message}
#     ]
#
#     try:
#         # ==========================================
#         # ！！！这是全场最核心的一行代码！！！
#         # 我们用 await 挂起这个网络请求。在这几秒钟里，FastAPI 绝不会阻塞！
#         # ==========================================
#         response = await client.chat.completions.create(
#             model="deepseek-chat",  # 指定使用的模型名称
#             messages=messages,
#             temperature=0.7  # 控制猫咪发散思维的程度 (0-2之间)
#         )
#
#         # 剥开大模型返回的层层 JSON 外衣，提取真正的那句话
#         ai_reply = response.choices[0].message.content
#
#         return {
#             "status": "success",
#             "reply": ai_reply
#         }
#
#     except Exception as e:
#         # 万一网络断了或者 API 欠费了，优雅地返回错误，而不是让程序崩溃
#         return {"status": "error", "message": str(e)}
#
# # 新增流式响应
# # ==========================================
# # 核心机制：定义一个“流水线”（异步生成器）
# # 这个函数不再使用 return 返回完整结果，而是使用 yield “挤牙膏”
# # ==========================================
# async def get_llm_stream(user_message: str):
#     messages = [
#         {"role": "system", "content": "你是一个幽默的脱口秀演员。请用中文回答。"},
#         {"role": "user", "content": user_message}
#     ]
#
#     # 1. 发起请求，注意这里的终极魔法开关：stream=True
#     response_stream = await client.chat.completions.create(
#         model="deepseek-chat",
#         messages=messages,
#         temperature=0.7,
#         stream=True  # ！！！开启流式输出！！！
#     )
#
#     # 2. 站在流水线末端，接大模型吐出来的字
#     # 因为大模型是异步一段一段吐数据的，所以我们用 async for 来循环接收
#     async for chunk in response_stream:
#         # 剥开每一小块 (chunk) 的外衣，提取那个字 (delta)
#         # 注意：这里不再是 .message.content，而是 .delta.content
#         text_chunk = chunk.choices[0].delta.content
#
#         if text_chunk is not None:
#             # ！！！全场最核心的关键字：yield ！！！
#             # return 是“老板发月薪，一次性结清，然后下班”
#             # yield 是“包工头发现日结，干一点活就发一点钱，程序不结束，继续循环”
#             yield text_chunk
#
#
# # ==========================================
# # FastAPI 接口：接入前端
# # ==========================================
# @app.post("/chat_stream")
# async def chat_stream_endpoint(request: ChatRequest):
#     print(f"收到流式请求: {request.user_message}")
#
#     # ！！！注意看这里，我们不再返回一个普通的字典 { "reply": ai_reply } ！！！
#     # 而是用 FastAPI 专门的 StreamingResponse 把我们的流水线包装起来。
#     # media_type="text/event-stream" 是告诉前端（微信小程序）：准备好，我要像挤牙膏一样给你发数据了！
#     return StreamingResponse(
#         get_llm_stream(request.user_message),
#         media_type="text/event-stream"
#     )