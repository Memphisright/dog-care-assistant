import requests

url = "http://127.0.0.1:8000/api/v1/chat/stream"

headers = {
    "x-api-key": "test-key",
    "Content-Type": "application/json",
}

payload = {
    "session_id": "user_001",
    "user_message": "你好，介绍一下你自己",
}

with requests.post(url, headers=headers, json=payload, stream=True) as response:
    print("status_code:", response.status_code)
    print("headers:", response.headers)
    print("-" * 40)

    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            print(chunk, end="")