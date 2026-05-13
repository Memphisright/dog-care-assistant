import requests

url = "http://127.0.0.1:8000/chat_with_memory"

headers = {
    "x-api-key": "test-key",
    "Content-Type": "application/json",
}

payload = {
    "session_id": "user_001",
    "user_message": "我刚才说了什么？",
}

with requests.post(url, headers=headers, json=payload, stream=True) as response:
    print("status_code:", response.status_code)
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            print(chunk, end="")