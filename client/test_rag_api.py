import json

import requests


url = "http://127.0.0.1:8000/api/v1/rag/query"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "test-key",
}
payload = {
    "user_message": "幼犬什么时候适合开始做社会化训练？",
    "top_k": 3,
}

resp = requests.post(url, headers=headers, json=payload, timeout=30)
print(resp.status_code)
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
