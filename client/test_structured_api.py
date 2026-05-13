import requests
import json

url = "http://127.0.0.1:8000/api/v1/chat/structured"

headers = {
    "x-api-key": "test-key",
    "Content-Type": "application/json",
}

payload = {
    "session_id": "user_001",
    "user_message": "它这两天食欲下降，而且不太愿意动",
    "task_type": "risk_analysis",
}

response = requests.post(url, headers=headers, json=payload)

print("status_code:", response.status_code)
print(json.dumps(response.json(), ensure_ascii=False, indent=2))