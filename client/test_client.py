import json

import requests

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "dev-key"


def stream_chat(session_id: str, user_message: str) -> None:
    url = f"{BASE_URL}/api/v1/chat/stream"
    payload = {
        "session_id": session_id,
        "user_message": user_message,
    }
    headers = {"x-api-key": API_KEY}

    print(f"\nUser: {user_message}")
    print("Assistant: ", end="", flush=True)
    event_name = None

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event: "):
                event_name = line.split("event: ", 1)[1].strip()
                continue
            if not line.startswith("data: "):
                continue

            raw = line.split("data: ", 1)[1]
            data = json.loads(raw)

            if event_name == "token":
                print(data["token"], end="", flush=True)
            elif event_name == "error":
                print(f"\n[ERROR] {data['code']}: {data['message']}")

    print("\n" + "-" * 40)


if __name__ == "__main__":
    stream_chat("user_999", "你好，我叫王大锤，我最喜欢吃红烧肉。")
    stream_chat("user_999", "我是谁？我最喜欢吃什么？")

