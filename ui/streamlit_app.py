import json
from typing import Any

import requests
import streamlit as st


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["x-api-key"] = api_key.strip()
    return headers


def _render_kv_list(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not items:
        st.caption("暂无")
        return
    for item in items:
        st.write(f"- {item}")


def _render_json_block(data: Any) -> None:
    st.json(data)


def _clean_source_snippet(text: Any, *, limit: int = 220) -> str:
    snippet = str(text or "").strip()
    snippet = snippet.replace("\r", " ").replace("\n", " ")
    snippet = " ".join(snippet.split())
    snippet = snippet.lstrip("#").strip()
    if len(snippet) > limit:
        return f"{snippet[: limit - 3].rstrip()}..."
    return snippet


def _render_http_error(status_code: int, data: dict[str, Any]) -> None:
    error = data.get("error", {}) if isinstance(data, dict) else {}
    error_code = error.get("code", "")
    if status_code == 409 or error_code == "SESSION_LOCKED":
        st.warning("当前 session_id 正在处理中，请等待当前请求完成，或更换新的 session_id。")
        st.caption("如果重复点击了按钮，同一个 session_id 的并发请求会被后端锁保护。")
    elif status_code == 503 or error_code == "RAG_INDEX_NOT_READY":
        st.warning("知识库索引尚未准备好，当前 RAG 模块暂时不可用。")
        st.caption("这不会影响模块一、二、三，你仍然可以继续演示原有聊天与结构化能力。")
    else:
        st.error(f"请求失败（HTTP {status_code}）")
    st.code(json.dumps(data, ensure_ascii=False, indent=2))


def call_stream_api(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    on_token,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/chat/stream"
    current_event = ""
    raw_lines: list[str] = []
    full_text = ""
    done_payload: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None

    try:
        with requests.post(
            url,
            json=payload,
            headers=_headers(api_key),
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                try:
                    return {"http_error": resp.json(), "status_code": resp.status_code}
                except ValueError:
                    return {"http_error": {"message": resp.text}, "status_code": resp.status_code}

            for line in resp.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                text_line = line.strip()
                if not text_line:
                    continue
                raw_lines.append(text_line)

                if text_line.startswith("event:"):
                    current_event = text_line.split("event:", 1)[1].strip()
                    continue
                if not text_line.startswith("data:"):
                    continue

                data_text = text_line.split("data:", 1)[1].strip()
                try:
                    payload_obj = json.loads(data_text)
                except json.JSONDecodeError:
                    payload_obj = {"raw": data_text}

                if current_event == "token":
                    token = str(payload_obj.get("token", ""))
                    full_text += token
                    on_token(full_text)
                elif current_event == "done":
                    done_payload = payload_obj
                elif current_event == "error":
                    error_payload = payload_obj
    except requests.RequestException as exc:
        return {"transport_error": str(exc)}

    return {
        "text": full_text,
        "done": done_payload,
        "error": error_payload,
        "raw_lines": raw_lines,
    }


def call_structured_api(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/v1/chat/structured"
    try:
        resp = requests.post(url, json=payload, headers=_headers(api_key), timeout=90)
    except requests.RequestException as exc:
        return 0, {"error": {"message": str(exc)}}

    try:
        data = resp.json()
    except ValueError:
        data = {"error": {"message": resp.text}}
    return resp.status_code, data


def call_rag_api(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/v1/rag/query"
    try:
        resp = requests.post(url, json=payload, headers=_headers(api_key), timeout=90)
    except requests.RequestException as exc:
        return 0, {"error": {"message": str(exc)}}

    try:
        data = resp.json()
    except ValueError:
        data = {"error": {"message": resp.text}}
    return resp.status_code, data


st.set_page_config(page_title="A+B 项目演示台", page_icon="🐾", layout="wide")
st.title("A+B 项目最小演示前端")
st.caption("单页分区：宠物对话、宠物今日状态卡片、行为观察与风险提示、宠物知识问答")

with st.sidebar:
    st.header("连接配置")
    base_url = st.text_input("base_url", value="http://127.0.0.1:8000")
    api_key = st.text_input("x-api-key", type="password")
    session_id = st.text_input("session_id", value="demo_user_001")
    model = st.text_input("model（可选）", value="")
    st.caption(f"当前目标：{base_url.rstrip('/')}")
    st.info("如果旧历史影响人设或输出，直接换一个新的 session_id 即可重新开始。")


st.subheader("模块一｜宠物对话（拟人陪伴）")
with st.form("stream_chat_form_v2"):
    stream_message = st.text_area(
        "输入消息",
        placeholder="例如：今天是不是有点困呀？下午怎么一直想趴在我旁边？",
    )
    submit_stream = st.form_submit_button("发送并实时展示")

if submit_stream:
    if not stream_message.strip():
        st.warning("请先输入消息。")
    else:
        stream_payload: dict[str, Any] = {
            "session_id": session_id,
            "user_message": stream_message.strip(),
        }
        if model.strip():
            stream_payload["model"] = model.strip()

        token_box = st.empty()
        token_box.markdown("**宠物回复：**")

        result = call_stream_api(
            base_url=base_url,
            api_key=api_key,
            payload=stream_payload,
            on_token=lambda text: token_box.markdown(f"**宠物回复：**\n\n{text}"),
        )

        if "transport_error" in result:
            st.error(f"请求失败：{result['transport_error']}")
        elif "http_error" in result:
            _render_http_error(result["status_code"], result["http_error"])
        else:
            if result.get("error"):
                err = result["error"]
                st.error(f"{err.get('code', 'ERROR')}: {err.get('message', 'unknown error')}")
            done = result.get("done") or {}
            request_id = done.get("request_id") or (result.get("error") or {}).get(
                "request_id",
                "",
            )
            with st.expander("流式调试信息", expanded=False):
                _render_json_block({"request_id": request_id})
                st.code("\n".join(result.get("raw_lines", [])) or "(empty)")

st.divider()

st.subheader("模块二｜宠物今日状态卡片")
with st.form("daily_card_form_v2"):
    daily_input = st.text_area(
        "今日观察",
        placeholder=(
            "例如：今天早上进食正常，午后睡得更久一些，傍晚主动来玩，"
            "喝水和猫砂使用都比较稳定。"
        ),
    )
    submit_daily = st.form_submit_button("生成每日卡片")

if submit_daily:
    if not daily_input.strip():
        st.warning("请先输入观察内容。")
    else:
        daily_payload: dict[str, Any] = {
            "session_id": session_id,
            "user_message": daily_input.strip(),
            "task_type": "daily_card",
        }
        if model.strip():
            daily_payload["model"] = model.strip()

        status_code, data = call_structured_api(
            base_url=base_url,
            api_key=api_key,
            payload=daily_payload,
        )
        if status_code != 200:
            _render_http_error(status_code, data)
        else:
            output = data.get("structured_output", {})
            with st.container(border=True):
                st.markdown("### 宠物今日状态卡片")
                st.write(f"**日期**：{output.get('date', '-')}")
                st.write(f"**摘要**：{output.get('summary', '-')}")
                st.write(f"**今日重点**：{output.get('focus', '-')}")
                _render_kv_list("优先事项", output.get("priorities", []))
                _render_kv_list("关注点", output.get("risks", []))
                _render_kv_list("行动建议", output.get("action_items", []))
                st.markdown("**回答文本**")
                st.write(data.get("answer", "-"))

            with st.expander("调试信息", expanded=False):
                _render_json_block({"request_id": data.get("request_id")})
                _render_json_block({"tool_calls": data.get("tool_calls", [])})
                st.code(json.dumps(data, ensure_ascii=False, indent=2))

st.divider()

st.subheader("模块三｜行为观察与风险提示")
with st.form("risk_analysis_form_v2"):
    risk_input = st.text_area(
        "行为观察描述",
        placeholder="例如：最近两天夜间活动增多，白天食欲略有下降。",
    )
    submit_risk = st.form_submit_button("生成观察提示")

if submit_risk:
    if not risk_input.strip():
        st.warning("请先输入观察内容。")
    else:
        risk_payload: dict[str, Any] = {
            "session_id": session_id,
            "user_message": risk_input.strip(),
            "task_type": "risk_analysis",
        }
        if model.strip():
            risk_payload["model"] = model.strip()

        status_code, data = call_structured_api(
            base_url=base_url,
            api_key=api_key,
            payload=risk_payload,
        )
        if status_code != 200:
            _render_http_error(status_code, data)
        else:
            output = data.get("structured_output", {})
            level = output.get("overall_level", "unknown")
            level_map = {"low": "低", "medium": "中", "high": "高"}

            with st.container(border=True):
                st.markdown("### 行为观察提示")
                st.write(f"**提示等级**：{level_map.get(level, level)}（仅供观察参考）")
                st.write(f"**整体说明**：{output.get('summary', '-')}")
                st.markdown("**关键观察点**")
                for idx, item in enumerate(output.get("key_risks", []), start=1):
                    st.write(f"{idx}. 现象：{item.get('risk', '-')}")
                    st.write(f"   影响：{item.get('impact', '-')}")
                    st.write(f"   可尝试：{item.get('mitigation', '-')}")
                _render_kv_list("温和建议", output.get("recommendations", []))
                st.markdown("**回答文本**")
                st.write(data.get("answer", "-"))

            st.info("本结果仅供参考，不替代专业诊断或医疗建议。")

            with st.expander("调试信息", expanded=False):
                _render_json_block({"request_id": data.get("request_id")})
                _render_json_block({"tool_calls": data.get("tool_calls", [])})
                st.code(json.dumps(data, ensure_ascii=False, indent=2))

st.divider()

st.subheader("模块四｜宠物知识问答（狗狗照护）")
st.caption("本模块基于知识库检索作答，不使用会话记忆。")
with st.form("rag_query_form_v1"):
    rag_question = st.text_area(
        "请描述你的问题",
        placeholder=(
            "例如：幼犬什么时候适合开始做社会化训练？\n"
            "或者：刚接回家的小狗前几天怎么安排作息比较稳妥？"
        ),
    )
    rag_top_k = st.slider("检索片段数量（top_k）", min_value=1, max_value=8, value=3)
    submit_rag = st.form_submit_button("查询知识库")

if submit_rag:
    if not rag_question.strip():
        st.warning("请先输入一个狗狗照护相关问题。")
    else:
        rag_payload = {
            "user_message": rag_question.strip(),
            "top_k": rag_top_k,
        }
        status_code, data = call_rag_api(
            base_url=base_url,
            api_key=api_key,
            payload=rag_payload,
        )
        if status_code != 200:
            _render_http_error(status_code, data)
        else:
            with st.container(border=True):
                st.markdown("### 知识库回答")
                st.write(data.get("answer", "-"))

            sources = data.get("sources", [])
            if sources:
                st.markdown("**来源片段**")
                for idx, source in enumerate(sources, start=1):
                    with st.container(border=True):
                        st.write(f"来源 {idx} | {source.get('title', '-')}")
                        st.caption(
                            f"score={source.get('score', '-')} | chunk_id={source.get('chunk_id', '-')}"
                        )
                        st.caption(f"source={source.get('source', '-')}")
                        st.text(_clean_source_snippet(source.get("snippet", "-")))

            with st.expander("原始响应", expanded=False):
                _render_json_block({"request_id": data.get("request_id")})
                _render_json_block({"sources": sources})
                st.code(json.dumps(data, ensure_ascii=False, indent=2))

st.stop()


st.set_page_config(page_title="A 项目演示台", page_icon="🐾", layout="wide")
st.title("A 项目最小演示前端（Streamlit）")
st.caption("单页分区：流式对话 + 每日状态卡片 + 行为观察与风险提示")

with st.sidebar:
    st.header("连接配置")
    base_url = st.text_input("base_url", value="http://127.0.0.1:8000")
    api_key = st.text_input("x-api-key", type="password")
    session_id = st.text_input("session_id", value="demo_user_001")
    model = st.text_input("model（可选）", value="")
    st.caption(f"当前目标：{base_url.rstrip('/')}")


st.subheader("模块一｜宠物对话（流式）")
with st.form("stream_chat_form"):
    stream_message = st.text_area("输入消息", placeholder="例如：今天它精神状态怎么样？")
    submit_stream = st.form_submit_button("发送并流式展示")

if submit_stream:
    if not stream_message.strip():
        st.warning("请先输入消息。")
    else:
        stream_payload: dict[str, Any] = {
            "session_id": session_id,
            "user_message": stream_message.strip(),
        }
        if model.strip():
            stream_payload["model"] = model.strip()

        token_box = st.empty()
        token_box.markdown("**AI 回复：**")

        result = call_stream_api(
            base_url=base_url,
            api_key=api_key,
            payload=stream_payload,
            on_token=lambda text: token_box.markdown(f"**AI 回复：**\n\n{text}"),
        )

        if "transport_error" in result:
            st.error(f"请求失败：{result['transport_error']}")
        elif "http_error" in result:
            st.error(f"HTTP {result['status_code']}：{json.dumps(result['http_error'], ensure_ascii=False)}")
        else:
            if result.get("error"):
                err = result["error"]
                st.error(f"{err.get('code', 'ERROR')}: {err.get('message', 'unknown error')}")
            done = result.get("done") or {}
            request_id = done.get("request_id") or (result.get("error") or {}).get("request_id", "")
            with st.expander("流式调试信息（request_id / 原始事件）", expanded=False):
                st.write({"request_id": request_id})
                st.code("\n".join(result.get("raw_lines", [])) or "(empty)")

st.divider()

st.subheader("模块二｜每日状态卡片")
with st.form("daily_card_form"):
    daily_input = st.text_area("今日观察", placeholder="例如：今天进食正常，午后比较安静，傍晚玩耍积极。")
    submit_daily = st.form_submit_button("生成每日卡片")

if submit_daily:
    if not daily_input.strip():
        st.warning("请先输入观察内容。")
    else:
        daily_payload: dict[str, Any] = {
            "session_id": session_id,
            "user_message": daily_input.strip(),
            "task_type": "daily_card",
        }
        if model.strip():
            daily_payload["model"] = model.strip()

        status_code, data = call_structured_api(
            base_url=base_url,
            api_key=api_key,
            payload=daily_payload,
        )
        if status_code != 200:
            st.error(f"请求失败（HTTP {status_code}）")
            st.code(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            output = data.get("structured_output", {})
            with st.container(border=True):
                st.markdown("### 今日状态卡片")
                st.write(f"**日期**：{output.get('date', '-')}")
                st.write(f"**摘要**：{output.get('summary', '-')}")
                st.write(f"**今日重点**：{output.get('focus', '-')}")
                _render_kv_list("优先事项", output.get("priorities", []))
                _render_kv_list("关注点", output.get("risks", []))
                _render_kv_list("行动建议", output.get("action_items", []))
                st.markdown("**回答文本**")
                st.write(data.get("answer", "-"))

            with st.expander("调试信息（tool_calls / request_id / 原始 JSON）", expanded=False):
                st.write("request_id:", data.get("request_id"))
                st.write("tool_calls:", data.get("tool_calls", []))
                st.code(json.dumps(data, ensure_ascii=False, indent=2))

st.divider()

st.subheader("模块三｜行为观察与风险提示")
with st.form("risk_analysis_form"):
    risk_input = st.text_area(
        "行为观察描述",
        placeholder="例如：最近两天夜间活动增多，白天食欲略有下降。",
    )
    submit_risk = st.form_submit_button("生成观察提示")

if submit_risk:
    if not risk_input.strip():
        st.warning("请先输入观察内容。")
    else:
        risk_payload: dict[str, Any] = {
            "session_id": session_id,
            "user_message": risk_input.strip(),
            "task_type": "risk_analysis",
        }
        if model.strip():
            risk_payload["model"] = model.strip()

        status_code, data = call_structured_api(
            base_url=base_url,
            api_key=api_key,
            payload=risk_payload,
        )
        if status_code != 200:
            st.error(f"请求失败（HTTP {status_code}）")
            st.code(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            output = data.get("structured_output", {})
            level = output.get("overall_level", "unknown")
            level_map = {"low": "低", "medium": "中", "high": "高"}

            with st.container(border=True):
                st.markdown("### 行为观察提示")
                st.write(f"**观察提示等级**：{level_map.get(level, level)}（仅作参考）")
                st.write(f"**整体说明**：{output.get('summary', '-')}")
                st.markdown("**关键观察点**")
                for idx, item in enumerate(output.get("key_risks", []), start=1):
                    st.write(f"{idx}. 现象：{item.get('risk', '-')}")
                    st.write(f"   影响：{item.get('impact', '-')}")
                    st.write(f"   可尝试：{item.get('mitigation', '-')}")
                _render_kv_list("温和建议", output.get("recommendations", []))
                st.markdown("**回答文本**")
                st.write(data.get("answer", "-"))

            st.info("本结果仅供参考，不替代专业诊断或医疗建议。")

            with st.expander("调试信息（tool_calls / request_id / 原始 JSON）", expanded=False):
                st.write("request_id:", data.get("request_id"))
                st.write("tool_calls:", data.get("tool_calls", []))
                st.code(json.dumps(data, ensure_ascii=False, indent=2))
