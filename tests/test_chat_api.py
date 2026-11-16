import json

from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel

from app.api import chat as chat_api
from app.main import app

def make_client(responses: list[str]) -> TestClient:
    fake = FakeListChatModel(responses=responses)
    app.dependency_overrides[chat_api.get_model] = lambda: fake
    return TestClient(app)

def collect_sse(resp) -> tuple[list[str], str]:
    """返回 (delta 列表, 终止帧)。"""
    deltas, last = [], ""
    for line in resp.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            last = payload
        else:
            deltas.append(json.loads(payload)["delta"])
    return deltas, last

def teardown_function():
    app.dependency_overrides.clear()
    chat_api.store._sessions.clear()

def test_chat_streams_tokens_and_done():
    client = make_client(["你好呀"])
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s1", "message": "在吗"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        deltas, last = collect_sse(resp)
    assert "".join(deltas) == "你好呀"
    assert len(deltas) > 1  # 逐 token,不是一次性整段
    assert last == "[DONE]"

def test_chat_persists_history_for_next_turn():
    client = make_client(["第一轮答", "第二轮答"])
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s2", "message": "第一问"}
    ) as r:
        collect_sse(r)
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s2", "message": "第二问"}
    ) as r:
        collect_sse(r)
    history = chat_api.store.get("s2")
    assert [m.content for m in history] == ["第一问", "第一轮答", "第二问", "第二轮答"]

def test_chat_upstream_error_emits_error_frame_without_done_or_history():
    """astream 抛异常时:出现 event: error 帧 + 错误 JSON,无 [DONE],历史不落该轮。"""

    class ExplodingChatModel(FakeListChatModel):
        async def astream(self, *args, **kwargs):
            raise RuntimeError("upstream stream down")
            yield  # pragma: no cover - 使本方法成为异步生成器

    app.dependency_overrides[chat_api.get_model] = lambda: ExplodingChatModel(
        responses=["unused"]
    )
    client = TestClient(app)
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s-err", "message": "在吗"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = list(resp.iter_lines())
    assert "event: error" in lines
    payloads = [
        json.loads(line[len("data: "):]) for line in lines if line.startswith("data: ")
    ]
    assert payloads == [{"message": "上游模型暂时不可用,请稍后重试"}]
    assert "data: [DONE]" not in lines
    assert chat_api.store.get("s-err") == []

def test_chat_validates_empty_message():
    client = make_client(["x"])
    assert (
        client.post("/api/chat", json={"session_id": "s3", "message": ""}).status_code
        == 422
    )
