import json

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.api import chat as chat_api
from app.db import repository as repo
from app.main import app
from tests.test_agent_orchestration import FakeModel

pytestmark = pytest.mark.asyncio(loop_scope="session")

@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()

def make_client(model: FakeModel) -> AsyncClient:
    app.dependency_overrides[chat_api.get_model] = lambda: model
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

def parse_sse(body: str) -> list[tuple[str | None, str]]:
    """解析为 (event 名或 None, data 载荷) 对;event 行附着到其后的 data 帧。"""
    frames: list[tuple[str | None, str]] = []
    event = None
    for line in body.splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            frames.append((event, line[len("data: "):]))
            event = None
    return frames

async def test_chat_tool_path_streams_tool_delta_done(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["您的订单 ", "1001 正在运输中。"])
    async with make_client(model) as client:
        resp = await client.post(
            "/api/chat", json={"user_id": "u1", "message": "订单 1001 到哪了"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse(resp.text)
    payloads = [json.loads(p) for _, p in frames if p != "[DONE]"]
    assert {"event": "tool", "name": "query_logistics"} in payloads
    assert "".join(p["delta"] for p in payloads if "delta" in p) == "您的订单 1001 正在运输中。"
    done = payloads[-1]
    assert done["event"] == "done"
    assert isinstance(done["conversation_id"], int)
    assert frames[-1] == (None, "[DONE]")

async def test_chat_no_tool_path_delta_and_persistence(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好,喵~ 有什么可以帮您?")])
    async with make_client(model) as client:
        resp = await client.post("/api/chat", json={"user_id": "u1", "message": "你好"})
    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    payloads = [json.loads(p) for _, p in frames if p != "[DONE]"]
    assert not any(p.get("event") == "tool" for p in payloads)
    assert "".join(p["delta"] for p in payloads if "delta" in p) == "你好,喵~ 有什么可以帮您?"
    done = payloads[-1]
    assert done["event"] == "done"
    assert frames[-1] == (None, "[DONE]")
    msgs = await repo.list_messages(done["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].content == "你好"
    assert msgs[1].content == "你好,喵~ 有什么可以帮您?"

async def test_chat_unknown_conversation_error_frame_no_done(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="unused")])
    async with make_client(model) as client:
        resp = await client.post(
            "/api/chat",
            json={"user_id": "u1", "message": "hi", "conversation_id": 999999})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse(resp.text)
    assert frames[0][0] == "error"
    assert json.loads(frames[0][1]) == {"message": "会话不存在"}
    assert all(p != "[DONE]" for _, p in frames)

async def test_chat_validates_empty_message(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="unused")])
    async with make_client(model) as client:
        resp = await client.post("/api/chat", json={"user_id": "u1", "message": ""})
    assert resp.status_code == 422
