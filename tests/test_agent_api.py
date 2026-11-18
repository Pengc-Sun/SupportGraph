from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage

from app.core import agent
from app.core.agent import AgentResult
from app.main import app
from app.tools.infra import ToolRun

def _fake_result():
    tm = ToolMessage(content='{"status":"运输中"}', tool_call_id="c1", name="query_logistics")
    return AgentResult(
        conversation_id=12,
        answer="订单 1001 正在运输中。",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        tool_runs=[ToolRun("c1", "query_logistics", True, tm)],
        citations=[{"n": 1, "id": 5, "section_path": "运费政策", "question": "运费",
                    "answer": "满99包邮", "content_type": "faq"}],
    )

def test_agent_endpoint_returns_tool_trace(monkeypatch):
    async def fake_run(user_id, message, conversation_id, model=None):
        return _fake_result()
    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "订单 1001 到哪了"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 12
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["ok"] is True
    # Ruling 19-4:/api/agent 显式序列化 AgentResult 字段,citations 同步透出(与流式对齐)
    assert body["citations"][0]["id"] == 5

def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run(*a, **k):
        raise agent.ConversationNotFound(999)
    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999})
    assert r.status_code == 404

def test_agent_endpoint_422_on_missing_fields():
    client = TestClient(app)
    assert client.post("/api/agent", json={"message": "hi"}).status_code == 422
