import json
import logging

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.core import agent
from app.tools.infra import ToolRun
from tests.test_agent_orchestration import FakeModel


class _FakeToolRun:
    def __init__(self, name, payload, cid="tc1"):
        self.name = name
        self.tool_call_id = cid
        class _M:
            content = json.dumps(payload, ensure_ascii=False)
        self.tool_message = _M()


def test_extract_faq_citations():
    runs = [_FakeToolRun("query_faq", {"sufficient": True, "evidence": "[1] x",
            "citations": [{"n": 1, "id": 5, "section_path": "运费政策", "question": "运费", "answer": "满99包邮", "content_type": "faq"}]})]
    faq = agent._faq_result(runs)
    assert faq["sufficient"] is True and faq["citations"][0]["id"] == 5


def test_extract_faq_none_when_absent():
    runs = [_FakeToolRun("query_order", {"x": 1})]
    assert agent._faq_result(runs) is None


# ---- 接线测试:mock repository + execute_tool_call,不碰真实 LLM/DB ----

_CITATIONS = [{"n": 1, "id": 5, "section_path": "运费政策", "question": "运费",
               "answer": "满99包邮", "content_type": "faq"}]
_SUFFICIENT = {"sufficient": True, "evidence": "[1] 运费: 满99包邮", "citations": _CITATIONS}
_INSUFFICIENT = {"sufficient": False, "source": "retrieval_low_conf",
                 "reason": "检索证据不足(top=0.100)", "citations": []}
_ORIGINAL = "喵喵机 Pro 那个邮费咋算啊?"


def _faq_tool_call():
    return {"name": "query_faq", "args": {"keyword": _ORIGINAL}, "id": "c1"}


def _faq_run(payload):
    return ToolRun("c1", "query_faq", True,
                   ToolMessage(content=json.dumps(payload, ensure_ascii=False),
                               tool_call_id="c1", name="query_faq"))


@pytest.fixture
def fake_repo(monkeypatch):
    """mock agent 用到的全部 repository 出口;insert_low_confidence 带捕获。"""
    state = {"appended": [], "inserts": []}

    async def create_conversation(user_id):
        return 42

    async def get_conversation(cid):
        return object()

    async def append_message(cid, role, content=None, tool_calls=None, tool_call_id=None):
        state["appended"].append((cid, role, content))

    async def list_messages(cid):
        return []

    async def insert_low_confidence(cid, raw_question, source, reason):
        state["inserts"].append((cid, raw_question, source, reason))
        return 1

    monkeypatch.setattr(agent.repository, "create_conversation", create_conversation)
    monkeypatch.setattr(agent.repository, "get_conversation", get_conversation)
    monkeypatch.setattr(agent.repository, "append_message", append_message)
    monkeypatch.setattr(agent.repository, "list_messages", list_messages)
    monkeypatch.setattr(agent.repository, "insert_low_confidence", insert_low_confidence)
    return state


def _patch_tool(monkeypatch, payload):
    async def fake_execute(tc, cid, **kw):
        return _faq_run(payload)
    monkeypatch.setattr(agent, "execute_tool_call", fake_execute)


@pytest.mark.asyncio
async def test_stream_sufficient_emits_citations_before_deltas(fake_repo, monkeypatch):
    _patch_tool(monkeypatch, _SUFFICIENT)
    model = FakeModel([AIMessage(content="", tool_calls=[_faq_tool_call()])],
                      stream_tokens=["满99元包邮", "[1]。"])
    events = [e async for e in agent.stream_agent_turn("u1", _ORIGINAL, None, model=model)]
    types = [e["type"] for e in events]
    assert types == ["tool", "citations", "delta", "delta", "done"]
    cit = next(e for e in events if e["type"] == "citations")
    assert cit["items"] == _CITATIONS
    assert fake_repo["inserts"] == []                    # 足证据不落池
    assert model.bind_calls == 1                         # 收敛不 bind


@pytest.mark.asyncio
async def test_stream_insufficient_pools_original_question(fake_repo, monkeypatch):
    _patch_tool(monkeypatch, _INSUFFICIENT)
    model = FakeModel([AIMessage(content="", tool_calls=[_faq_tool_call()])],
                      stream_tokens=["暂时没有查到相关信息", ",建议联系人工客服。"])
    events = [e async for e in agent.stream_agent_turn("u1", _ORIGINAL, None, model=model)]
    types = [e["type"] for e in events]
    assert "citations" not in types
    assert types == ["tool", "delta", "delta", "done"]   # 拒答生成照常收敛
    assert fake_repo["inserts"] == [
        (42, _ORIGINAL, "retrieval_low_conf", "检索证据不足(top=0.100)")]


@pytest.mark.asyncio
async def test_stream_pool_failure_still_completes(fake_repo, monkeypatch, caplog):
    _patch_tool(monkeypatch, _INSUFFICIENT)

    async def boom(cid, raw_question, source, reason):
        raise RuntimeError("mysql down")
    monkeypatch.setattr(agent.repository, "insert_low_confidence", boom)
    model = FakeModel([AIMessage(content="", tool_calls=[_faq_tool_call()])],
                      stream_tokens=["暂时没有查到相关信息。"])
    with caplog.at_level(logging.WARNING, logger="app.core.agent"):
        events = [e async for e in agent.stream_agent_turn("u1", _ORIGINAL, None, model=model)]
    types = [e["type"] for e in events]
    assert types == ["tool", "delta", "done"]            # 落池失败不 502,流照常完成
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.asyncio
async def test_nonstream_sufficient_populates_citations(fake_repo, monkeypatch):
    _patch_tool(monkeypatch, _SUFFICIENT)
    model = FakeModel([AIMessage(content="", tool_calls=[_faq_tool_call()]),
                       AIMessage(content="满99元包邮[1]。")])
    res = await agent.run_agent_turn("u1", _ORIGINAL, None, model=model)
    assert res.citations == _CITATIONS
    assert fake_repo["inserts"] == []


@pytest.mark.asyncio
async def test_nonstream_insufficient_pools_and_empty_citations(fake_repo, monkeypatch):
    _patch_tool(monkeypatch, _INSUFFICIENT)
    model = FakeModel([AIMessage(content="", tool_calls=[_faq_tool_call()]),
                       AIMessage(content="暂时没有查到相关信息,建议联系人工客服。")])
    res = await agent.run_agent_turn("u1", _ORIGINAL, None, model=model)
    assert res.citations == []
    assert fake_repo["inserts"] == [
        (42, _ORIGINAL, "retrieval_low_conf", "检索证据不足(top=0.100)")]


@pytest.mark.asyncio
async def test_nonstream_pool_failure_still_returns_answer(fake_repo, monkeypatch, caplog):
    _patch_tool(monkeypatch, _INSUFFICIENT)

    async def boom(cid, raw_question, source, reason):
        raise RuntimeError("mysql down")
    monkeypatch.setattr(agent.repository, "insert_low_confidence", boom)
    model = FakeModel([AIMessage(content="", tool_calls=[_faq_tool_call()]),
                       AIMessage(content="暂时没有查到相关信息。")])
    with caplog.at_level(logging.WARNING, logger="app.core.agent"):
        res = await agent.run_agent_turn("u1", _ORIGINAL, None, model=model)
    assert res.answer == "暂时没有查到相关信息。"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.asyncio
async def test_nonstream_no_faq_defaults_empty_citations(fake_repo, monkeypatch):
    model = FakeModel([AIMessage(content="你好,喵~")])
    res = await agent.run_agent_turn("u1", "你好", None, model=model)
    assert res.citations == []


@pytest.mark.asyncio
async def test_chat_sse_forwards_citations_event(fake_repo, monkeypatch):
    """/api/chat SSE:citations 事件在 delta 之前转发为 {"event":"citations","items":...}。"""
    from httpx import ASGITransport, AsyncClient

    from app.api import chat as chat_api
    from app.main import app

    _patch_tool(monkeypatch, _SUFFICIENT)
    model = FakeModel([AIMessage(content="", tool_calls=[_faq_tool_call()])],
                      stream_tokens=["满99元包邮[1]。"])
    app.dependency_overrides[chat_api.get_model] = lambda: model
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/chat", json={"user_id": "u1", "message": _ORIGINAL})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    payloads = [json.loads(ln[len("data: "):]) for ln in resp.text.splitlines()
                if ln.startswith("data: ") and ln != "data: [DONE]"]
    assert {"event": "citations", "items": _CITATIONS} in payloads
    cit_idx = next(i for i, p in enumerate(payloads) if p.get("event") == "citations")
    delta_idx = next(i for i, p in enumerate(payloads) if "delta" in p)
    assert cit_idx < delta_idx
