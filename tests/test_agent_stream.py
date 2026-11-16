import pytest
from langchain_core.messages import AIMessage

from app.core import agent
from app.db import repository as repo
from tests.test_agent_orchestration import FakeModel

pytestmark = pytest.mark.asyncio(loop_scope="session")

async def test_stream_with_tools_emits_tool_delta_done(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["您的订单", " 1001 ", "正在运输中。"])
    events = [e async for e in agent.stream_agent_turn("u1", "订单 1001 到哪了", None, model=model)]
    types = [e["type"] for e in events]
    deltas = [e for e in events if e["type"] == "delta"]
    # 事件序列:tool(query_logistics)→ 若干 delta → done
    assert types == ["tool"] + ["delta"] * len(deltas) + ["done"]
    assert len(deltas) >= 1
    assert events[0]["name"] == "query_logistics"
    assert "".join(d["text"] for d in deltas) == "您的订单 1001 正在运输中。"
    cid = events[-1]["conversation_id"]
    assert isinstance(cid, int)
    msgs = await repo.list_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[-1].content == "您的订单 1001 正在运输中。"
    assert model.bind_calls == 1                       # astream 收敛未 bind

async def test_stream_without_tools_emits_delta_done_only(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好,喵~ 有什么可以帮您?")])
    events = [e async for e in agent.stream_agent_turn("u1", "你好", None, model=model)]
    types = [e["type"] for e in events]
    assert "tool" not in types                          # 无 tool 事件
    assert types == ["delta", "done"]
    assert events[0]["text"] == "你好,喵~ 有什么可以帮您?"  # turn1 文本整块吐出
    assert model.astream_messages == []                 # 无工具不走 astream
    assert model.bind_calls == 1
    msgs = await repo.list_messages(events[-1]["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant"]

async def test_stream_new_conversation_returns_new_id(db_session_factory, db_clean):
    existing = await repo.create_conversation("u0")
    model = FakeModel([AIMessage(content="嗨~")])
    events = [e async for e in agent.stream_agent_turn("u1", "在吗", None, model=model)]
    cid = events[-1]["conversation_id"]
    assert events[-1]["type"] == "done"
    assert cid != existing                              # 回传的是新建 id
    conv = await repo.get_conversation(cid)
    assert conv is not None and conv.user_id == "u1"

async def test_stream_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    gen = agent.stream_agent_turn("u1", "hi", 999999, model=model)
    # async generator:异常在首次迭代时抛出
    with pytest.raises(agent.ConversationNotFound):
        await gen.__anext__()
