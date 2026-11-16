import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from app.core import agent
from app.core.prompts import AGENT_SYSTEM
from app.db import repository as repo

pytestmark = pytest.mark.asyncio(loop_scope="session")

class FakeModel:
    """turn1 返回 scripted[0](可带 tool_calls);非流式收敛返回 scripted[1];
    流式收敛按 stream_tokens 逐个 yield AIMessageChunk。
    bind_calls 记录 bind_tools 次数(收敛不 bind → 应恒为 1)。"""
    def __init__(self, scripted, stream_tokens=None):
        self._scripted = list(scripted)
        self._stream_tokens = list(stream_tokens or [])
        self.bind_calls = 0
        self.invoke_messages = []
        self.astream_messages = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        self.invoke_messages.append(list(messages))
        return self._scripted.pop(0)

    async def astream(self, messages):
        self.astream_messages.append(list(messages))
        for t in self._stream_tokens:
            yield AIMessageChunk(content=t)

async def test_no_tool_calls_returns_direct_answer(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好,喵~ 有什么可以帮您?")])
    res = await agent.run_agent_turn("u1", "你好", None, model=model)
    assert res.answer.startswith("你好") and res.tool_calls == []
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"]

async def test_tool_call_flow_executes_and_converges(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first, AIMessage(content="您的订单 1001 正在运输中。")])
    res = await agent.run_agent_turn("u1", "订单 1001 到哪了", None, model=model)
    assert res.answer == "您的订单 1001 正在运输中。"
    assert res.tool_calls[0]["name"] == "query_logistics"
    assert res.tool_runs[0].ok is True
    assert model.bind_calls == 1                       # 收敛不 bind
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[2].tool_call_id == "c1"

async def test_continue_conversation_replays_only_final_answers(db_session_factory, db_clean):
    m1 = FakeModel([
        AIMessage(content="我帮您查一下:", tool_calls=[
            {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]),
        AIMessage(content="订单 1001 已揽件。")])
    cid = (await agent.run_agent_turn("u1", "订单1001到哪了", None, model=m1)).conversation_id
    m2 = FakeModel([AIMessage(content="还有什么可以帮您?")])
    await agent.run_agent_turn("u1", "谢谢", cid, model=m2)
    ai_contents = [m.content for m in m2.invoke_messages[0] if isinstance(m, AIMessage)]
    assert "订单 1001 已揽件。" in ai_contents            # 最终答回放
    assert "我帮您查一下:" not in ai_contents             # tool-calling preamble 不跨轮

async def test_turn1_messages_start_with_agent_system(db_session_factory, db_clean):
    """spec:context = [AGENT_SYSTEM] + trimmed history。
    trim_messages 默认 include_system=False,须先裁剪再前置 system。"""
    model = FakeModel([AIMessage(content="你好,喵~")])
    await agent.run_agent_turn("u1", "你好", None, model=model)
    msgs = model.invoke_messages[0]
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == AGENT_SYSTEM
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content == "你好"

async def test_system_prompt_kept_when_history_over_budget(db_session_factory, db_clean):
    """超预算裁剪后 system prompt 仍必须在场(不占历史 token 预算,同 ch01 姿态)。"""
    cid = await repo.create_conversation("u1")
    long_text = "长历史内容。" * 200                     # 单条约 1200 字符,两条即超 2000 token 预算
    for i in range(4):
        await repo.append_message(cid, "user", content=f"{long_text}{i}")
        await repo.append_message(cid, "assistant", content=f"{long_text}{i}")
    model = FakeModel([AIMessage(content="好的")])
    await agent.run_agent_turn("u1", "当前问题", cid, model=model)
    msgs = model.invoke_messages[0]
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == AGENT_SYSTEM
    assert isinstance(msgs[-1], HumanMessage)
    assert msgs[-1].content == "当前问题"
    assert len(msgs) < 10                                # 历史确实被裁剪(全量回放为 9 条 + system)

async def test_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    try:
        await agent.run_agent_turn("u1", "hi", 999999, model=model)
        assert False, "应抛 ConversationNotFound"
    except agent.ConversationNotFound:
        pass
