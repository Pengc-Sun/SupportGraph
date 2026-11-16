import pytest

from app.db.models import Conversation, Ticket
from app.tools.business import create_ticket

pytestmark = pytest.mark.asyncio(loop_scope="session")

async def test_conversation_id_hidden_from_model_schema():
    # InjectedToolArg 的参数不应出现在暴露给 LLM 的 args schema 里。
    # 实测 langchain-core 1.8.1:get_input_schema()(完整执行面)仍含 injected 参,
    # 模型可见面是 tool_call_schema(bind_tools/convert_to_openai_tool 用它),故断言后者。
    model_props = create_ticket.tool_call_schema.model_json_schema().get("properties", {})
    assert "description" in model_props and "ticket_type" in model_props
    assert "conversation_id" not in model_props  # 关键:模型看不到会话主键
    assert "conversation_id" not in create_ticket.args
    # 执行面仍接受 injected 参(系统注入后可 ainvoke 传入)
    full_props = create_ticket.get_input_schema().model_json_schema().get("properties", {})
    assert "conversation_id" in full_props

async def test_create_ticket_injects_conversation_id_and_writes(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id
    r = await create_ticket.ainvoke(
        {"description": "商品损坏要退货", "ticket_type": "售后", "conversation_id": cid}
    )
    assert r["ticket_no"].startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, r["ticket_no"])
        assert t is not None and t.conversation_id == cid
