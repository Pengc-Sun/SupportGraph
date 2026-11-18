import pytest
from sqlalchemy import text
from sqlalchemy.exc import DataError

from app.db import repository

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_insert_low_confidence(db_session_factory, db_clean):
    cid = await repository.create_conversation("u1")
    lid = await repository.insert_low_confidence(cid, "邮费到底多少啊", "self_check", "证据不足")
    assert lid > 0
    factory = db_session_factory
    async with factory() as s:
        row = (await s.execute(text("SELECT raw_question, source, reason FROM low_confidence_questions WHERE id=:i"), {"i": lid})).first()
    assert row.source == "self_check" and row.raw_question == "邮费到底多少啊"


async def test_insert_low_confidence_null_conv(db_session_factory, db_clean):
    lid = await repository.insert_low_confidence(None, "无会话问题", "retrieval_low_conf", None)
    assert lid > 0


async def test_insert_low_confidence_rejects_invalid_source(db_session_factory, db_clean):
    # source 枚举只有三个合法值;非法值被 MySQL ENUM 列拒收(asyncmy 1265 Data truncated),
    # SQLAlchemy 包装为 DataError;失败事务不提交,db_clean 兜底清表。
    with pytest.raises(DataError):
        await repository.insert_low_confidence(None, "非法来源问题", "bogus", None)
