"""CLI:清空知识库——MySQL knowledge_chunks + qa_extraction_staging + Milvus knowledge 集合。
打印清理前后计数。运行:PYTHONPATH=. uv run python scripts/reset_kb.py"""
import asyncio

from sqlalchemy import text

import app.db.base as db
from app.kb import milvus_client


async def _counts(s) -> tuple[int, int]:
    chunks = (await s.execute(text("SELECT COUNT(*) FROM knowledge_chunks"))).scalar_one()
    staging = (await s.execute(text("SELECT COUNT(*) FROM qa_extraction_staging"))).scalar_one()
    return int(chunks), int(staging)


async def main() -> None:
    async with db.async_session() as s:
        before = await _counts(s)
        # knowledge_chunks 自引用 FK(prev/next):关外键检查后 DELETE,先暂存表后知识块
        await s.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        await s.execute(text("DELETE FROM qa_extraction_staging"))
        await s.execute(text("DELETE FROM knowledge_chunks"))
        await s.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        await s.commit()
        after = await _counts(s)
    print(f"MySQL:knowledge_chunks {before[0]} → {after[0]},"
          f"qa_extraction_staging {before[1]} → {after[1]}")
    try:
        client = milvus_client.get_client()
        if client.has_collection(milvus_client.COLLECTION):
            n = milvus_client.count(client)
            client.drop_collection(milvus_client.COLLECTION)
            print(f"Milvus:已删除集合 {milvus_client.COLLECTION}(原 {n} 条)")
        else:
            print(f"Milvus:集合 {milvus_client.COLLECTION} 不存在,无需删除")
    except Exception as exc:  # Milvus 离线不挡 MySQL 清理;集合残留可重跑本脚本补删
        print(f"⚠️ Milvus 清理失败(可重跑):{exc}")


if __name__ == "__main__":
    asyncio.run(main())
