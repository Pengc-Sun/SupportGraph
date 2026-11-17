import time
import uuid

import pytest

from app.kb import dualwrite, milvus_client as mc
from app.kb.documents import Chunk

pytestmark = pytest.mark.asyncio(loop_scope="session")

URI = "http://localhost:19530"


async def test_vectorize_writes_bm25_and_metadata(db_session_factory, db_clean, monkeypatch):
    # 嵌入 mock 成定长向量(patch dualwrite 实际查找的名字)
    async def fake_embed(texts):
        return [[0.05] * 1024 for _ in texts]
    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", fake_embed)

    chunks = [Chunk(category="运费", questions="运费怎么算", answer="满99包邮 型号无关",
                    section_path="运费政策", content_type="faq", is_key_clause=1)]
    await dualwrite.write_pending(chunks)

    client = mc.get_client(uri=URI)
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(client, collection=name)
    try:
        done = await dualwrite.vectorize_pending(client, collection=name)
        client.load_collection(name)
        time.sleep(2)  # 等 sparse 索引对新段生效(vectorize_pending 内部已 flush)
        assert done == 1
        hits = mc.bm25_search(client, "运费", top_k=1, collection=name)
        assert hits and hits[0]["section_path"] == "运费政策" and hits[0]["category"] == "运费"
    finally:
        mc.drop(client, name)
