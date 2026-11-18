import time
import uuid

import pytest

from app.core import retrieval
from app.kb import milvus_client

pytestmark = pytest.mark.asyncio(loop_scope="session")

URI = "http://localhost:19530"


@pytest.fixture()
def milvus():
    # ch04 迁移:Lite → 真实 Standalone(BM25 Function 不支持 Lite),唯一集合名 + teardown drop
    c = milvus_client.get_client(uri=URI)
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    milvus_client.ensure_collection(c, collection=name)
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (milvus_client.DIM - 3)
    milvus_client.upsert_vectors(c, [
        {"id": 1, "dense": v_hit, "text": "运费 包邮 满99",
         "question": "运费怎么算", "answer": "满99包邮",
         "section_path": "运费政策", "content_type": "faq", "category": "运费"},
        {"id": 2, "dense": v_other, "text": "发货 时效 48小时",
         "question": "发货时效", "answer": "48小时",
         "section_path": "发货政策", "content_type": "faq", "category": "发货"},
    ], collection=name)
    milvus_client.flush(c, collection=name)  # 新数据须 flush 才可被检索
    c.load_collection(name)
    time.sleep(2)
    yield c, name
    milvus_client.drop(c, name)


async def _coro(v):
    return v


async def test_search_returns_top_hits_above_threshold(monkeypatch, milvus):
    c, name = milvus
    query_vec = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query",
                        lambda q: _coro(query_vec))
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.retrieval.understand", fake_understand)
    hits = await retrieval.search_knowledge("邮费是多少", strategy="vector", top_k=1, min_score=0.5,
                                            client=c, collection=name)
    assert hits and hits[0]["question"] == "运费怎么算"


async def test_below_threshold_filtered(monkeypatch, milvus):
    c, name = milvus
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query",
                        lambda q: _coro([1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)))
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.retrieval.understand", fake_understand)
    hits = await retrieval.search_knowledge("邮费", strategy="vector", top_k=2, min_score=0.999999,
                                            client=c, collection=name)
    assert hits == [] or all(h["score"] >= 0.999999 for h in hits)
