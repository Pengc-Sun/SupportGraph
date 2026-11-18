import pytest

from app.core import retrieval


def test_arrange_head_tail():
    # 输入按相关度降序 [a,b,c,d,e] → a 首、b 尾、剩 c,d,e 居中
    out = retrieval.arrange_head_tail(["a", "b", "c", "d", "e"])
    assert out[0] == "a" and out[-1] == "b"
    assert set(out[1:-1]) == {"c", "d", "e"}


@pytest.mark.asyncio
async def test_strategy_vector(monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.retrieval.understand", fake_understand)
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    called = {}
    def fake_dense(client, vec, top_k, category=None, collection="knowledge"):
        called["dense"] = True
        return [{"id": 1, "score": 0.9, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.kb.milvus_client.dense_search", fake_dense)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="vector", client=object())
    assert called.get("dense") and out[0]["id"] == 1


@pytest.mark.asyncio
async def test_strategy_hybrid_rerank(monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.retrieval.understand", fake_understand)
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    def fake_hybrid(client, vec, text, top_k, recall=50, category=None, collection="knowledge"):
        return [
            {"id": 1, "score": 0.5, "question": "运费", "answer": "满99包邮",
             "section_path": "p1", "content_type": "faq", "category": "运费"},
            {"id": 2, "score": 0.4, "question": "型号", "answer": "Pro自动清理",
             "section_path": "p2", "content_type": "manual", "category": "商品"},
        ]
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    async def fake_rerank(q, docs, top_n=None):
        return [(1, 0.95), (0, 0.2)]  # 第 2 条(index1)最相关
    monkeypatch.setattr("app.core.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("Pro 型号", strategy="hybrid_rerank", client=object())
    assert out[0]["id"] == 2 and out[0]["rerank_score"] == 0.95


@pytest.mark.asyncio
async def test_expanded_feeds_bm25_not_dense(monkeypatch):
    # Ruling 15.1:扩展词只进 BM25 文本,绝不进 dense 嵌入;原词子串回声要被过滤
    async def fake_understand(q):
        return {"standard": "运费怎么算", "expanded": ["邮费", "快递费"]}
    monkeypatch.setattr("app.core.retrieval.understand", fake_understand)
    seen = {}
    async def fake_embed(q):
        seen["embed"] = q
        return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    def fake_hybrid(client, vec, text, top_k, recall=50, category=None, collection="knowledge"):
        seen["text"] = text
        return []
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    out = await retrieval.search_knowledge("邮费是多少", strategy="hybrid", client=object())
    assert out == []
    assert seen["embed"] == "运费怎么算"        # dense 只吃 standard
    # 「邮费」是原 query 子串 → 过滤;「快递费」保留进 BM25
    assert seen["text"] == "运费怎么算 快递费"


@pytest.mark.asyncio
async def test_understand_failure_degrades_to_raw_query(monkeypatch):
    # Ruling 15.1:LLM 挂了检索必须活下来,降级用原 query
    async def boom(q):
        raise RuntimeError("LLM down")
    monkeypatch.setattr("app.core.retrieval.understand", boom)
    seen = {}
    async def fake_embed(q):
        seen["embed"] = q
        return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    def fake_dense(client, vec, top_k, category=None, collection="knowledge"):
        return [{"id": 1, "score": 0.9, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.kb.milvus_client.dense_search", fake_dense)
    out = await retrieval.search_knowledge("邮费是多少", strategy="vector", client=object())
    assert seen["embed"] == "邮费是多少" and out[0]["id"] == 1


@pytest.mark.asyncio
async def test_min_score_filters_by_rerank_score(monkeypatch):
    # Ruling 15.2:hybrid_rerank 的 min_score 按 rerank_score 过滤;rerank query 用 standard
    async def fake_understand(q): return {"standard": "Pro 型号是什么", "expanded": []}
    monkeypatch.setattr("app.core.retrieval.understand", fake_understand)
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    def fake_hybrid(client, vec, text, top_k, recall=50, category=None, collection="knowledge"):
        return [
            {"id": 1, "score": 0.9, "question": "运费", "answer": "满99包邮",
             "section_path": "p1", "content_type": "faq", "category": "运费"},
            {"id": 2, "score": 0.4, "question": "型号", "answer": "Pro自动清理",
             "section_path": "p2", "content_type": "manual", "category": "商品"},
        ]
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    seen = {}
    async def fake_rerank(q, docs, top_n=None):
        seen["rerank_query"] = q
        return [(1, 0.95), (0, 0.2)]
    monkeypatch.setattr("app.core.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("Pro 型号", strategy="hybrid_rerank",
                                           min_score=0.5, client=object())
    assert seen["rerank_query"] == "Pro 型号是什么"
    # id1 原始分 0.9 虽高,但 rerank_score 0.2 < 0.5 → 被过滤
    assert [h["id"] for h in out] == [2] and out[0]["rerank_score"] == 0.95
