import pytest

from app.tools.business import query_faq

pytestmark = pytest.mark.asyncio(loop_scope="session")

# Ruling 18-6:ch03 旧 {"hits": [...]} 契约测试转换为 RAG 新契约。
# 本文件保留「参数透传」覆盖:原话直传 + strategy=hybrid_rerank + category 过滤转发;
# 管线两道闸/编号细节见 tests/tools/test_query_faq_rag.py。

async def test_query_faq_passes_raw_keyword_strategy_category(monkeypatch):
    async def fake_search(query, strategy="hybrid_rerank", top_k=None, min_score=None,
                          category=None, client=None, collection="knowledge"):
        assert query == "退货"                    # 原话透传,不在工具层做 Query 理解
        assert strategy == "hybrid_rerank"        # 固定混合+重排
        assert category == "退货"                 # 品类过滤转发
        return [{"id": 1, "rerank_score": 0.9, "question": "退货政策", "answer": "7 天无理由",
                 "section_path": "退货政策", "content_type": "policy", "category": "退货"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts): return {"useful": True, "reason": ""}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    r = await query_faq.ainvoke({"keyword": "退货", "category": "退货"})
    assert r["sufficient"] is True
    assert r["citations"][0]["question"] == "退货政策"
    assert "7 天无理由" in r["evidence"]

async def test_query_faq_miss_returns_insufficient_signal(monkeypatch):
    async def fake_search(query, **kw):
        return []
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    r = await query_faq.ainvoke({"keyword": "鞋子"})
    assert r["sufficient"] is False
    assert r["source"] == "retrieval_low_conf"
    assert r["citations"] == [] and r["reason"]
