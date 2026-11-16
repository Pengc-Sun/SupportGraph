import pytest

from app.tools.business import query_faq

pytestmark = pytest.mark.asyncio(loop_scope="session")

async def test_query_faq_hit(monkeypatch):
    async def fake_search(keyword, **kw):
        assert keyword == "退货"  # 关键词原样透传给向量检索
        return [{"id": 1, "score": 0.9, "question": "退货政策", "answer": "7 天无理由"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    r = await query_faq.ainvoke({"keyword": "退货"})
    assert r["hits"] and r["hits"][0]["question"] == "退货政策"
    assert r["hits"][0]["answer"] == "7 天无理由"

async def test_query_faq_miss_returns_message(monkeypatch):
    async def fake_search(keyword, **kw):
        return []
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    r = await query_faq.ainvoke({"keyword": "鞋子"})
    assert r["hits"] == [] and "未找到" in r["message"]
