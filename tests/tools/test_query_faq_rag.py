import pytest

from app.tools import business


@pytest.mark.asyncio
async def test_query_faq_sufficient(monkeypatch):
    async def fake_search(query, strategy="hybrid_rerank", top_k=None, min_score=None,
                          category=None, client=None, collection="knowledge"):
        # Ruling 18:原话直传检索(理解在 retrieval 内部只做一次),策略固定 hybrid_rerank
        assert query == "邮费多少" and strategy == "hybrid_rerank"
        return [
            {"id": 1, "rerank_score": 0.9, "question": "运费", "answer": "满99包邮",
             "section_path": "运费政策", "content_type": "faq", "category": "运费"},
            {"id": 2, "rerank_score": 0.6, "question": "退货", "answer": "七天无理由",
             "section_path": "退货政策", "content_type": "policy", "category": "退货"},
        ]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts):
        assert q == "邮费多少"  # 语义闸收原话,不是改写后的标准问法
        return {"useful": True, "reason": "够"}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)

    out = await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert out["sufficient"] is True
    assert out["citations"][0]["n"] == 1 and out["citations"][0]["section_path"]
    assert "[1]" in out["evidence"]


@pytest.mark.asyncio
async def test_query_faq_retrieval_low_conf(monkeypatch):
    async def fake_search(**kw): return []  # 无召回
    monkeypatch.setattr("app.core.retrieval.search_knowledge", lambda *a, **k: fake_search(**k))
    out = await business.query_faq.ainvoke({"keyword": "火星车怎么买"})
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"


@pytest.mark.asyncio
async def test_query_faq_self_check_fail(monkeypatch):
    async def fake_search(query, **kw):
        return [{"id": 1, "rerank_score": 0.8, "question": "运费", "answer": "满99包邮",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts): return {"useful": False, "reason": "问型号但证据只讲运费"}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "Pro型号能自动铲屎吗"})
    assert out["sufficient"] is False and out["source"] == "self_check"
    assert "型号" in out["reason"]


@pytest.mark.asyncio
async def test_query_faq_mechanical_gate_low_score(monkeypatch):
    """Ruling 18-3/8a:最高 rerank 分 0.1 < 0.3 → retrieval_low_conf,语义闸不跑。"""
    async def fake_search(query, **kw):
        return [{"id": 1, "rerank_score": 0.1, "question": "运费", "answer": "满99包邮",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    called = False
    async def fake_check(q, texts):
        nonlocal called
        called = True
        return {"useful": True, "reason": ""}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "运费"})
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"
    assert out["citations"] == []
    assert called is False  # 机械闸先于语义闸,自评绝不被调用


@pytest.mark.asyncio
async def test_query_faq_head_tail_numbering(monkeypatch):
    """Ruling 18-5/8b:首尾组装后编号,evidence [n] 与 citations 顺序一致。"""
    hits = [
        {"id": 1, "rerank_score": 0.9, "question": "Q1", "answer": "A1",
         "section_path": "s1", "content_type": "faq", "category": "c"},
        {"id": 2, "rerank_score": 0.8, "question": "Q2", "answer": "A2",
         "section_path": "s2", "content_type": "faq", "category": "c"},
        {"id": 3, "rerank_score": 0.7, "question": "Q3", "answer": "A3",
         "section_path": "s3", "content_type": "policy", "category": "c"},
    ]
    async def fake_search(query, **kw): return hits
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts): return {"useful": True, "reason": ""}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)

    out = await business.query_faq.ainvoke({"keyword": "问题"})
    assert out["sufficient"] is True
    cits = out["citations"]
    assert [c["id"] for c in cits] == [1, 3, 2]  # 最相关首、次相关尾
    assert [c["n"] for c in cits] == [1, 2, 3]
    assert cits[0]["content_type"] == "faq" and cits[-1]["id"] == 2
    lines = out["evidence"].split("\n")
    assert lines == ["[1] Q1: A1", "[2] Q3: A3", "[3] Q2: A2"]


@pytest.mark.asyncio
async def test_query_faq_gate_order_selfcheck_never_invoked(monkeypatch):
    """Ruling 18-8c:低分路径下语义闸若被调用直接炸测试。"""
    async def fake_search(query, **kw):
        return [{"id": 9, "rerank_score": 0.05, "question": "q", "answer": "a",
                 "section_path": "s", "content_type": "faq", "category": "c"}]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def boom(q, texts):
        raise AssertionError("机械闸未拦住,语义闸不该被调用")
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", boom)
    out = await business.query_faq.ainvoke({"keyword": "无关"})
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"
