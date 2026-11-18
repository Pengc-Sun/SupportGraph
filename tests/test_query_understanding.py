from langchain_core.runnables import RunnableLambda

from app.core import query_understanding as qu


def _patch(monkeypatch, result):
    monkeypatch.setattr(qu, "get_rewriter", lambda: RunnableLambda(lambda _: result))


async def test_understand_none_falls_back_to_raw_query(monkeypatch):
    """with_structured_output 解析失败可能返回 None 而非抛异常,须降级用原 query。"""
    _patch(monkeypatch, None)
    assert await qu.understand("坏了咋整") == {"standard": "坏了咋整", "expanded": []}


async def test_understand_empty_standard_falls_back(monkeypatch):
    _patch(monkeypatch, qu._Rewrite(standard="", expanded=["维修"]))
    assert await qu.understand("坏了咋整") == {"standard": "坏了咋整", "expanded": ["维修"]}


async def test_understand_returns_flat_fields(monkeypatch):
    _patch(monkeypatch, qu._Rewrite(standard="邮费是多少钱", expanded=["运费", "快递费"]))
    r = await qu.understand("那个邮费到底要给多少钱啊急")
    assert r == {"standard": "邮费是多少钱", "expanded": ["运费", "快递费"]}
    assert isinstance(r["expanded"], list)


async def test_understand_default_expanded_empty(monkeypatch):
    _patch(monkeypatch, qu._Rewrite(standard="商品损坏如何处理"))
    assert await qu.understand("坏了咋整") == {"standard": "商品损坏如何处理", "expanded": []}
