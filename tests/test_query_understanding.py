import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.core import query_understanding as qu


def _result(parsed=None, raw=None, parsing_error=None):
    """include_raw=True 时 chain ainvoke 的返回形状。"""
    return {
        "raw": raw if raw is not None else AIMessage(content=""),
        "parsed": parsed,
        "parsing_error": parsing_error,
    }


def _patch(monkeypatch, result):
    monkeypatch.setattr(qu, "get_rewriter", lambda: RunnableLambda(lambda _: result))


async def test_understand_none_falls_back_to_raw_query(monkeypatch):
    """parsed=None 且 raw 无 tool_calls → 降级用原 query。"""
    _patch(monkeypatch, _result(parsed=None, parsing_error=ValueError("no tool call")))
    assert await qu.understand("坏了咋整") == {"standard": "坏了咋整", "expanded": []}


async def test_understand_empty_standard_falls_back(monkeypatch):
    _patch(monkeypatch, _result(parsed=qu.QueryRewrite(standard="", expanded=["维修"])))
    assert await qu.understand("坏了咋整") == {"standard": "坏了咋整", "expanded": ["维修"]}


async def test_understand_returns_flat_fields(monkeypatch):
    _patch(monkeypatch, _result(parsed=qu.QueryRewrite(standard="邮费是多少钱", expanded=["运费", "快递费"])))
    r = await qu.understand("那个邮费到底要给多少钱啊急")
    assert r == {"standard": "邮费是多少钱", "expanded": ["运费", "快递费"]}
    assert isinstance(r["expanded"], list)


async def test_understand_default_expanded_empty(monkeypatch):
    _patch(monkeypatch, _result(parsed=qu.QueryRewrite(standard="商品损坏如何处理")))
    assert await qu.understand("坏了咋整") == {"standard": "商品损坏如何处理", "expanded": []}


@pytest.mark.parametrize("tool_name", ["Tool", "\\_Rewrite"])
async def test_understand_wrong_tool_name_tolerant_parse(monkeypatch, tool_name):
    """Ruling 17:错误工具名 → 忽略名字,按 args 宽容解析。"""
    raw = AIMessage(content="", tool_calls=[
        {"name": tool_name, "args": {"standard": "邮费是多少钱", "expanded": ["运费", "快递费"]}, "id": "x"},
    ])
    _patch(monkeypatch, _result(parsed=None, raw=raw, parsing_error=ValueError(f"Unknown tool type: {tool_name!r}")))
    r = await qu.understand("那个邮费到底要给多少钱啊急")
    assert r == {"standard": "邮费是多少钱", "expanded": ["运费", "快递费"]}


@pytest.mark.parametrize("args", [
    {"expanded": ["维修"]},                 # 缺必填 standard
    {"standard": "维修", "expanded": "维修"},  # expanded 类型错(str 非 list)
])
async def test_understand_garbage_args_falls_back(monkeypatch, args):
    """Ruling 17:args 校验也失败 → 降级用原 query。"""
    raw = AIMessage(content="", tool_calls=[{"name": "Tool", "args": args, "id": "x"}])
    _patch(monkeypatch, _result(parsed=None, raw=raw, parsing_error=ValueError("bad args")))
    assert await qu.understand("坏了咋整") == {"standard": "坏了咋整", "expanded": []}


async def test_understand_upstream_exception_falls_back(monkeypatch):
    """上游异常(网络等)→ 降级用原 query,检索仍可继续。"""
    def boom():
        raise RuntimeError("upstream 502")
    monkeypatch.setattr(qu, "get_rewriter", boom)
    assert await qu.understand("坏了咋整") == {"standard": "坏了咋整", "expanded": []}
