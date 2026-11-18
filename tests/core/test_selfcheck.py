import pytest
from langchain_core.messages import AIMessage

from app.core import selfcheck


def _result(parsed=None, raw=None, parsing_error=None):
    """include_raw=True 时 chain ainvoke 的返回形状。"""
    return {
        "raw": raw if raw is not None else AIMessage(content=""),
        "parsed": parsed,
        "parsing_error": parsing_error,
    }


@pytest.mark.asyncio
async def test_check_sufficient_parses_flat(monkeypatch):
    parsed = selfcheck.SelfCheck(useful=False, reason="证据只讲运费,未涉及型号")
    class Chain:
        async def ainvoke(self, _): return _result(parsed=parsed)
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("Pro型号功能", ["满99包邮"])
    assert out == {"useful": False, "reason": "证据只讲运费,未涉及型号"}


@pytest.mark.asyncio
async def test_check_sufficient_none_result_fails_open(monkeypatch):
    """Ruling 16:parsed=None 且 raw 无 tool_calls → fail-open 放行。"""
    class Chain:
        async def ainvoke(self, _):
            return _result(parsed=None, parsing_error=ValueError("no tool call"))
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("退货运费谁承担", ["满99元包邮"])
    assert out == {"useful": True, "reason": ""}


@pytest.mark.asyncio
async def test_check_sufficient_upstream_exception_fails_open(monkeypatch):
    """Ruling 16:上游异常(基础设施故障)不构成证据不足 → fail-open 放行。"""
    class Chain:
        async def ainvoke(self, _): raise RuntimeError("upstream 502")
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("退货运费谁承担", ["满99元包邮"])
    assert out == {"useful": True, "reason": ""}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["Tool", "\\_Check"])
async def test_check_sufficient_wrong_tool_name_tolerant_parse(monkeypatch, tool_name):
    """Ruling 17:DeepSeek 间歇输出错误工具名(Tool / \\_Check)→ 忽略名字,按 args 宽容解析。"""
    raw = AIMessage(content="", tool_calls=[
        {"name": tool_name, "args": {"useful": False, "reason": "证据无关"}, "id": "x"},
    ])
    class Chain:
        async def ainvoke(self, _):
            return _result(parsed=None, raw=raw, parsing_error=ValueError(f"Unknown tool type: {tool_name!r}"))
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("Pro型号猫砂盆功能", ["满99元包邮"])
    assert out == {"useful": False, "reason": "证据无关"}


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    {"reason": "缺 useful 字段"},          # 缺必填字段
    {"useful": [1, 2], "reason": "类型错"},  # useful 类型错
])
async def test_check_sufficient_garbage_args_fails_open(monkeypatch, args):
    """Ruling 17:args 也校验失败 → 仍走 Ruling 16 fail-open。"""
    raw = AIMessage(content="", tool_calls=[{"name": "Tool", "args": args, "id": "x"}])
    class Chain:
        async def ainvoke(self, _):
            return _result(parsed=None, raw=raw, parsing_error=ValueError("bad args"))
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("Pro型号猫砂盆功能", ["满99元包邮"])
    assert out == {"useful": True, "reason": ""}
