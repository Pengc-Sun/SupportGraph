import pytest

from app.core import selfcheck


@pytest.mark.asyncio
async def test_check_sufficient_parses_flat(monkeypatch):
    class Fake:
        useful = False
        reason = "证据只讲运费,未涉及型号"
    async def fake_invoke(_): return Fake()
    class Chain:
        def __or__(self, o): return self
        async def ainvoke(self, _): return Fake()
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("Pro型号功能", ["满99包邮"])
    assert out == {"useful": False, "reason": "证据只讲运费,未涉及型号"}


@pytest.mark.asyncio
async def test_check_sufficient_none_result_fails_open(monkeypatch):
    """Ruling 16:with_structured_output 解析失败返回 None → fail-open 放行。"""
    class Chain:
        async def ainvoke(self, _): return None
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
