import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import admin
from app.db import repository
from app.kb import milvus_client
from app.main import app

pytestmark = pytest.mark.asyncio(loop_scope="session")

URI = "http://localhost:19530"


@pytest.fixture()
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture()
def milvus(monkeypatch):
    # ch04 迁移:Lite → 真实 Standalone(BM25 Function 不支持 Lite),唯一集合名 + teardown drop。
    # admin 读数走默认集合:patch get_client + 包装 count,强制落在临时集合,不碰共享 knowledge。
    c = milvus_client.get_client(uri=URI)
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    milvus_client.ensure_collection(c, collection=name)
    monkeypatch.setattr(milvus_client, "get_client", lambda uri=None: c)
    monkeypatch.setattr(milvus_client, "COLLECTION", name)
    orig_count = milvus_client.count
    def wrap_count(*args, _orig=orig_count, **kwargs):
        kwargs["collection"] = name
        return _orig(*args, **kwargs)
    monkeypatch.setattr(milvus_client, "count", wrap_count)
    yield c
    milvus_client.drop(c, name)


def _kb_card(cards):
    return next(c for c in cards if c["key"] == "kb")


async def test_admin_overview_kb_card_ok(client, db_session_factory, db_clean, milvus):
    id1 = await repository.insert_knowledge_chunk("c", "q1", "a1")
    await repository.insert_knowledge_chunk("c", "q2", "a2")
    await repository.mark_chunk_vectorized(id1, str(id1))
    milvus_client.upsert_vectors(milvus, [{
        "id": id1, "dense": [1.0] + [0.0] * (milvus_client.DIM - 1),
        "text": "c\nq1\na1", "question": "q1", "answer": "a1",
        "section_path": "", "content_type": "", "category": "c",
    }], collection=milvus_client.COLLECTION)
    milvus_client.flush(milvus, collection=milvus_client.COLLECTION)  # flush 后 count 才可见
    async with client as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    cards = r.json()["cards"]
    card = _kb_card(cards)
    assert card["title"] and card["path"]  # 每张卡带入口
    assert card["status"] == "正常"
    m = card["metrics"]
    assert m["total"] == 2 and m["pending"] == 1 and m["done"] == 1
    assert m["milvus_count"] == 1 and m["consistent"] is True
    assert isinstance(m["embed_configured"], bool)
    assert card["summary"]


async def test_admin_card_reports_mismatch_as_false(client, db_session_factory, db_clean, milvus):
    id1 = await repository.insert_knowledge_chunk("c", "q1", "a1")
    await repository.mark_chunk_vectorized(id1, str(id1))
    # Milvus 空集合:done=1 ≠ milvus=0 → 真·对不上
    async with client as c:
        r = await c.get("/api/admin/overview")
    card = _kb_card(r.json()["cards"])
    assert card["status"] == "正常"
    assert card["metrics"]["consistent"] is False


async def test_mysql_down_fails_only_that_card(client, monkeypatch):
    """mysql 读不到 → 该卡数字全 null + 读数失败;别的模块卡不连坐。"""
    async def boom():
        raise ConnectionError("MySQL 没起")
    monkeypatch.setattr(admin, "_probe_mysql", boom)

    async def ok_read():
        return {"status": "正常", "metrics": {"x": 1}, "summary": "别的模块很好"}
    fake = admin.ModuleSpec(key="fake", title="假模块", path="/fake", read=ok_read)
    monkeypatch.setattr(admin, "MODULES", [*admin.MODULES, fake])

    async with client as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200  # 整页不 5xx
    cards = r.json()["cards"]
    kb = _kb_card(cards)
    assert kb["status"] == "读数失败"
    assert all(v is None for v in kb["metrics"].values())
    assert "mysql" in kb["summary"]
    other = next(c for c in cards if c["key"] == "fake")
    assert other["status"] == "正常" and other["metrics"] == {"x": 1}


async def test_milvus_down_fails_only_that_card(client, db_session_factory, db_clean, monkeypatch):
    async def boom():
        raise ConnectionError("Milvus 离线")
    monkeypatch.setattr(admin, "_probe_milvus", boom)
    async with client as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    kb = _kb_card(r.json()["cards"])
    assert kb["status"] == "读数失败"
    assert all(v is None for v in kb["metrics"].values())
    assert "milvus" in kb["summary"]


async def test_embed_probe_failure_is_isolated(client, db_session_factory, db_clean, milvus, monkeypatch):
    async def boom():
        raise RuntimeError("embed 配置读不到")
    monkeypatch.setattr(admin, "_probe_embed", boom)
    async with client as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    kb = _kb_card(r.json()["cards"])
    assert kb["status"] == "读数失败"
    assert "embed" in kb["summary"]


async def test_reader_crash_does_not_sink_page(client, db_session_factory, db_clean, milvus, monkeypatch):
    """某模块 reader 整个炸掉也只糊它自己那张卡。"""
    async def crash():
        raise RuntimeError("模块内部错误")
    bad = admin.ModuleSpec(key="bad", title="坏模块", path="/bad", read=crash)
    monkeypatch.setattr(admin, "MODULES", [*admin.MODULES, bad])
    async with client as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert _kb_card(cards)["status"] == "正常"
    bad_card = next(c for c in cards if c["key"] == "bad")
    assert bad_card["status"] == "读数失败"


async def test_registry_is_extensible(client, monkeypatch):
    """后面新增一块 = 往注册表加一行。"""
    async def ok_read():
        return {"status": "正常", "metrics": {"n": 7}, "summary": "新模块"}
    monkeypatch.setattr(admin, "MODULES", [
        admin.ModuleSpec(key="pool", title="问题池", path="/pool", read=ok_read),
    ])
    async with client as c:
        r = await c.get("/api/admin/overview")
    cards = r.json()["cards"]
    assert len(cards) == 1
    assert cards[0]["key"] == "pool" and cards[0]["metrics"] == {"n": 7}
