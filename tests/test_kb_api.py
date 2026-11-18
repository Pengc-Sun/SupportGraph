import time
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository
from app.kb import chunking, documents, milvus_client
from app.kb.sources import SOURCE_TYPES
from app.main import app

pytestmark = pytest.mark.asyncio(loop_scope="session")

URI = "http://localhost:19530"

MD = """# 售后知识

## 运费怎么算

满 99 元包邮,未满收 10 元运费。

## 退款时效多久

审核通过后 3 个工作日内原路退回。
"""
BODY = {"content": MD, "content_type": "faq", "filename": "product-faq.md"}

_TABLE_ROWS = "\n".join(f"| 商品{i} | 满{i}元 | 规则{i} |" for i in range(12))
TABLE_MD = f"# 运费手册\n\n## 运费规则表\n\n| 商品 | 门槛 | 说明 |\n| --- | --- | --- |\n{_TABLE_ROWS}\n"
LONG_MD = "# 保修手册\n\n## 保修说明\n\n" + "本产品保修一年,凭订单号申请。" * 40 + "\n"


@pytest.fixture()
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture()
def milvus(monkeypatch):
    # ch04 迁移:Lite → 真实 Standalone(BM25 Function 不支持 Lite),唯一集合名 + teardown drop。
    # app 代码走默认集合:patch COLLECTION + 包装默认参数在 def 时绑定的函数,
    # 强制所有读写落在临时集合,不污染共享 knowledge。
    c = milvus_client.get_client(uri=URI)
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    milvus_client.ensure_collection(c, collection=name)
    monkeypatch.setattr(milvus_client, "get_client", lambda uri=None: c)
    monkeypatch.setattr(milvus_client, "COLLECTION", name)
    for fn in ("ensure_collection", "upsert_vectors", "flush", "count",
               "dense_search", "bm25_search", "hybrid_search"):
        orig = getattr(milvus_client, fn)
        def wrap(*args, _orig=orig, **kwargs):
            kwargs["collection"] = name
            return _orig(*args, **kwargs)
        monkeypatch.setattr(milvus_client, fn, wrap)
    yield c
    milvus_client.drop(c, name)


@pytest.fixture()
def fake_embed(monkeypatch):
    async def ok(texts):
        return [[1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for _ in texts]
    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok)


def _chunk_dicts(chunks):
    return [
        {"category": c.category, "questions": c.questions, "answer": c.answer,
         "section_path": c.section_path, "content_type": c.content_type,
         "is_key_clause": c.is_key_clause,
         "is_table": chunking.is_table_block(c.answer), "duplicate": False}
        for c in chunks
    ]


# ---- Step 1: 录入 API 的四条口径 ----

async def test_preview_is_dry_run(client, db_session_factory, db_clean):
    """预览只切不写:按完预览再看库存,一块都不该多。"""
    before = await repository.knowledge_stats()
    async with client as c:
        r = await c.post("/api/kb/preview", json=BODY)
    assert r.status_code == 200
    data = r.json()
    expected = documents.build_chunks(MD, content_type="faq")
    assert len(expected) == 2
    # 预览与离线 CLI 同一份切块实现:结果逐字段一致
    assert data["count"] == 2
    assert data["chunks"] == _chunk_dicts(expected)
    assert data["key_clauses"] == sum(c.is_key_clause for c in expected)
    after = await repository.knowledge_stats()
    assert after == before
    assert after["total"] == 0
    assert await repository.list_recent_chunks() == []


async def test_ingest_then_reingest_is_idempotent(client, db_session_factory, db_clean):
    """同一份正文重录:第二次 inserted=0、skipped=全部,库里总数不变。"""
    async with client as c:
        r1 = await c.post("/api/kb/ingest", json=BODY)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["inserted"] == 2 and d1["skipped"] == 0
        total_after_first = (await repository.knowledge_stats())["total"]
        assert total_after_first == 2

        r2 = await c.post("/api/kb/ingest", json=BODY)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["inserted"] == 0
        assert d2["skipped"] == d1["inserted"] == 2
    assert (await repository.knowledge_stats())["total"] == total_after_first
    # 全部以 pending 落库(未勾选向量化)
    assert await repository.count_chunks_by_status("pending") == 2


async def test_ingest_keeps_multiple_pieces_of_one_section(client, db_session_factory, db_clean):
    """大表格按行拆出的多块共用节标题——只按问法查重会误杀,这里锁住不许。"""
    table_chunks = documents.build_chunks(TABLE_MD, content_type="manual")
    assert len(table_chunks) == 2  # 12 行 > table_max_rows=10 → 拆 2 块
    assert table_chunks[0].questions == table_chunks[1].questions == "运费规则表"
    assert table_chunks[0].answer != table_chunks[1].answer

    long_chunks = documents.build_chunks(LONG_MD, content_type="manual")
    assert len(long_chunks) >= 2  # 超长正文递归切 + 句级重叠
    assert len({c.questions for c in long_chunks}) == 1
    assert len({c.answer for c in long_chunks}) == len(long_chunks)

    body = {"content": TABLE_MD, "content_type": "manual", "filename": "after-sales-manual.md"}
    async with client as c:
        r = await c.post("/api/kb/ingest", json=body)
        assert r.status_code == 200
        assert r.json()["inserted"] == 2 and r.json()["skipped"] == 0
        # 重录同表:指纹含正文 → 全跳过而非误杀/重插
        r2 = await c.post("/api/kb/ingest", json=body)
        assert r2.json()["inserted"] == 0 and r2.json()["skipped"] == 2

        body_long = {"content": LONG_MD, "content_type": "manual", "filename": "after-sales-manual.md"}
        r3 = await c.post("/api/kb/ingest", json=body_long)
        assert r3.json()["inserted"] == len(long_chunks) and r3.json()["skipped"] == 0
    assert (await repository.knowledge_stats())["total"] == 2 + len(long_chunks)


@pytest.mark.parametrize("endpoint", ["/api/kb/preview", "/api/kb/ingest"])
@pytest.mark.parametrize("payload", [
    {},                                          # 空 body
    {**BODY, "content": ""},                     # 空正文
    {**BODY, "content": "   \n  "},              # 纯空白正文
    {**BODY, "content_type": "blog"},            # 非法 content_type
    {**BODY, "content_type": ""},
    {**BODY, "filename": "../../etc/passwd"},    # 路径穿越
    {**BODY, "filename": "../kb/product-faq.md"},
    {**BODY, "filename": "not-listed.md"},       # 不在材料清单
    {**BODY, "filename": ""},
], ids=["empty-body", "empty-content", "blank-content", "bad-ctype", "empty-ctype",
        "traversal", "relative-traversal", "unlisted", "empty-filename"])
async def test_preview_rejects_bad_input(client, db_session_factory, db_clean, endpoint, payload):
    """空正文 / 非法 content_type / 不在材料清单里的文件名(含路径穿越)一律 400。"""
    async with client as c:
        r = await c.post(endpoint, json=payload)
    assert r.status_code == 400
    assert (await repository.knowledge_stats())["total"] == 0


# ---- 双写顺序与失败口径 ----

async def test_ingest_with_vectorize_success(client, db_session_factory, db_clean, milvus, fake_embed):
    async with client as c:
        r = await c.post("/api/kb/ingest", json={**BODY, "vectorize": True})
    assert r.status_code == 200
    d = r.json()
    assert d["inserted"] == 2 and d["vectorized"] == 2
    stats = await repository.knowledge_stats()
    assert stats["done"] == 2 and stats["pending"] == 0
    assert milvus_client.count(milvus) == 2


async def test_ingest_vectorize_failure_502_no_rollback(client, db_session_factory, db_clean, milvus, monkeypatch):
    """向量化失败:502 + 「已入库 N 块」+ 补跑提示;MySQL 里的 pending 块不回滚。"""
    async def broken(texts):
        raise RuntimeError("嵌入上游挂了")
    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", broken)
    async with client as c:
        r = await c.post("/api/kb/ingest", json={**BODY, "vectorize": True})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "已入库 2 块" in detail
    assert "补跑" in detail
    stats = await repository.knowledge_stats()
    assert stats["total"] == 2 and stats["pending"] == 2 and stats["done"] == 0


async def test_vectorize_endpoint_picks_up_pending(client, db_session_factory, db_clean, milvus, fake_embed):
    async with client as c:
        r = await c.post("/api/kb/ingest", json=BODY)
        assert r.json()["inserted"] == 2
        r2 = await c.post("/api/kb/vectorize")
    assert r2.status_code == 200
    d = r2.json()
    assert d["vectorized"] == 2 and d["pending"] == 0 and d["milvus_count"] == 2
    assert await repository.count_chunks_by_status("done") == 2


# ---- overview:读不到 ≠ 对不上 ----

async def test_overview_consistent_none_when_milvus_down(client, db_session_factory, db_clean, monkeypatch):
    await repository.insert_knowledge_chunk("c", "q", "a")
    def boom(uri=None):
        raise ConnectionError("Milvus 离线")
    monkeypatch.setattr(milvus_client, "get_client", boom)
    async with client as c:
        r = await c.get("/api/kb/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["mysql"]["total"] == 1 and d["mysql"]["pending"] == 1
    assert d["milvus_count"] is None
    assert d["consistent"] is None  # 读不到,不是 False(对不上)
    assert len(d["sources"]) == len(SOURCE_TYPES)   # 材料清单不依赖 Milvus
    assert d["jobs"]                # 作业状态照样给


async def test_overview_consistent_none_when_mysql_down(client, db_session_factory, db_clean, milvus, monkeypatch):
    async def boom():
        raise ConnectionError("MySQL 没起")
    monkeypatch.setattr(repository, "knowledge_stats", boom)
    async with client as c:
        r = await c.get("/api/kb/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["mysql"] is None
    assert d["milvus_count"] == 0
    assert d["consistent"] is None
    assert len(d["sources"]) == len(SOURCE_TYPES)
    assert d["staging"] is None     # 暂存读数同样不连坐
    assert d["jobs"]


async def test_overview_consistent_true_then_false(client, db_session_factory, db_clean, milvus, fake_embed):
    async with client as c:
        await c.post("/api/kb/ingest", json={**BODY, "vectorize": True})
        r = await c.get("/api/kb/overview")
        assert r.json()["consistent"] is True
        # Milvus 多出一条 → done 数对不上 → False(真·对不上)
        milvus_client.upsert_vectors(milvus, [{
            "id": 99999, "dense": [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3),
            "text": "幽灵 多出来的", "question": "幽灵", "answer": "多出来的",
            "section_path": "", "content_type": "", "category": "",
        }])
        milvus_client.flush(milvus)
        r2 = await c.get("/api/kb/overview")
        assert r2.json()["consistent"] is False
        assert r2.json()["mysql"]["done"] == 2 and r2.json()["milvus_count"] == 3


async def test_overview_lists_sources_and_recent(client, db_session_factory, db_clean, milvus):
    from app.kb.sources import SOURCE_TYPES
    await repository.insert_knowledge_chunk("c", "最近的问题", "a")
    async with client as c:
        r = await c.get("/api/kb/overview")
    d = r.json()
    assert {s["filename"] for s in d["sources"]} == set(SOURCE_TYPES)
    assert all(s["exists"] for s in d["sources"])
    assert d["recent"][0]["questions"] == "最近的问题"
    assert d["recent"][0]["vectorize_status"] == "pending"


# ---- 检索自测 ----

async def test_search_endpoint(client, milvus, monkeypatch):
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (milvus_client.DIM - 3)
    milvus_client.upsert_vectors(milvus, [
        {"id": 1, "dense": v_hit, "text": "运费 包邮 满99",
         "question": "运费怎么算", "answer": "满99包邮",
         "section_path": "运费政策", "content_type": "faq", "category": "运费"},
        {"id": 2, "dense": v_other, "text": "发货 时效 48小时",
         "question": "发货时效", "answer": "48小时",
         "section_path": "发货政策", "content_type": "faq", "category": "发货"},
    ])
    milvus_client.flush(milvus)  # 新数据须 flush 才可被检索
    milvus.load_collection(milvus_client.COLLECTION)
    time.sleep(2)
    async def fake_query(q):
        return v_hit
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_query)
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.retrieval.understand", fake_understand)
    async with client as c:
        r = await c.post("/api/kb/search", json={"query": "邮费是多少", "top_k": 1, "min_score": 0.5})
        assert r.status_code == 200
        hits = r.json()["hits"]
        assert len(hits) == 1 and hits[0]["question"] == "运费怎么算"
        assert hits[0]["score"] >= 0.5
        # 阈值抬高 → 过滤
        r2 = await c.post("/api/kb/search", json={"query": "邮费", "min_score": 0.999999})
        assert r2.json()["hits"] == [] or all(h["score"] >= 0.999999 for h in r2.json()["hits"])
        # strategy 透传:bm25 关键词单路命中 text 含「运费」的块
        r4 = await c.post("/api/kb/search", json={"query": "运费", "strategy": "bm25", "top_k": 2})
        assert r4.status_code == 200
        assert r4.json()["hits"] and r4.json()["hits"][0]["id"] == 1
        # 空 query → 400
        r3 = await c.post("/api/kb/search", json={"query": "  "})
        assert r3.status_code == 400


async def test_search_rejects_unknown_strategy(client, milvus, monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.retrieval.understand", fake_understand)
    async def fake_query(q): return [1.0] + [0.0] * (milvus_client.DIM - 1)
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_query)
    async with client as c:
        r = await c.post("/api/kb/search", json={"query": "运费", "strategy": "nope"})
        assert r.status_code == 400
        assert "strategy" in r.json()["detail"]


# ---- 暂存表 ----

async def test_staging_endpoint(client, db_session_factory, db_clean):
    i1 = await repository.insert_staging("b1", "conv:1", "q1", "a1")
    i2 = await repository.insert_staging("b1", "conv:1", "q2", "a2")
    await repository.insert_staging("b2", "conv:2", "q3", "a3")
    await repository.set_staging_status([i1], "kept")
    await repository.set_staging_status([i2], "discarded")
    async with client as c:
        r = await c.get("/api/kb/staging")
        assert r.status_code == 200
        d = r.json()
        assert d["stats"]["extracted"] == 1 and d["stats"]["kept"] == 1
        assert d["stats"]["discarded"] == 1 and d["stats"]["batches"] == 2
        assert len(d["items"]) == 3
        assert {it["status"] for it in d["items"]} == {"extracted", "kept", "discarded"}
        r2 = await c.get("/api/kb/staging", params={"status": "kept"})
        assert r2.status_code == 200
        items = r2.json()["items"]
        assert len(items) == 1 and items[0]["question"] == "q1"
        r3 = await c.get("/api/kb/staging", params={"status": "bogus"})
        assert r3.status_code == 400


# ---- T18:预览标记 / 材料清单读数 / 材料原文白名单 ----

async def test_preview_flags_duplicates_after_ingest(client, db_session_factory, db_clean):
    """预览逐块标「是否与库里重复」:入库前全 False,入库后全 True;仍是 dry-run。"""
    async with client as c:
        r0 = await c.post("/api/kb/preview", json=BODY)
        assert all(ch["duplicate"] is False for ch in r0.json()["chunks"])
        await c.post("/api/kb/ingest", json=BODY)
        r1 = await c.post("/api/kb/preview", json=BODY)
        assert all(ch["duplicate"] is True for ch in r1.json()["chunks"])
    assert (await repository.knowledge_stats())["total"] == 2  # 预览没多写一块


async def test_preview_marks_table_chunks_and_features(client, db_session_factory, db_clean):
    table_body = {"content": TABLE_MD, "content_type": "manual", "filename": "after-sales-manual.md"}
    long_body = {"content": LONG_MD, "content_type": "manual", "filename": "after-sales-manual.md"}
    async with client as c:
        d = (await c.post("/api/kb/preview", json=table_body)).json()
        assert all(ch["is_table"] for ch in d["chunks"])
        assert "表格按行拆" in d["features"]
        d2 = (await c.post("/api/kb/preview", json=long_body)).json()
        assert not any(ch["is_table"] for ch in d2["chunks"])
        assert "递归切分+句级重叠" in d2["features"]
        d3 = (await c.post("/api/kb/preview", json=BODY)).json()
        assert d3["features"] == []  # 短 FAQ 两个特性都不触发


async def test_preview_duplicate_none_when_mysql_down(client, db_session_factory, db_clean, monkeypatch):
    """MySQL 读不到:重复标记回 None(读不到),预览本身照样 200。"""
    async def boom():
        raise ConnectionError("MySQL 没起")
    monkeypatch.setattr(repository, "list_chunk_pairs", boom)
    async with client as c:
        r = await c.post("/api/kb/preview", json=BODY)
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == 2
    assert all(ch["duplicate"] is None for ch in d["chunks"])


async def test_overview_sources_carry_chunk_stats(client, db_session_factory, db_clean):
    """材料清单带就地切块读数:块数 / 关键条款 / 触发特性,与 build_chunks 一致。"""
    from app.kb.sources import KB_DIR
    async with client as c:
        r = await c.get("/api/kb/overview")
    for s in r.json()["sources"]:
        md = (KB_DIR / s["filename"]).read_text(encoding="utf-8")
        expected = documents.build_chunks(md, content_type=s["content_type"])
        assert s["chunks"] == len(expected)
        assert s["key_clauses"] == sum(ch.is_key_clause for ch in expected)
        assert isinstance(s["features"], list)
    assert (await repository.knowledge_stats())["total"] == 0  # 就地切块不写库


async def test_source_content_whitelist(client):
    from app.kb.sources import KB_DIR
    async with client as c:
        r = await c.get("/api/kb/sources/product-faq.md")
        assert r.status_code == 200
        d = r.json()
        assert d["content_type"] == "faq"
        assert d["content"] == (KB_DIR / "product-faq.md").read_text(encoding="utf-8")
        assert (await c.get("/api/kb/sources/not-listed.md")).status_code == 404
        assert (await c.get("/api/kb/sources/..%2F..%2Fapp%2Fconfig.py")).status_code == 404


# ---- 配套 CLI(sources.py 单一清单)----

async def test_show_kb_previews_all_sources_without_db_writes(db_session_factory, db_clean, capsys):
    from scripts import show_kb
    from app.kb.sources import SOURCE_TYPES
    show_kb.main()
    out = capsys.readouterr().out
    for fname in SOURCE_TYPES:
        assert fname in out
    assert "dry-run" in out
    assert (await repository.knowledge_stats())["total"] == 0


async def test_reset_kb_clears_mysql_and_milvus(db_session_factory, db_clean, milvus, capsys):
    from scripts import reset_kb
    await repository.insert_knowledge_chunk("c", "q", "a")
    await repository.insert_staging("b1", None, "q", "a")
    milvus_client.upsert_vectors(milvus, [{
        "id": 1, "dense": [1.0] + [0.0] * (milvus_client.DIM - 1),
        "text": "问题 答案", "question": "q", "answer": "a",
        "section_path": "", "content_type": "", "category": "",
    }])
    milvus_client.flush(milvus)
    await reset_kb.main()
    out = capsys.readouterr().out
    assert "1 → 0" in out  # before/after 计数
    stats = await repository.knowledge_stats()
    assert stats["total"] == 0
    assert (await repository.staging_stats())["total"] == 0
    assert not milvus.has_collection(milvus_client.COLLECTION)
