import uuid

import pytest

from app.kb import milvus_client as mc

URI = "http://localhost:19530"


@pytest.fixture()
def coll():
    client = mc.get_client(uri=URI)
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(client, collection=name)
    rows = [
        {"id": 1, "dense": [0.1] * 1024, "text": "运费 满99元包邮 不满10元",
         "question": "运费怎么算", "answer": "满99包邮,否则10元",
         "section_path": "运费政策", "content_type": "faq", "category": "运费"},
        {"id": 2, "dense": [0.2] * 1024, "text": "智能猫砂盆 Pro 型号 自动清理",
         "question": "Pro 型号功能", "answer": "自动清理除臭",
         "section_path": "商品手册 / 猫砂盆", "content_type": "manual", "category": "商品手册"},
    ]
    mc.upsert_vectors(client, rows, collection=name)
    mc.flush(client, collection=name)          # 新数据须 flush 才可被检索
    client.load_collection(name)
    import time; time.sleep(2)                  # 等 sparse 索引对新段生效
    yield client, name
    mc.drop(client, name)


def test_bm25_hits_model_number(coll):
    client, name = coll
    hits = mc.bm25_search(client, "猫砂盆 Pro 型号", top_k=2, collection=name)
    assert hits[0]["id"] == 2
    assert hits[0]["section_path"] == "商品手册 / 猫砂盆"


def test_hybrid_returns_scored_hits(coll):
    client, name = coll
    hits = mc.hybrid_search(client, [0.1] * 1024, "运费", top_k=2, recall=10, collection=name)
    assert hits
    assert {"id", "score", "question", "answer", "section_path", "content_type", "category"} <= hits[0].keys()


def test_category_filter(coll):
    client, name = coll
    hits = mc.bm25_search(client, "型号 运费", top_k=5, category="运费", collection=name)
    assert all(h["category"] == "运费" for h in hits)
    assert all(h["id"] == 1 for h in hits)


def test_upsert_is_idempotent_by_pk(coll):
    # ch03 回归迁移:同 id 重写不产生重复行(按 PK upsert 幂等)
    client, name = coll
    row = {"id": 1, "dense": [0.1] * 1024, "text": "运费 满99元包邮 不满10元",
           "question": "运费怎么算", "answer": "满99包邮,否则10元",
           "section_path": "运费政策", "content_type": "faq", "category": "运费"}
    mc.upsert_vectors(client, [row], collection=name)
    mc.upsert_vectors(client, [row], collection=name)  # 同 id 重写
    mc.flush(client, collection=name)
    assert mc.count(client, collection=name) == 2  # 仍是 fixture 的 2 条,无重


def test_dense_cosine_score_is_similarity(coll):
    # Controller Ruling 2:COSINE distance 即相似度(越大越相似);近同向量得分须高于正交向量。
    # 注意 fixture 两行 dense([0.1]*1024 / [0.2]*1024)互为平行向量,与查询余弦同为 1.0,
    # 并列时返回顺序不定(实测两种顺序都出现过),故只断言 id 属于两行之一,不断言具体排序。
    client, name = coll
    near = mc.dense_search(client, [0.1] * 1024, top_k=1, collection=name)
    orth_vec = [1.0, -1.0] + [0.0] * (mc.DIM - 2)  # 与 [0.1]*1024 点积为 0
    orth = mc.dense_search(client, orth_vec, top_k=1, collection=name)
    assert near[0]["id"] in (1, 2)
    assert near[0]["score"] > 0.9          # 同向量 ≈ 1(实测 1.0)
    assert orth[0]["score"] < 0.1          # 正交 ≈ 0(实测 0.0)
    assert near[0]["score"] > orth[0]["score"]
