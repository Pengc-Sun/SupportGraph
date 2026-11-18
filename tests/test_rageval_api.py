"""T16 评估页 API:只读产物 json,不重算;缺失/写坏按「没跑过」,半份结果照样端出来。
不碰 MySQL/Milvus(admin 卡的 kb 探针用替身隔离)。"""
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import admin, rageval
from app.main import app


@pytest.fixture()
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture()
def report_path(tmp_path, monkeypatch):
    path = tmp_path / "rag_eval.json"
    monkeypatch.setattr(rageval, "_REPORT", path)
    return path


def _report(mrr=None):
    """与 scripts/eval_ch04.py _write_report 同构的产物夹具。"""
    mrr = mrr or {"vector": 0.42, "bm25": 0.55, "hybrid": 0.61, "hybrid_rerank": 0.78}
    buckets = ["A_policy", "B_model", "C_colloquial", "overall"]
    return {
        "meta": {"timestamp": "2026-11-20T10:00:00+08:00",
                 "eval_file": "tests/data/eval_ch04.jsonl", "n_samples": 80,
                 "strategies": list(mrr), "k": 10, "chat_model": "glm-x",
                 "embed_model": "BAAI/bge-m3", "rerank_model": "BAAI/bge-reranker-v2-m3",
                 "judge_model": "glm-x", "n_chunks": 123, "errors": []},
        "retrieval": {st: {b: {"recall": 0.9, "mrr": m, "n": 20} for b in buckets}
                      for st, m in mrr.items()},
        "evidence_coverage": {st: {b: 0.7 for b in buckets} for st in mrr},
        "generation": {
            "answer_coverage": {st: {b: 0.6 for b in buckets} for st in mrr},
            "faithfulness": {"hybrid_rerank": 0.95},
            "faithfulness_cases": [{"id": "a1", "bucket": "A_policy", "query": "q",
                                    "strategy": "hybrid_rerank", "answer": "a",
                                    "reason": "r", "citations": []}],
            "refusal_rate": 0.875, "refusal_misses": ["d2"], "errors": [],
        },
    }


def _write(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


# ---- 产物缺失是状态,不是 500 ----

async def test_missing_report_is_state_not_error(client, report_path):
    async with client as c:
        r = await c.get("/api/rag-eval/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["present"] is False
    assert d["job"]["name"] == "eval-rag"   # 页面知道该按哪个作业
    assert d["job"]["heavy"] is True        # 分钟级重活,二次确认
    for k in ("meta", "retrieval", "evidence_coverage", "generation", "best"):
        assert d[k] is None
    assert d["generation_done"] is False


# ---- 产物在:一字不差端出去,API 不重算 ----

async def test_present_passes_through_verbatim(client, report_path):
    report = _report()
    _write(report_path, report)
    async with client as c:
        r = await c.get("/api/rag-eval/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["present"] is True
    assert d["retrieval"] == report["retrieval"]
    assert d["evidence_coverage"] == report["evidence_coverage"]
    assert d["meta"] == report["meta"]
    assert d["generation"] == report["generation"]
    assert d["generation_done"] is True
    assert d["job"]["name"] == "eval-rag"


# ---- best 取总体 MRR 最高那一路,分数换了结论跟着换 ----

async def test_best_follows_overall_mrr(client, report_path):
    _write(report_path, _report({"vector": 0.9, "bm25": 0.5, "hybrid": 0.6, "hybrid_rerank": 0.7}))
    async with client as c:
        assert (await c.get("/api/rag-eval/overview")).json()["best"] == "vector"
        _write(report_path, _report({"vector": 0.4, "bm25": 0.5, "hybrid": 0.6, "hybrid_rerank": 0.95}))
        assert (await c.get("/api/rag-eval/overview")).json()["best"] == "hybrid_rerank"


async def test_best_none_safe(client, report_path):
    report = _report()
    for st in report["retrieval"]:
        report["retrieval"][st]["overall"]["mrr"] = None
    _write(report_path, report)
    async with client as c:
        d = (await c.get("/api/rag-eval/overview")).json()
    assert d["present"] is True and d["best"] is None


# ---- 半份结果:generation 为 null,检索段照样端出去 ----

async def test_generation_null_still_serves_retrieval(client, report_path):
    report = _report()
    report["generation"] = None
    _write(report_path, report)
    async with client as c:
        r = await c.get("/api/rag-eval/overview")
    d = r.json()
    assert d["present"] is True
    assert d["retrieval"] == report["retrieval"]
    assert d["evidence_coverage"] == report["evidence_coverage"]
    assert d["generation"] is None and d["generation_done"] is False
    assert d["best"] == "hybrid_rerank"


# ---- 产物写坏(半个 json / 非 dict)按没跑过处理 ----

async def test_corrupt_half_json_treated_as_absent(client, report_path):
    report_path.write_text('{"meta": {"n_samples": 80}, "retr', encoding="utf-8")
    async with client as c:
        r = await c.get("/api/rag-eval/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["present"] is False and d["job"]["name"] == "eval-rag"


async def test_non_dict_json_treated_as_absent(client, report_path):
    report_path.write_text("[1, 2, 3]", encoding="utf-8")
    async with client as c:
        d = (await c.get("/api/rag-eval/overview")).json()
    assert d["present"] is False


# ---- admin 首页卡:与评估页同一份读数(共享 read helper) ----

@pytest.fixture()
def kb_probes_ok(monkeypatch):
    """隔离 kb 卡探针:本文件只验 rageval 卡,不碰 MySQL/Milvus。"""
    async def mysql():
        return {"total": 0, "pending": 0, "done": 0, "key_clauses": 0}
    async def milvus():
        return 0
    async def embed():
        return True
    monkeypatch.setattr(admin, "_probe_mysql", mysql)
    monkeypatch.setattr(admin, "_probe_milvus", milvus)
    monkeypatch.setattr(admin, "_probe_embed", embed)


async def test_admin_overview_has_rageval_card(client, report_path, kb_probes_ok):
    _write(report_path, _report())
    async with client as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    card = next(c for c in r.json()["cards"] if c["key"] == "rageval")
    assert card["title"] == "RAG 评估" and card["path"] == "/rag-eval"
    assert card["status"] == "正常"
    assert card["metrics"] == {"best_mrr": 0.78, "n_samples": 80, "refusal_rate": 0.875}
    assert card["summary"]


async def test_admin_rageval_card_absent_and_half(client, report_path, kb_probes_ok):
    async with client as c:
        cards = (await c.get("/api/admin/overview")).json()["cards"]
        card = next(c for c in cards if c["key"] == "rageval")
        assert card["status"] == "未跑过"
        assert card["metrics"] == {"best_mrr": None, "n_samples": None, "refusal_rate": None}
        # 半份结果:检索段在,拒答率 None,卡不炸
        report = _report()
        report["generation"] = None
        _write(report_path, report)
        cards = (await c.get("/api/admin/overview")).json()["cards"]
    card = next(c for c in cards if c["key"] == "rageval")
    assert card["status"] == "正常"
    assert card["metrics"]["best_mrr"] == 0.78
    assert card["metrics"]["refusal_rate"] is None
