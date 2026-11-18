"""T19 编造个案台账:一题一行、复发退回、处置说明必填、幻觉率只算本轮。
DB 走 mewhelp_test(db_session_factory/db_clean);报告产物 monkeypatch _REPORT 到 tmp。"""
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import rageval
from app.db import repository
from app.main import app

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture()
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture()
def report_path(tmp_path, monkeypatch):
    path = tmp_path / "rag_eval.json"
    monkeypatch.setattr(rageval, "_REPORT", path)
    return path


def _case(cid="a1", **kw):
    c = {"id": cid, "bucket": "A_policy", "query": "几天无理由退货",
         "strategy": "hybrid_rerank", "answer": "支持 30 天无理由退货[1]",
         "reason": "证据只说 7 天",
         "citations": [{"n": 1, "id": "h1", "section_path": "售后/退货政策是什么",
                        "question": "退货政策是什么", "answer": "支持 7 天无理由退货",
                        "content_type": "text"},
                       {"n": 2, "id": "h2", "section_path": "售前/运费怎么算",
                        "question": "运费怎么算", "answer": "满 99 包邮", "content_type": "text"}]}
    c.update(kw)
    return c


def _report(case_ids=("a1",), n_samples=300):
    """与 scripts/eval_ch04.py _write_report 同构的产物夹具(只留台账用得到的段)。"""
    return {
        "meta": {"n_samples": n_samples, "chat_model": "glm-x", "judge_model": "glm-x"},
        "retrieval": {}, "evidence_coverage": {},
        "generation": {
            "answer_coverage": {}, "faithfulness": {"hybrid_rerank": 0.99},
            "faithfulness_cases": [_case(cid) for cid in case_ids],
            "refusal_rate": None, "refusal_misses": [], "errors": [],
        },
    }


def _write(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


# ---- 一题一行:同 eval_id 再判出只更新正文快照 + seen_count+1 ----

async def test_upsert_one_row_per_eval_id(db_session_factory, db_clean):
    first = await repository.upsert_faith_case(_case(), judge_model="glm-x")
    assert first["id"] > 0 and first["recurred"] is False
    again = await repository.upsert_faith_case(
        _case(bucket="B_model", answer="新答案[2]", reason="新理由"), judge_model="glm-y")
    assert again["id"] == first["id"] and again["recurred"] is False
    out = await repository.list_faith_cases(None, 1, 20)
    assert out["total"] == 1
    row = out["items"][0]
    assert row.seen_count == 2
    assert row.bucket == "B_model" and row.answer == "新答案[2]" and row.reason == "新理由"
    assert row.judge_model == "glm-y"
    assert row.status == "未解决" and row.resolution is None
    assert row.first_seen_at is not None and row.last_seen_at is not None


# ---- 复发退回:已处置过的再判出来 → 未解决 + 清说明/处置时间 + recurred=True ----

@pytest.mark.parametrize("done_status", ["已解决", "无需解决"])
async def test_recurrence_reverts_resolved(db_session_factory, db_clean, done_status):
    out = await repository.upsert_faith_case(_case(), judge_model=None)
    row = await repository.set_faith_case_status(out["id"], done_status, "改好了")
    assert row.status == done_status and row.resolved_at is not None
    again = await repository.upsert_faith_case(_case(answer="又编造了"), judge_model=None)
    assert again["id"] == out["id"] and again["recurred"] is True
    row = (await repository.list_faith_cases(None, 1, 20))["items"][0]
    assert row.status == "未解决" and row.resolution is None and row.resolved_at is None
    assert row.seen_count == 2 and row.answer == "又编造了"


# ---- 分页 + 状态筛选:未解决排前;counts 永远全局,不受筛选影响 ----

async def test_list_pagination_filter_and_global_counts(db_session_factory, db_clean):
    ids = {}
    for cid in ["a1", "a2", "a3", "b1", "b2"]:
        ids[cid] = (await repository.upsert_faith_case(_case(cid), judge_model=None))["id"]
    await repository.set_faith_case_status(ids["b1"], "已解决", "改了提示词")
    await repository.set_faith_case_status(ids["b2"], "无需解决", "裁判误判")
    counts = {"未解决": 3, "已解决": 1, "无需解决": 1}

    out = await repository.list_faith_cases(None, 1, 2)
    assert out["total"] == 5 and out["page"] == 1 and out["size"] == 2
    assert out["counts"] == counts
    # 未解决排前,再按 last_seen_at 倒序(同秒并列按 id 倒序兜底)
    assert [r.eval_id for r in out["items"]] == ["a3", "a2"]
    assert [r.eval_id for r in (await repository.list_faith_cases(None, 2, 2))["items"]] == ["a1", "b2"]
    assert [r.eval_id for r in (await repository.list_faith_cases(None, 3, 2))["items"]] == ["b1"]
    assert (await repository.list_faith_cases(None, 4, 2))["items"] == []

    filtered = await repository.list_faith_cases("已解决", 1, 20)
    assert filtered["total"] == 1 and filtered["items"][0].eval_id == "b1"
    assert filtered["counts"] == counts


async def test_status_map(db_session_factory, db_clean):
    a = await repository.upsert_faith_case(_case("a1"), judge_model=None)
    await repository.upsert_faith_case(_case("a2"), judge_model=None)
    await repository.set_faith_case_status(a["id"], "已解决", "补了负面知识")
    assert await repository.faith_case_status_map() == {"a1": "已解决", "a2": "未解决"}


# ---- API:证据快照如实带出,分页参数默认值 ----

async def test_api_list_serves_evidence_verbatim(client, db_session_factory, db_clean):
    case = _case()
    await repository.upsert_faith_case(case, judge_model="glm-x")
    async with client as c:
        r = await c.get("/api/rag-eval/faith-cases")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 1 and d["page"] == 1 and d["size"] == 20
    assert d["counts"] == {"未解决": 1, "已解决": 0, "无需解决": 0}
    it = d["items"][0]
    assert it["eval_id"] == "a1" and it["bucket"] == "A_policy"
    assert it["citations"] == case["citations"]   # 快照一字不差,不重算不裁剪
    assert it["answer"] == case["answer"] and it["reason"] == case["reason"]
    assert it["status"] == "未解决" and it["seen_count"] == 1
    assert it["resolution"] is None and it["resolved_at"] is None
    assert it["judge_model"] == "glm-x"


# ---- 处置说明必填:空白也算没填 → 400,状态不动 ----

@pytest.mark.parametrize("status", ["已解决", "无需解决"])
@pytest.mark.parametrize("resolution", [None, "", "   "])
async def test_resolution_required(client, db_session_factory, db_clean, status, resolution):
    out = await repository.upsert_faith_case(_case(), judge_model=None)
    async with client as c:
        r = await c.post(f"/api/rag-eval/faith-cases/{out['id']}/status",
                         json={"status": status, "resolution": resolution})
    assert r.status_code == 400
    row = (await repository.list_faith_cases(None, 1, 20))["items"][0]
    assert row.status == "未解决" and row.resolved_at is None


# ---- 处置流转:标已解决带说明;退回未解决连说明与处置时间一起清空 ----

async def test_resolve_then_revert_clears(client, db_session_factory, db_clean):
    out = await repository.upsert_faith_case(_case(), judge_model=None)
    url = f"/api/rag-eval/faith-cases/{out['id']}/status"
    async with client as c:
        r = await c.post(url, json={"status": "已解决", "resolution": "补了负面知识"})
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "已解决" and d["resolution"] == "补了负面知识" and d["resolved_at"]
        r = await c.post(url, json={"status": "未解决"})
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "未解决" and d["resolution"] is None and d["resolved_at"] is None


# ---- 脏值挡在门口:非法状态/分页参数 422;不存在的 id 404 ----

async def test_dirty_status_and_paging_422(client, db_session_factory, db_clean):
    out = await repository.upsert_faith_case(_case(), judge_model=None)
    async with client as c:
        assert (await c.get("/api/rag-eval/faith-cases", params={"status": "随便"})).status_code == 422
        assert (await c.get("/api/rag-eval/faith-cases", params={"page": 0})).status_code == 422
        assert (await c.get("/api/rag-eval/faith-cases", params={"size": 0})).status_code == 422
        assert (await c.get("/api/rag-eval/faith-cases", params={"size": 101})).status_code == 422
        r = await c.post(f"/api/rag-eval/faith-cases/{out['id']}/status", json={"status": "半解决"})
        assert r.status_code == 422


async def test_missing_id_404(client, db_session_factory, db_clean):
    async with client as c:
        r = await c.post("/api/rag-eval/faith-cases/999999/status", json={"status": "未解决"})
    assert r.status_code == 404


# ---- 幻觉率只算本轮:台账 3 条、本轮只判出 1 条 → 判出率 1/300,不是 3/300 ----

async def test_hallucination_this_round_only(client, report_path, db_session_factory, db_clean):
    old1 = (await repository.upsert_faith_case(_case("old1"), judge_model=None))["id"]
    old2 = (await repository.upsert_faith_case(_case("old2"), judge_model=None))["id"]
    cur = (await repository.upsert_faith_case(_case("a1"), judge_model=None))["id"]
    await repository.set_faith_case_status(old1, "已解决", "修了")
    await repository.set_faith_case_status(old2, "已解决", "修了")
    _write(report_path, _report(case_ids=("a1",), n_samples=300))
    async with client as c:
        h = (await c.get("/api/rag-eval/overview")).json()["hallucination"]
        assert h["judge_rate"] == pytest.approx(1 / 300)
        assert h["confirmed_rate"] == 0.0   # 本轮 a1 还未解决,旧账不摊进来
        assert h["ledger"] == {"total": 3, "unresolved": 1, "resolved": 2, "dismissed": 0}

        await repository.set_faith_case_status(cur, "已解决", "补了负面知识")
        h = (await c.get("/api/rag-eval/overview")).json()["hallucination"]
        assert h["judge_rate"] == pytest.approx(1 / 300)
        assert h["confirmed_rate"] == pytest.approx(1 / 300)   # 分子只取本轮题号
        assert h["ledger"] == {"total": 3, "unresolved": 0, "resolved": 3, "dismissed": 0}

        # 无需解决 ≠ 人工确认幻觉,不进确认率分子
        await repository.set_faith_case_status(cur, "无需解决", "裁判误判")
        h = (await c.get("/api/rag-eval/overview")).json()["hallucination"]
        assert h["confirmed_rate"] == 0.0
        assert h["ledger"]["dismissed"] == 1


# ---- DB 挂了:判出率照给,确认率/累计 None,HTTP 仍 200 ----

async def test_overview_db_failure_still_serves_judge_rate(
        client, report_path, db_session_factory, db_clean, monkeypatch):
    _write(report_path, _report())

    async def boom():
        raise RuntimeError("MySQL 没起")
    monkeypatch.setattr(repository, "faith_case_status_map", boom)
    async with client as c:
        r = await c.get("/api/rag-eval/overview")
    assert r.status_code == 200
    h = r.json()["hallucination"]
    assert h["judge_rate"] == pytest.approx(1 / 300)
    assert h["confirmed_rate"] is None and h["ledger"] is None


# ---- 生成段未完成:三项全 None,不碰 DB ----

async def test_hallucination_all_none_when_generation_missing(client, report_path):
    report = _report()
    report["generation"] = None
    _write(report_path, report)
    async with client as c:
        h = (await c.get("/api/rag-eval/overview")).json()["hallucination"]
    assert h == {"judge_rate": None, "confirmed_rate": None, "ledger": None}
