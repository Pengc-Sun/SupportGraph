"""RAG 评估页 API:只读 make eval-rag 落盘的产物 json,web 层一个指标都不重算。
产物缺失/写坏(半个 json、非 dict)都是「没跑过」状态,回 200 + 该按哪个作业。
例外:编造个案台账是跨轮 DB 视图(列表/处置),幻觉率块只回查状态、不在 web 层重算。"""
import json
import logging
import pathlib
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core import jobs
from app.db import repository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag-eval")

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_REPORT = _ROOT / "data/ch04/reports/rag_eval.json"   # 测试 monkeypatch 此常量

FaithStatus = Literal["未解决", "已解决", "无需解决"]
_HALLUCINATION_EMPTY = {"judge_rate": None, "confirmed_rate": None, "ledger": None}


def read_report() -> dict | None:
    """共享读数(admin 卡也走这里,单一真相):读不到/写坏一律 None。"""
    try:
        data = json.loads(_REPORT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def best_strategy(retrieval) -> str | None:
    """总体 MRR 最高那一路(None-safe):KPI 与读图句都指着它,只算这一次。"""
    if not isinstance(retrieval, dict):
        return None
    best, best_mrr = None, None
    for st, buckets in retrieval.items():
        try:
            mrr = buckets["overall"]["mrr"]
        except (TypeError, KeyError):
            continue
        if not isinstance(mrr, (int, float)):   # 脏值(None/字符串)不参与比较
            continue
        if best_mrr is None or mrr > best_mrr:
            best, best_mrr = st, mrr
    return best


async def _hallucination(report: dict) -> dict:
    """幻觉率只算本轮:分子取本轮报告的题号,台账累计单独报——
    拿台账总条数当分子会把旧账摊到新一轮头上(实测虚高 2.33% vs 0.33%)。"""
    generation = report.get("generation")
    if not isinstance(generation, dict):
        return dict(_HALLUCINATION_EMPTY)
    cases = generation.get("faithfulness_cases") or []
    n = (report.get("meta") or {}).get("n_samples")
    judge_rate = len(cases) / n if n else None
    try:
        smap = await repository.faith_case_status_map()
        counts = await repository.faith_case_counts()
    except Exception:
        # DB 挂了只降级确认率与累计,判出率来自报告文件照常给
        logger.warning("幻觉率块:台账回查失败,确认率与累计不可用", exc_info=True)
        return {"judge_rate": judge_rate, "confirmed_rate": None, "ledger": None}
    confirmed = sum(1 for c in cases
                    if isinstance(c, dict) and smap.get(c.get("id")) == "已解决")
    return {
        "judge_rate": judge_rate,
        "confirmed_rate": confirmed / n if n else None,
        "ledger": {"total": sum(counts.values()), "unresolved": counts["未解决"],
                   "resolved": counts["已解决"], "dismissed": counts["无需解决"]},
    }


@router.get("/overview")
async def overview() -> dict:
    report = read_report()
    job = jobs.status("eval-rag")   # 缺失也要给,页面知道该按哪个按钮
    if report is None:
        return {"present": False, "meta": None, "retrieval": None,
                "evidence_coverage": None, "generation": None,
                "generation_done": False, "best": None, "job": job,
                "hallucination": dict(_HALLUCINATION_EMPTY)}
    generation = report.get("generation")
    return {
        "present": True,
        "meta": report.get("meta"),
        "retrieval": report.get("retrieval"),
        "evidence_coverage": report.get("evidence_coverage"),
        "generation": generation,
        "generation_done": generation is not None,   # 半份结果:检索段照样端出去
        "best": best_strategy(report.get("retrieval")),
        "job": job,
        "hallucination": await _hallucination(report),
    }


# ---------- 编造个案台账 ----------

def _faith_dict(r) -> dict:
    return {
        "id": r.id, "eval_id": r.eval_id, "bucket": r.bucket, "query": r.query,
        "strategy": r.strategy, "answer": r.answer, "reason": r.reason,
        "citations": r.citations, "judge_model": r.judge_model,
        "status": r.status, "resolution": r.resolution, "seen_count": r.seen_count,
        "first_seen_at": r.first_seen_at.isoformat() if r.first_seen_at else None,
        "last_seen_at": r.last_seen_at.isoformat() if r.last_seen_at else None,
        "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
    }


@router.get("/faith-cases")
async def list_faith_cases(status: FaithStatus | None = None,
                           page: int = Query(1, ge=1),
                           size: int = Query(20, ge=1, le=100)) -> dict:
    out = await repository.list_faith_cases(status, page, size)
    return {**out, "items": [_faith_dict(r) for r in out["items"]]}


class FaithStatusRequest(BaseModel):
    status: FaithStatus   # Literal 挡脏值 → 422
    resolution: str | None = None


@router.post("/faith-cases/{case_id}/status")
async def set_faith_case_status(case_id: int, req: FaithStatusRequest) -> dict:
    try:
        row = await repository.set_faith_case_status(case_id, req.status, req.resolution)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if row is None:
        raise HTTPException(status_code=404, detail="个案不存在")
    return _faith_dict(row)
