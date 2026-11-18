"""RAG 评估页 API:只读 make eval-rag 落盘的产物 json,web 层一个指标都不重算。
产物缺失/写坏(半个 json、非 dict)都是「没跑过」状态,回 200 + 该按哪个作业。"""
import json
import pathlib

from fastapi import APIRouter

from app.core import jobs

router = APIRouter(prefix="/api/rag-eval")

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_REPORT = _ROOT / "data/ch04/reports/rag_eval.json"   # 测试 monkeypatch 此常量


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
        if mrr is None:
            continue
        if best_mrr is None or mrr > best_mrr:
            best, best_mrr = st, mrr
    return best


@router.get("/overview")
async def overview() -> dict:
    report = read_report()
    job = jobs.status("eval-rag")   # 缺失也要给,页面知道该按哪个按钮
    if report is None:
        return {"present": False, "meta": None, "retrieval": None,
                "evidence_coverage": None, "generation": None,
                "generation_done": False, "best": None, "job": job}
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
    }
