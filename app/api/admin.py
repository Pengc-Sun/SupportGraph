"""后台聚合首页 API(/api/admin):每个注册模块一张卡,给几个关键数与一句结论。

后面新增一块 = 往 MODULES 注册表加一行。某块的依赖没起只让它自己那张卡
显示「读数失败」(数字全 null),不连坐整页;每个依赖各自 try/except。
"""
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import APIRouter

from app.api import rageval
from app.config import settings
from app.db import repository
from app.kb import milvus_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin")


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    title: str
    path: str                              # 模块页面入口(各模块在自己原本的路径上)
    read: Callable[[], Awaitable[dict]]    # 返回 {"status", "metrics", "summary"}


# ---- 知识库卡的三个依赖探针:各自独立,读不到只影响自己 ----

async def _probe_mysql() -> dict:
    return await repository.knowledge_stats()


async def _probe_milvus() -> int:
    client = milvus_client.get_client()
    if not client.has_collection(milvus_client.COLLECTION):
        return 0
    return milvus_client.count(client)


async def _probe_embed() -> bool:
    # 只读配置,不打真实上游(首页聚合不该产生网络调用)
    return bool(settings.embed_api_key)


_KB_METRIC_KEYS = ("total", "pending", "done", "key_clauses",
                   "milvus_count", "consistent", "embed_configured")


async def _read_kb() -> dict:
    failed: list[str] = []
    try:
        mysql = await _probe_mysql()
    except Exception:
        logger.warning("admin 卡:MySQL 读不到", exc_info=True)
        mysql = None
        failed.append("mysql")
    try:
        milvus_count = await _probe_milvus()
    except Exception:
        logger.warning("admin 卡:Milvus 读不到", exc_info=True)
        milvus_count = None
        failed.append("milvus")
    try:
        embed_configured = await _probe_embed()
    except Exception:
        logger.warning("admin 卡:embed 配置读不到", exc_info=True)
        embed_configured = None
        failed.append("embed")
    if failed:
        return {
            "status": "读数失败",
            "metrics": {k: None for k in _KB_METRIC_KEYS},
            "summary": "读数失败:" + "、".join(failed),
        }
    consistent = mysql["done"] == milvus_count
    return {
        "status": "正常",
        "metrics": {**mysql, "milvus_count": milvus_count,
                    "consistent": consistent, "embed_configured": embed_configured},
        "summary": (f"共 {mysql['total']} 块,pending {mysql['pending']},"
                    f"双写{'一致' if consistent else '对不上'}"),
    }


async def _read_rageval() -> dict:
    # 与 /api/rag-eval/overview 同一份读数(read_report),首页卡不另算指标
    report = rageval.read_report()
    metrics = {"best_mrr": None, "n_samples": None, "refusal_rate": None}
    if report is None:
        return {"status": "未跑过", "metrics": metrics,
                "summary": "还没跑过 RAG 评估,去页面按一次「重跑」"}
    meta = report.get("meta") or {}
    retrieval = report.get("retrieval") or {}
    generation = report.get("generation") or {}
    best = rageval.best_strategy(retrieval)
    best_mrr = retrieval[best]["overall"]["mrr"] if best else None
    metrics = {"best_mrr": best_mrr, "n_samples": meta.get("n_samples"),
               "refusal_rate": generation.get("refusal_rate")}
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"
    summary = (f"最佳 {best or '—'} MRR {fmt(best_mrr)},"
               f"{meta.get('n_samples') or '—'} 题,库外拒答率 {fmt(metrics['refusal_rate'])}")
    return {"status": "正常", "metrics": metrics, "summary": summary}


MODULES: list[ModuleSpec] = [
    ModuleSpec(key="kb", title="知识库", path="/kb", read=_read_kb),
    ModuleSpec(key="rageval", title="RAG 评估", path="/rag-eval", read=_read_rageval),
]


@router.get("/overview")
async def overview() -> dict:
    cards = []
    for m in MODULES:
        try:
            data = await m.read()
        except Exception:
            logger.exception("admin 卡:%s 读取失败", m.key)
            data = {"status": "读数失败", "metrics": {}, "summary": "读数失败"}
        cards.append({"key": m.key, "title": m.title, "path": m.path, **data})
    return {"cards": cards}
