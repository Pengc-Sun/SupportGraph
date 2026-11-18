"""知识库录入 API(/api/kb):库存盘点、切块预览(dry-run)、录入落库、补向量、检索自测、暂存表。

不变量(spec §10.2):
- 切块只有一份实现:preview/ingest 都调 documents.build_chunks;
- 双写顺序不许反:ingest 先写 MySQL(pending)再 vectorize;向量化失败 502 不回滚;
- 查重按「问法+正文」指纹(复用 dedup.normalize_question),同节多块不误杀,重录幂等;
- 依赖缺失是状态不是错误:overview 里 mysql/Milvus 各自 try/except,读不到 consistent=None。
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import jobs, retrieval
from app.db import repository
from app.kb import chunking, documents, dualwrite, milvus_client
from app.kb.dedup import normalize_question
from app.kb.sources import CONTENT_TYPES, KB_DIR, SOURCE_TYPES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb")


class PreviewRequest(BaseModel):
    # 全部给默认值:空 body 也进得来,统一由 _validate 回 400(而非 422)
    content: str = ""
    content_type: str = ""
    filename: str = ""


class IngestRequest(PreviewRequest):
    vectorize: bool = False


class SearchRequest(BaseModel):
    query: str = ""
    strategy: str = "vector"
    top_k: int | None = None
    min_score: float | None = None


def _validate(req: PreviewRequest) -> None:
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="正文为空")
    if req.content_type not in CONTENT_TYPES:
        raise HTTPException(
            status_code=400, detail=f"content_type 必须是 {sorted(CONTENT_TYPES)} 之一")
    # 白名单精确匹配:路径穿越(../../etc/passwd)不在清单里,天然 400
    if req.filename not in SOURCE_TYPES:
        raise HTTPException(status_code=400, detail="filename 不在建库材料清单中")


def _fingerprint(questions: str, answer: str) -> str:
    return normalize_question(questions) + "|" + normalize_question(answer)


def _chunk_dict(c: documents.Chunk) -> dict:
    return {
        "category": c.category, "questions": c.questions, "answer": c.answer,
        "section_path": c.section_path, "content_type": c.content_type,
        "is_key_clause": c.is_key_clause,
    }


def _row_dict(r) -> dict:
    return {
        "id": r.id, "category": r.category, "questions": r.questions, "answer": r.answer,
        "section_path": r.section_path, "content_type": r.content_type,
        "is_key_clause": r.is_key_clause, "vectorize_status": r.vectorize_status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@router.get("/overview")
async def overview() -> dict:
    # mysql 与 Milvus 各自 try/except:任一读不到,consistent=None(读不到),不是 False(对不上)
    mysql = staging = None
    recent: list[dict] = []
    try:
        mysql = await repository.knowledge_stats()
        staging = await repository.staging_stats()
        recent = [_row_dict(r) for r in await repository.list_recent_chunks(limit=20)]
    except Exception:
        logger.warning("overview:MySQL 读不到", exc_info=True)
    milvus_count = None
    try:
        client = milvus_client.get_client()
        milvus_count = (
            milvus_client.count(client)
            if client.has_collection(milvus_client.COLLECTION) else 0
        )
    except Exception:
        logger.warning("overview:Milvus 读不到", exc_info=True)
    consistent = None if mysql is None or milvus_count is None else mysql["done"] == milvus_count
    # 材料清单与作业状态不依赖任何外部服务,永远给得出来;就地切块读数也是 dry-run
    sources = []
    for f, t in SOURCE_TYPES.items():
        entry = {"filename": f, "content_type": t, "exists": (KB_DIR / f).exists(),
                 "chunks": None, "key_clauses": None, "features": None}
        if entry["exists"]:
            try:
                md = (KB_DIR / f).read_text(encoding="utf-8")
                cs = documents.build_chunks(md, content_type=t)
                entry.update(chunks=len(cs), key_clauses=sum(c.is_key_clause for c in cs),
                             features=documents.detect_features(md))
            except Exception:
                logger.warning("overview:材料 %s 切块读数失败", f, exc_info=True)
        sources.append(entry)
    return {
        "mysql": mysql, "milvus_count": milvus_count, "consistent": consistent,
        "staging": staging, "sources": sources, "jobs": jobs.list_jobs(), "recent": recent,
    }


@router.get("/sources/{filename}")
async def source_content(filename: str) -> dict:
    """材料原文(白名单精确匹配,路径穿越天然 404),供页面「看切块」送预览区。"""
    if filename not in SOURCE_TYPES:
        raise HTTPException(status_code=404, detail="文件不在建库材料清单中")
    path = KB_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"filename": filename, "content_type": SOURCE_TYPES[filename],
            "content": path.read_text(encoding="utf-8")}


@router.post("/preview")
async def preview(req: PreviewRequest) -> dict:
    """dry-run:只切不写,零 DB 写入。预览什么样,入库就是什么样。"""
    _validate(req)
    chunks = documents.build_chunks(req.content, content_type=req.content_type)
    # 重复标记只读指纹;MySQL 读不到回 None(读不到 ≠ 不重复)
    try:
        existing = {_fingerprint(q, a) for q, a in await repository.list_chunk_pairs()}
    except Exception:
        logger.warning("preview:MySQL 读不到,重复标记不可判", exc_info=True)
        existing = None
    out = []
    for c in chunks:
        d = _chunk_dict(c)
        d["is_table"] = chunking.is_table_block(c.answer)
        d["duplicate"] = (
            None if existing is None
            else _fingerprint(c.questions, c.answer) in existing
        )
        out.append(d)
    return {
        "count": len(chunks),
        "key_clauses": sum(c.is_key_clause for c in chunks),
        "features": documents.detect_features(req.content),
        "chunks": out,
    }


@router.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    _validate(req)
    chunks = documents.build_chunks(req.content, content_type=req.content_type)
    # 指纹 = 归一化问法 + "|" + 归一化正文:同节多块(表格按行拆/递归切)正文不同,不误杀
    existing = {_fingerprint(q, a) for q, a in await repository.list_chunk_pairs()}
    kept, skipped = [], 0
    for c in chunks:
        fp = _fingerprint(c.questions, c.answer)
        if fp in existing:
            skipped += 1
        else:
            existing.add(fp)  # 同批内重复也跳过
            kept.append(c)
    ids = await dualwrite.write_pending(kept)  # 先写 MySQL(pending),权威源先落地
    vectorized = 0
    if req.vectorize:
        try:
            client = milvus_client.get_client()
            milvus_client.ensure_collection(client)
            vectorized = await dualwrite.vectorize_pending(client)
        except Exception as exc:
            logger.exception("录入后向量化失败(不回滚)")
            raise HTTPException(
                status_code=502,
                detail=f"已入库 {len(ids)} 块(pending),向量化失败:{exc};补跑一次即可(kb-vectorize)",
            ) from exc
    return {"inserted": len(ids), "skipped": skipped, "ids": ids, "vectorized": vectorized}


@router.post("/vectorize")
async def vectorize() -> dict:
    try:
        client = milvus_client.get_client()
        milvus_client.ensure_collection(client)
        n = await dualwrite.vectorize_pending(client)
        return {
            "vectorized": n,
            "pending": await repository.count_chunks_by_status("pending"),
            "milvus_count": milvus_client.count(client),
        }
    except Exception as exc:
        logger.exception("向量化失败")
        raise HTTPException(
            status_code=502,
            detail=f"向量化失败:{exc};块仍在 MySQL(pending),补跑一次即可",
        ) from exc


@router.post("/search")
async def search(req: SearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 为空")
    if req.strategy not in retrieval.STRATEGIES:
        raise HTTPException(
            status_code=400, detail=f"strategy 必须是 {list(retrieval.STRATEGIES)} 之一")
    try:
        hits = await retrieval.search_knowledge(
            req.query, strategy=req.strategy, top_k=req.top_k, min_score=req.min_score)
    except Exception as exc:
        logger.exception("检索自测失败")
        raise HTTPException(status_code=502, detail=f"检索失败:{exc}") from exc
    return {"hits": hits}


@router.get("/staging")
async def staging(status: str | None = None) -> dict:
    statuses = ["extracted", "kept", "discarded"]
    if status is not None and status not in statuses:
        raise HTTPException(status_code=400, detail=f"status 必须是 {statuses} 之一")
    try:
        stats = await repository.staging_stats()
        items = []
        for st in ([status] if status else statuses):
            items += [
                {"id": r.id, "batch_no": r.batch_no, "source_ref": r.source_ref,
                 "question": r.question, "answer": r.answer, "status": r.status}
                for r in await repository.list_staging_by_status(st)
            ]
    except Exception as exc:
        logger.exception("暂存表读取失败")
        raise HTTPException(status_code=502, detail=f"暂存表读取失败:{exc}") from exc
    items.sort(key=lambda it: it["id"])
    return {"stats": stats, "items": items}
