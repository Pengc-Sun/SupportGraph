from datetime import datetime

from sqlalchemy import case as sa_case, func, select

import app.db.base as db          # 用模块属性引用,便于测试 monkeypatch async_session
from app.db.models import (
    Conversation,
    Faq,
    FaithCase,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    Ticket,
)

_TICKET_SEQ = 0

def _gen_ticket_no() -> str:
    global _TICKET_SEQ
    _TICKET_SEQ += 1
    return f"T{datetime.now():%Y%m%d%H%M%S}{_TICKET_SEQ:03d}"

async def create_conversation(user_id: str) -> int:
    async with db.async_session() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        await s.commit()
        return conv.id

async def get_conversation(conversation_id: int) -> Conversation | None:
    async with db.async_session() as s:
        return await s.get(Conversation, conversation_id)

async def append_message(
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with db.async_session() as s:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        s.add(msg)
        await s.commit()
        return msg.id

async def list_messages(conversation_id: int) -> list[Message]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())

async def search_faq(keyword: str) -> list[Faq]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Faq).where(Faq.question.like(f"%{keyword}%"))
        )
        return list(result.scalars())

async def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
    ticket_no = _gen_ticket_no()
    async with db.async_session() as s:
        s.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "已转人工"
        await s.commit()
    return ticket_no

async def insert_knowledge_chunk(
    category: str, questions: str, answer: str,
    section_path: str | None = None, content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session() as s:
        row = KnowledgeChunk(
            category=category, questions=questions, answer=answer,
            section_path=section_path, content_type=content_type,
            is_key_clause=is_key_clause,
        )
        s.add(row)
        await s.commit()
        return row.id

async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())

async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"
            await s.commit()

async def set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id
            await s.commit()

async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())

async def list_all_questions() -> list[str]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions))
        return list(result.scalars())

async def insert_staging(batch_no: str, source_ref: str | None, question: str, answer: str) -> int:
    async with db.async_session() as s:
        row = QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
        )
        s.add(row)
        await s.commit()
        return row.id

async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())

async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(QaExtractionStaging, i)
            if row is not None:
                row.status = status
        await s.commit()

async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None
) -> int:
    """低置信问题池写入;source ∈ {retrieval_low_conf, self_check, user_feedback}。"""
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id

async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    async with db.async_session() as s:
        conv_ids = list((await s.execute(select(Conversation.id).order_by(Conversation.id))).scalars())
        out = []
        for cid in conv_ids:
            msgs = list((await s.execute(
                select(Message).where(Message.conversation_id == cid).order_by(Message.id)
            )).scalars())
            out.append((cid, msgs))
        return out

async def knowledge_stats() -> dict:
    """库存盘点(只读):总数 / pending / done / 关键条款数。"""
    async with db.async_session() as s:
        rows = (await s.execute(
            select(KnowledgeChunk.vectorize_status, func.count())
            .group_by(KnowledgeChunk.vectorize_status)
        )).all()
        by_status = {st: int(n) for st, n in rows}
        key_clauses = int((await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.is_key_clause == 1)
        )).scalar_one())
        return {
            "total": sum(by_status.values()),
            "pending": by_status.get("pending", 0),
            "done": by_status.get("done", 0),
            "key_clauses": key_clauses,
        }

async def list_recent_chunks(limit: int = 20) -> list[KnowledgeChunk]:
    """最近入库的块,按 id 倒序。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit)
        )
        return list(result.scalars())

async def list_chunk_pairs() -> list[tuple[str, str]]:
    """全部 (questions, answer) 对,供录入按「问法+正文」指纹查重。"""
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions, KnowledgeChunk.answer))
        return [(q, a) for q, a in result.all()]

async def staging_stats() -> dict:
    """暂存表三态计数 + 批次数(只读)。"""
    async with db.async_session() as s:
        rows = (await s.execute(
            select(QaExtractionStaging.status, func.count())
            .group_by(QaExtractionStaging.status)
        )).all()
        by_status = {st: int(n) for st, n in rows}
        batches = int((await s.execute(
            select(func.count(func.distinct(QaExtractionStaging.batch_no)))
        )).scalar_one())
        return {
            "extracted": by_status.get("extracted", 0),
            "kept": by_status.get("kept", 0),
            "discarded": by_status.get("discarded", 0),
            "total": sum(by_status.values()),
            "batches": batches,
        }

# ---------- 编造个案台账(跨轮累计,一题一行) ----------

_FAITH_STATUSES = ("未解决", "已解决", "无需解决")

async def _faith_counts(s) -> dict:
    rows = (await s.execute(
        select(FaithCase.status, func.count()).group_by(FaithCase.status)
    )).all()
    by = {st: int(n) for st, n in rows}
    return {st: by.get(st, 0) for st in _FAITH_STATUSES}

async def faith_case_counts() -> dict:
    """台账全局三态计数(overview 的 ledger 与列表页签同一口径)。"""
    async with db.async_session() as s:
        return await _faith_counts(s)

async def upsert_faith_case(case: dict, judge_model: str | None = None) -> dict:
    """一题一行:已存在则更新正文/证据快照 + seen_count+1;
    已处置过的再判出来退回「未解决」并清说明/处置时间(复发)。"""
    async with db.async_session() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == case["id"])
        )).scalar_one_or_none()
        recurred = False
        if row is None:
            row = FaithCase(eval_id=case["id"])
            s.add(row)
        else:
            row.seen_count += 1
            if row.status in ("已解决", "无需解决"):
                row.status = "未解决"
                row.resolution = None
                row.resolved_at = None
                recurred = True
        row.bucket = case["bucket"]
        row.query = case["query"]
        row.strategy = case["strategy"]
        row.answer = case["answer"]
        row.reason = case.get("reason")
        row.citations = case.get("citations")
        row.judge_model = judge_model
        await s.commit()
        await s.refresh(row)
        return {"id": row.id, "recurred": recurred}

async def list_faith_cases(status: str | None, page: int, size: int) -> dict:
    """未解决排前,再按 last_seen_at 倒序(同秒并列按 id 倒序兜底);counts 永远全局。"""
    async with db.async_session() as s:
        counts = await _faith_counts(s)
        q = select(FaithCase)
        cq = select(func.count()).select_from(FaithCase)
        if status is not None:
            q = q.where(FaithCase.status == status)
            cq = cq.where(FaithCase.status == status)
        total = int((await s.execute(cq)).scalar_one())
        rows = (await s.execute(
            q.order_by(sa_case((FaithCase.status == "未解决", 0), else_=1),
                       FaithCase.last_seen_at.desc(), FaithCase.id.desc())
            .offset((page - 1) * size).limit(size)
        )).scalars()
        return {"items": list(rows), "total": total, "page": page, "size": size, "counts": counts}

async def set_faith_case_status(case_id: int, status: str, resolution: str | None) -> FaithCase | None:
    """标已解决/无需解决必须写处置说明(空白也算没填 → ValueError,API 映射 400);
    退回未解决连说明与处置时间一起清空。id 不存在回 None(API 映射 404)。"""
    if status in ("已解决", "无需解决") and not (resolution or "").strip():
        raise ValueError("标已解决/无需解决必须填写处置说明")
    async with db.async_session() as s:
        row = await s.get(FaithCase, case_id)
        if row is None:
            return None
        row.status = status
        if status == "未解决":
            row.resolution = None
            row.resolved_at = None
        else:
            row.resolution = resolution
            row.resolved_at = func.now()   # 与 last_seen_at 同源 DB 时钟,不混时区
        await s.commit()
        await s.refresh(row)
        return row

async def faith_case_status_map() -> dict[str, str]:
    """eval_id → 处置状态;幻觉率只回查本轮题号,不拿台账总数当分子。"""
    async with db.async_session() as s:
        rows = (await s.execute(select(FaithCase.eval_id, FaithCase.status))).all()
        return {eid: st for eid, st in rows}
