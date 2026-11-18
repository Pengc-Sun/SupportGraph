import logging

from app.config import settings
from app.core import embeddings, rerank
from app.core.query_understanding import understand
from app.kb import milvus_client

logger = logging.getLogger(__name__)

STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def arrange_head_tail(items: list) -> list:
    """最相关放首、次相关放尾,其余按序居中(缓解 lost-in-the-middle)。"""
    if len(items) <= 2:
        return items
    return [items[0], *items[2:], items[1]]


async def _understand(query: str) -> dict:
    """Query 理解只调一次;上游挂了降级原 query(检索必须活下来)。
    扩展词里原词子串的回声(T6 minor)确定性过滤掉。"""
    try:
        r = await understand(query)
    except Exception:
        logger.warning("Query 理解失败,降级为原 query: %s", query, exc_info=True)
        return {"standard": query, "expanded": []}
    standard = r.get("standard") or query
    expanded = [w for w in (r.get("expanded") or []) if w and w not in query]
    return {"standard": standard, "expanded": expanded}


async def search_knowledge(
    query: str, strategy: str = "hybrid_rerank", top_k: int | None = None,
    min_score: float | None = None, category: str | None = None,
    client=None, collection: str = "knowledge",
) -> list[dict]:
    """strategy: vector | bm25 | hybrid | hybrid_rerank。
    返回最终排序的 hit(hybrid_rerank 含 rerank_score);不做首尾组装(调用侧按需)。
    min_score 非 None 时按有效分后置过滤(hybrid_rerank 用 rerank_score,其余用 score)。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy 必须是 {list(STRATEGIES)} 之一")
    top_k = top_k or settings.rerank_top_k
    client = client or milvus_client.get_client()
    milvus_client.ensure_collection(client, collection=collection)

    u = await _understand(query)
    standard = u["standard"]
    # 扩展词只喂 BM25,绝不进 dense 嵌入;rerank query 也用 standard
    bm25_text = " ".join([standard, *u["expanded"]])

    if strategy == "vector":
        vec = await embeddings.embed_query(standard)
        hits = milvus_client.dense_search(client, vec, top_k,
                                          category=category, collection=collection)
    elif strategy == "bm25":
        hits = milvus_client.bm25_search(client, bm25_text, top_k,
                                         category=category, collection=collection)
    else:
        # hybrid / hybrid_rerank:先混合召回 recall_top_k
        vec = await embeddings.embed_query(standard)
        hits = milvus_client.hybrid_search(
            client, vec, bm25_text, settings.recall_top_k,
            recall=settings.recall_top_k, category=category, collection=collection)
        if strategy == "hybrid":
            hits = hits[:top_k]
        else:
            # hybrid_rerank:对召回结果精排
            docs = [f"{h['question']} {h['answer']}" for h in hits]
            ranked = await rerank.rerank(standard, docs, top_n=top_k)
            out = []
            for idx, score in ranked:
                h = dict(hits[idx])
                h["rerank_score"] = score
                out.append(h)
            hits = out

    if min_score is not None:
        hits = [h for h in hits if h.get("rerank_score", h["score"]) >= min_score]
    return hits
