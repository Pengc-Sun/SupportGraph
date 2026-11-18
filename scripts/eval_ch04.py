"""ch04 四策略分桶评估:检索 Recall@K/MRR + 证据覆盖度(确定性)+ 生成三段(LLM)。
需已建库(make kb-build && make kb-vectorize)+ EMBED/RERANK/CHAT 上游可用。
运行:PYTHONPATH=. uv run python scripts/eval_ch04.py(make eval-rag)
产物:data/ch04/reports/rag_eval.{txt,json}——txt 是运行日志给人读,json 给评估页读。"""
import asyncio
import json
import pathlib
import sys
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.core import retrieval
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.tools.business import query_faq

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
GRADED_BUCKETS = ["A_policy", "B_model", "C_colloquial"]
K = 10
RETR_CONCURRENCY, GEN_CONCURRENCY, CALL_TIMEOUT = 8, 3, 45.0   # 生成压低并发 + 每调用超时

_ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_FILE = _ROOT / "tests/data/eval_ch04.jsonl"
_OUT_TXT = _ROOT / "data/ch04/reports/rag_eval.txt"
_OUT_JSON = _ROOT / "data/ch04/reports/rag_eval.json"

_LINES: list[str] = []   # 运行日志(print 的同时捕获,落 txt)
_errs: list[str] = []    # 错误记账(label: ExcType),落 json meta.errors


def _reset():
    _LINES.clear()
    _errs.clear()


def _norm(s):
    return "".join((s or "").split())   # 去全部空白,兼容"每 2 周"↔"每2周"


def _log(line):
    print(line)
    _LINES.append(line)


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def _mean(vals):
    vals = [v for v in vals if v is not None]   # 失败题(None)不计均值
    return sum(vals) / len(vals) if vals else None


async def _try(coro, label):
    """带超时跑 LLM/检索;超时/异常记账返回 None,单点故障不拖垮整轮。"""
    try:
        return await asyncio.wait_for(coro, CALL_TIMEOUT)
    except Exception as e:
        _errs.append(f"{label}: {type(e).__name__}")
        return None


def _load():
    return [json.loads(ln) for ln in EVAL_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------- 段1+2:确定性 ----------

def _coverage_mech(points, hits):
    """机械证据覆盖度(Ruling 21-2):证据文本含 category/question/answer,
    表格行要点(如"退款 | 2 小时 | 3 个工作日")依赖去空白子串匹配。"""
    if not points:
        return None
    ev = _norm("\n".join(f"{h['category']}\n{h['question']}\n{h['answer']}" for h in hits))
    return sum(1 for p in points if _norm(p) in ev) / len(points)


def _section_match(expect_section, hits):
    """首个 section_path 命中 expect_section 任一关键词的 1-based 排名;0=未命中。"""
    for i, h in enumerate(hits):
        sp = _norm(h.get("section_path"))
        if any(_norm(kw) and _norm(kw) in sp for kw in expect_section):
            return i + 1
    return 0


def _recall_mrr(samples, hits_by_id):
    ranks = [_section_match(s["expect_section"], hits_by_id.get(s["id"], [])) for s in samples]
    n = len(ranks)
    return {
        "recall": sum(1 for r in ranks if r) / n if n else None,
        "mrr": sum(1.0 / r for r in ranks if r) / n if n else None,
        "n": n,
    }


def _install_cached_understand():
    """同一 query 四策略只触发一次真实改写(Ruling 21-3):
    缓存 Task 而非结果,并发同 query 也共享同一次上游调用。"""
    real = retrieval.understand
    cache: dict[str, asyncio.Task] = {}

    async def cached_understand(query: str) -> dict:
        if query not in cache:
            cache[query] = asyncio.ensure_future(real(query))
        return await cache[query]

    retrieval.understand = cached_understand
    return cached_understand


async def _deterministic(samples):
    """段1+2:每(策略,题)检索一次存 HITS,复用算 Recall/MRR + 证据覆盖度。"""
    _install_cached_understand()
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    sem = asyncio.Semaphore(RETR_CONCURRENCY)
    HITS: dict[tuple, list] = {}

    async def one(strategy, s):
        async with sem:
            hits = await _try(
                retrieval.search_knowledge(s["query"], strategy=strategy, top_k=K),
                f"retr {strategy} {s['id']}")
        HITS[(strategy, s["id"])] = hits or []

    await asyncio.gather(*(one(st, s) for st in STRATEGIES for s in graded),
                         return_exceptions=True)

    retrieval_d, coverage_d = {}, {}
    for st in STRATEGIES:
        r, c = {}, {}
        for b in [*GRADED_BUCKETS, "overall"]:
            ss = graded if b == "overall" else [s for s in graded if s["bucket"] == b]
            hits_map = {s["id"]: HITS[(st, s["id"])] for s in ss}
            r[b] = _recall_mrr(ss, hits_map)
            c[b] = _mean([_coverage_mech(s["expect_points"], hits_map[s["id"]]) for s in ss])
        retrieval_d[st], coverage_d[st] = r, c

    _log(f"== 段1 检索 Recall@{K} / MRR(graded {len(graded)} 题 × {len(STRATEGIES)} 策略)==")
    for st in STRATEGIES:
        parts = [f"{b}: R={_fmt(retrieval_d[st][b]['recall'])} M={_fmt(retrieval_d[st][b]['mrr'])}"
                 for b in GRADED_BUCKETS]
        ov = retrieval_d[st]["overall"]
        _log(f"{st:>13} | " + " | ".join(parts) + f" | overall: R={_fmt(ov['recall'])} M={_fmt(ov['mrr'])}")
    _log(f"== 段2 证据覆盖度(机械匹配 Top-{K} 证据 × expect_points)==")
    for st in STRATEGIES:
        parts = [f"{b}: {_fmt(coverage_d[st][b])}" for b in GRADED_BUCKETS]
        _log(f"{st:>13} | " + " | ".join(parts) + f" | overall: {_fmt(coverage_d[st]['overall'])}")
    return retrieval_d, coverage_d, HITS


# ---------- 段3:生成(LLM,best-effort) ----------

class AnswerCoverage(BaseModel):
    covered: list[int] = Field(default_factory=list, description="被回答覆盖的要点编号(1 起)")


class Faithfulness(BaseModel):
    faithful: bool = Field(description="回答的关键事实是否都能被证据支撑")
    reason: str | None = Field(default="", description="判断依据")


COVERAGE_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是答案覆盖度评审员。给定用户问题、编号要点清单和客服回答,判断回答覆盖了哪些要点。"
               "covered 返回被覆盖要点的编号(1 起);语义等价即算覆盖,不要求逐字一致;"
               "回答未提及的要点不要列入。"),
    ("human", "用户问题:{query}\n\n要点清单:\n{points}\n\n客服回答:\n{answer}"),
])


def _judge_chain(prompt, schema):
    # 同 selfcheck(Ruling 17/18):关 thinking + function_calling + include_raw 宽容解析
    model = get_chat_model(extra_body={"thinking": {"type": "disabled"}})
    return prompt | model.with_structured_output(schema, method="function_calling", include_raw=True)


async def _judge(prompt, schema, payload, label):
    out = await _try(_judge_chain(prompt, schema).ainvoke(payload), label)
    if out is None:
        return None
    r = out.get("parsed")
    if r is None:
        tool_calls = getattr(out.get("raw"), "tool_calls", None) or []
        if tool_calls:
            try:
                r = schema.model_validate(tool_calls[0]["args"])   # 忽略错误工具名
            except Exception:
                r = None
    if r is None:
        _errs.append(f"{label}: ParseError")
    return r


def _text(msg):
    if isinstance(msg.content, str):
        return msg.content
    # content 为分块 list 时拼接文本片段(同 app/core/agent.py)
    return "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))


async def _gen_one(strategy, s, hits, sem):
    """单题:编号证据(首尾组装,同 query_faq)→ 生成 → 覆盖度 judge;hybrid_rerank 加 faith。"""
    async with sem:
        arranged = retrieval.arrange_head_tail(hits)
        citations = [
            {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
             "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
            for i, h in enumerate(arranged)
        ]
        evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations) or "(无证据)"
        msg = await _try((RAG_ANSWER_PROMPT | get_chat_model()).ainvoke(
            {"query": s["query"], "evidence": evidence}), f"gen {strategy} {s['id']}")
        row = {"id": s["id"], "bucket": s["bucket"], "strategy": strategy,
               "coverage": None, "faithful": None, "case": None}
        if msg is None:
            return row   # 生成失败 → 该题覆盖度 None,不计均值
        answer = _text(msg)

        points = s["expect_points"]
        if points:
            numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
            r = await _judge(COVERAGE_JUDGE_PROMPT, AnswerCoverage,
                             {"query": s["query"], "points": numbered, "answer": answer},
                             f"cov {strategy} {s['id']}")
            if r is not None:
                valid = {i for i in r.covered if 1 <= i <= len(points)}
                row["coverage"] = len(valid) / len(points)

        if strategy == "hybrid_rerank":
            fr = await _judge(FAITHFULNESS_PROMPT, Faithfulness,
                              {"evidence": evidence, "answer": answer}, f"faith {s['id']}")
            if fr is not None:
                row["faithful"] = bool(fr.faithful)
                if not fr.faithful:
                    row["case"] = {"id": s["id"], "bucket": s["bucket"], "query": s["query"],
                                   "strategy": strategy, "answer": answer,
                                   "reason": fr.reason or "", "citations": citations}
        return row


async def _refusal_one(s, sem):
    """D 桶走产品真实闸门(query_faq 两道证据闸),sufficient=False 即拒答。"""
    async with sem:
        out = await _try(query_faq.ainvoke({"keyword": s["query"]}), f"refuse {s['id']}")
    return s["id"], out


def _aggregate_generation(d_samples, gen_results, refusal_results):
    answer_coverage = {}
    faith_vals, cases = [], []
    for st in STRATEGIES:
        rows = [r for r in gen_results if r["strategy"] == st]
        per = {b: _mean([r["coverage"] for r in rows if r["bucket"] == b]) for b in GRADED_BUCKETS}
        per["overall"] = _mean([r["coverage"] for r in rows])
        answer_coverage[st] = per
        if st == "hybrid_rerank":
            faith_vals = [r["faithful"] for r in rows if r["faithful"] is not None]
            cases = [r["case"] for r in rows if r["case"]]
    refused = [sid for sid, out in refusal_results if out is not None and out.get("sufficient") is False]
    misses = [sid for sid, out in refusal_results if out is not None and out.get("sufficient") is True]
    any_ok = any(out is not None for _, out in refusal_results)
    return {
        "answer_coverage": answer_coverage,
        "faithfulness": {"hybrid_rerank": sum(faith_vals) / len(faith_vals) if faith_vals else None},
        "faithfulness_cases": cases,
        "refusal_rate": len(refused) / len(d_samples) if any_ok and d_samples else None,
        "refusal_misses": misses,
        "errors": list(_errs),
    }


async def _generation(samples, HITS):
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    d_samples = [s for s in samples if s.get("should_refuse")]
    sem = asyncio.Semaphore(GEN_CONCURRENCY)
    tasks = [_gen_one(st, s, HITS.get((st, s["id"])) or [], sem)
             for st in STRATEGIES for s in graded]
    tasks += [_refusal_one(s, sem) for s in d_samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    n_gen = len(STRATEGIES) * len(graded)
    gen_results, refusal_results = [], []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            _errs.append(f"gen task: {type(r).__name__}")
            continue
        (gen_results if i < n_gen else refusal_results).append(r)
    out = _aggregate_generation(d_samples, gen_results, refusal_results)

    _log("== 段3 生成(LLM 评审,best-effort)==")
    for st in STRATEGIES:
        parts = [f"{b}: {_fmt(out['answer_coverage'][st][b])}" for b in GRADED_BUCKETS]
        _log(f"{st:>13} | 答案覆盖度 " + " | ".join(parts)
             + f" | overall: {_fmt(out['answer_coverage'][st]['overall'])}")
    _log(f"hybrid_rerank Faithfulness: {_fmt(out['faithfulness']['hybrid_rerank'])}"
         f"(不忠实 {len(out['faithfulness_cases'])} 例)")
    _log(f"D 桶拒答率: {_fmt(out['refusal_rate'])}(漏拒 {out['refusal_misses']})")
    if _errs:
        _log(f"本轮错误 {len(_errs)} 条(详见 json meta.errors)")
    return out


# ---------- 报告 ----------

def _write_report(retrieval_d, coverage_d, generation_d, meta):
    report = {"meta": meta, "retrieval": retrieval_d,
              "evidence_coverage": coverage_d, "generation": generation_d}
    _OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_TXT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    # generation 为 None(半程结果)也照样落,评估页标注未完成
    _OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


async def main():
    _reset()
    samples = _load()
    _log(f"评估集 {EVAL_FILE.name}:{len(samples)} 题 | 策略 {STRATEGIES} | K={K}")
    retrieval_d, coverage_d, HITS = await _deterministic(samples)   # 确定性,始终产出
    generation_d = None
    try:
        generation_d = await _generation(samples, HITS)
    except Exception as e:
        _errs.append(f"generation: {type(e).__name__}")
        _log(f"[生成段未完成] {e}")   # 上游恢复后重跑补全
    meta = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_file": str(EVAL_FILE),
        "n_samples": len(samples),
        "strategies": STRATEGIES,
        "k": K,
        "chat_model": settings.chat_model,
        "errors": list(_errs),
    }
    _log(f"报告落盘:{_OUT_TXT} / {_OUT_JSON}")
    _write_report(retrieval_d, coverage_d, generation_d, meta)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
