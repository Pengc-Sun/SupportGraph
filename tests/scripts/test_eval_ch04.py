"""T13 评估脚本单测:纯确定性逻辑 + mock,不碰真实 Milvus/LLM/MySQL。"""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

import scripts.eval_ch04 as ev
from app.core import retrieval


@pytest.fixture(autouse=True)
def _clean():
    ev._reset()
    yield
    ev._reset()


def _hit(section="售后/退货政策是什么", question="退货政策是什么",
         answer="支持 7 天无理由退货", category="退货", hid="h1"):
    return {"id": hid, "score": 0.9, "question": question, "answer": answer,
            "section_path": section, "content_type": "text", "category": category}


# ---------- _norm / 机械覆盖度(Ruling 21-2) ----------

def test_norm_strips_all_whitespace():
    assert ev._norm("每 2 周\t换行\n") == "每2周换行"
    assert ev._norm(None) == ""
    assert ev._norm("") == ""


def test_coverage_mech_whitespace_and_table_row():
    # "每 2 周"↔"每2周";表格行要点依赖去空白后的子串匹配
    hits = [_hit(answer="会员每2周可领优惠券。退款|2 小时|3 个工作日到账")]
    points = ["每 2 周", "退款 | 2 小时 | 3 个工作日"]
    assert ev._coverage_mech(points, hits) == 1.0


def test_coverage_mech_matches_category_and_question():
    # 要点只出现在 category/question 也算盖住(证据文本≠answer-only)
    hits = [_hit(category="运费", question="如何申请退款", answer="无关内容")]
    assert ev._coverage_mech(["运费", "如何申请退款"], hits) == 1.0


def test_coverage_mech_partial_none_and_empty():
    hits = [_hit(answer="只提到甲")]
    assert ev._coverage_mech(["甲", "乙"], hits) == 0.5
    assert ev._coverage_mech([], hits) is None
    assert ev._coverage_mech(["甲"], []) == 0.0


# ---------- 段1 Recall@K / MRR(Ruling 21-4) ----------

def test_section_match_substring_and_rank():
    hits = [_hit(section="售前/运费怎么算"), _hit(section="售后/退货政策是什么")]
    assert ev._section_match(["退货政策"], hits) == 2
    assert ev._section_match(["退货 政策"], hits) == 2  # 关键词也去空白
    assert ev._section_match(["发货时效"], hits) == 0
    assert ev._section_match(["退货"], []) == 0


def test_recall_mrr_first_hit_rank2():
    samples = [{"id": "q1", "expect_section": ["退货政策"]}]
    hits = {"q1": [_hit(section="售前/运费怎么算"), _hit(section="售后/退货政策是什么")]}
    out = ev._recall_mrr(samples, hits)
    assert out["recall"] == 1.0
    assert out["mrr"] == 0.5


def test_recall_mrr_no_hit_and_aggregation():
    samples = [{"id": "q1", "expect_section": ["退货政策"]},
               {"id": "q2", "expect_section": ["发货时效"]}]
    hits = {"q1": [_hit(section="售后/退货政策是什么")], "q2": []}
    out = ev._recall_mrr(samples, hits)
    assert out["recall"] == 0.5
    assert out["mrr"] == 0.5  # (1.0 + 0) / 2
    assert ev._recall_mrr([samples[1]], {"q2": []})["mrr"] == 0.0


# ---------- _try 超时/异常记账(Ruling 21-5) ----------

async def test_try_success_and_error_accounting():
    async def ok():
        return 42
    assert await ev._try(ok(), "ok") == 42
    assert ev._errs == []

    async def boom():
        raise RuntimeError("upstream 502")
    assert await ev._try(boom(), "gen A1") is None
    assert ev._errs == ["gen A1: RuntimeError"]


async def test_try_timeout(monkeypatch):
    monkeypatch.setattr(ev, "CALL_TIMEOUT", 0.01)
    assert await ev._try(asyncio.sleep(1), "slow") is None
    assert ev._errs == ["slow: TimeoutError"]


# ---------- 改写缓存(Ruling 21-3) ----------

async def test_cached_understand_once_per_unique_query(monkeypatch):
    calls = []

    async def fake(query):
        calls.append(query)
        return {"standard": query, "expanded": []}

    monkeypatch.setattr(retrieval, "understand", fake)
    restore = ev._install_cached_understand()
    try:
        # 并发同 query 也只触发一次真实改写
        await asyncio.gather(*(retrieval.understand("q1") for _ in range(4)),
                             retrieval.understand("q2"))
        await retrieval.understand("q1")
        assert calls.count("q1") == 1
        assert calls.count("q2") == 1
    finally:
        restore()
    assert retrieval.understand is fake


# ---------- 段1+2 端到端(mock 检索) ----------

async def test_deterministic_mock_aggregation_and_hits(monkeypatch):
    samples = [
        {"id": "a1", "bucket": "A_policy", "query": "几天无理由退货",
         "expect_section": ["退货政策"], "expect_points": ["7 天无理由退货"], "should_refuse": False},
        {"id": "b1", "bucket": "B_model", "query": "多久发货",
         "expect_section": ["发货时效"], "expect_points": ["48 小时内发货"], "should_refuse": False},
        {"id": "c1", "bucket": "C_colloquial", "query": "退货咋整",
         "expect_section": ["退货政策"], "expect_points": ["7 天无理由退货"], "should_refuse": False},
        {"id": "d1", "bucket": "D_absent", "query": "有恐龙卖吗",
         "expect_section": [], "expect_points": [], "should_refuse": True},
    ]
    seen = []

    async def fake_search(query, strategy="hybrid_rerank", top_k=None, **kw):
        seen.append((strategy, query, top_k, kw.get("min_score")))
        if strategy == "vector":
            return []
        if query in ("几天无理由退货", "退货咋整"):
            return [_hit(answer="支持7天无理由退货")]
        return [_hit(section="售前/运费怎么算", answer="满99包邮")]

    monkeypatch.setattr(retrieval, "search_knowledge", fake_search)
    retrieval_d, coverage_d, HITS = await ev._deterministic(samples)

    # 每(策略,题)只检索一次;top_k=K、无 min_score;D 桶不进确定性段
    assert len(seen) == 4 * 3
    assert all(t[2] == ev.K and t[3] is None for t in seen)
    assert set(HITS) == {(st, s["id"]) for st in ev.STRATEGIES for s in samples[:3]}

    assert retrieval_d["vector"]["overall"]["recall"] == 0.0
    assert retrieval_d["vector"]["overall"]["mrr"] == 0.0
    for st in ("bm25", "hybrid", "hybrid_rerank"):
        r = retrieval_d[st]
        assert r["A_policy"]["recall"] == 1.0 and r["A_policy"]["mrr"] == 1.0
        assert r["B_model"]["recall"] == 0.0 and r["B_model"]["mrr"] == 0.0
        assert r["C_colloquial"]["recall"] == 1.0
        assert r["overall"]["recall"] == pytest.approx(2 / 3)
        assert r["overall"]["mrr"] == pytest.approx(2 / 3)
        c = coverage_d[st]
        assert c["A_policy"] == 1.0 and c["B_model"] == 0.0 and c["C_colloquial"] == 1.0
        assert c["overall"] == pytest.approx(2 / 3)
    assert coverage_d["vector"]["overall"] == 0.0


# ---------- judge 宽容解析(Ruling 17 模式) ----------

def _raw_result(parsed=None, args=None, tool_name="Tool"):
    raw = AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": "x"}]) \
        if args is not None else AIMessage(content="")
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


async def test_judge_tolerant_parse(monkeypatch):
    class Chain:
        async def ainvoke(self, _):
            return _raw_result(args={"covered": [1, 2]})
    monkeypatch.setattr(ev, "_judge_chain", lambda *a: Chain())
    r = await ev._judge(ev.COVERAGE_JUDGE_PROMPT, ev.AnswerCoverage, {}, "cov A1")
    assert r is not None and r.covered == [1, 2]
    assert ev._errs == []


async def test_judge_parsed_and_garbage(monkeypatch):
    class OkChain:
        async def ainvoke(self, _):
            return _raw_result(parsed=ev.Faithfulness(faithful=False, reason="编造时效"))
    monkeypatch.setattr(ev, "_judge_chain", lambda *a: OkChain())
    r = await ev._judge(ev.FAITHFULNESS_PROMPT, ev.Faithfulness, {}, "faith A1")
    assert r.faithful is False and r.reason == "编造时效"

    class BadChain:
        async def ainvoke(self, _):
            return _raw_result(args={"covered": "不是列表"})
    monkeypatch.setattr(ev, "_judge_chain", lambda *a: BadChain())
    assert await ev._judge(ev.COVERAGE_JUDGE_PROMPT, ev.AnswerCoverage, {}, "cov A2") is None
    assert ev._errs == ["cov A2: ParseError"]


# ---------- 段3 聚合(Ruling 21-6) ----------

def _row(sid, bucket, strategy, coverage, faithful=None, case=None):
    return {"id": sid, "bucket": bucket, "strategy": strategy,
            "coverage": coverage, "faithful": faithful, "case": case}


def test_aggregate_generation_means_and_cases():
    case = {"id": "a1", "bucket": "A_policy", "query": "q", "strategy": "hybrid_rerank",
            "answer": "编造答案", "reason": "证据未支撑", "citations": [{"n": 1}]}
    gen = [
        _row("a1", "A_policy", "hybrid_rerank", 1.0, faithful=False, case=case),
        _row("b1", "B_model", "hybrid_rerank", 0.5, faithful=True),
        _row("c1", "C_colloquial", "hybrid_rerank", None, faithful=None),  # 失败题不计均值
        _row("a1", "A_policy", "vector", 0.0),
        _row("b1", "B_model", "vector", None),
    ]
    d = [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}]
    refusal = [("d1", {"sufficient": False}), ("d2", {"sufficient": True}), ("d3", None)]
    out = ev._aggregate_generation(d, gen, refusal)
    hr = out["answer_coverage"]["hybrid_rerank"]
    assert hr["A_policy"] == 1.0 and hr["B_model"] == 0.5
    assert hr["C_colloquial"] is None
    assert hr["overall"] == 0.75
    assert out["answer_coverage"]["vector"]["overall"] == 0.0
    assert out["faithfulness"] == {"hybrid_rerank": 0.5}
    assert out["faithfulness_cases"] == [case]
    assert out["refusal_rate"] == pytest.approx(1 / 3)
    assert out["refusal_misses"] == ["d2"]
    assert out["errors"] == []


def test_aggregate_generation_all_refusal_failed():
    out = ev._aggregate_generation([{"id": "d1"}], [], [("d1", None)])
    assert out["refusal_rate"] is None
    assert out["refusal_misses"] == []
    assert out["faithfulness"] == {"hybrid_rerank": None}


# ---------- 报告落盘(Ruling 21-7) ----------

def test_write_report_generation_none(tmp_path, monkeypatch):
    txt, js = tmp_path / "sub" / "rag_eval.txt", tmp_path / "sub" / "rag_eval.json"
    monkeypatch.setattr(ev, "_OUT_TXT", txt)
    monkeypatch.setattr(ev, "_OUT_JSON", js)
    ev._log("第一行")
    ev._log("中文日志")
    meta = {"timestamp": "2026-01-01T00:00:00+08:00", "n_samples": 80, "errors": []}
    ev._write_report({"vector": {"overall": {"recall": 0.5}}},
                     {"vector": {"overall": 0.5}}, None, meta)
    report = json.loads(js.read_text(encoding="utf-8"))
    assert set(report) == {"meta", "retrieval", "evidence_coverage", "generation"}
    assert report["generation"] is None
    assert report["meta"] == meta
    assert txt.read_text(encoding="utf-8").splitlines() == ["第一行", "中文日志"]
    # ensure_ascii=False:json 原文保留中文
    ev._write_report({"向量": {}}, {}, None, meta)
    assert "向量" in js.read_text(encoding="utf-8")


# ---------- 回归:共享改写任务屏蔽取消 + 单点隔离 + 缓存可恢复(review Finding 1/2) ----------

async def test_cached_understand_waiter_timeout_spares_shared_task(monkeypatch):
    # waiter A 超时不得取消共享 Task:waiter B 仍拿到真实改写结果
    monkeypatch.setattr(ev, "CALL_TIMEOUT", 0.02)
    calls, cancelled = [], []

    async def fake(query):
        calls.append(query)
        try:
            await asyncio.sleep(0.15)
            return {"standard": query + "-std", "expanded": ["扩展"]}
        except asyncio.CancelledError:
            cancelled.append(query)
            raise

    monkeypatch.setattr(retrieval, "understand", fake)
    restore = ev._install_cached_understand()
    try:
        a, b = await asyncio.gather(
            ev._try(retrieval.understand("q1"), "waiterA"),
            retrieval.understand("q1"))
        # A 的超时取消被 wrapper 屏蔽并降级原 query;B 不受影响拿到真实改写
        assert a == {"standard": "q1", "expanded": []}
        assert b == {"standard": "q1-std", "expanded": ["扩展"]}
        assert cancelled == []   # 共享 fake 调用未被取消
        assert calls == ["q1"]
        # 缓存未被死任务污染:后续策略复用同一结果
        assert await retrieval.understand("q1") == b
        assert calls == ["q1"]
    finally:
        restore()


async def test_cached_understand_cancelled_waiter_degrades(monkeypatch):
    # waiter 被外部取消 → 降级原 query,CancelledError 不得逃逸杀死调用方
    async def fake(query):
        await asyncio.sleep(0.05)
        return {"standard": query + "-std", "expanded": []}

    monkeypatch.setattr(retrieval, "understand", fake)
    restore = ev._install_cached_understand()
    try:
        t = asyncio.create_task(retrieval.understand("q1"))
        await asyncio.sleep(0)   # 让 t 先跑到 await 共享任务处
        t.cancel()
        assert await t == {"standard": "q1", "expanded": []}
        await asyncio.sleep(0.1)   # 收尾:让共享任务自然跑完,不留悬任务
    finally:
        restore()


async def test_cached_understand_evicts_failed_task_and_retries(monkeypatch):
    # 共享调用失败后缓存须逐出:下一策略对同 query 重新发起真实调用
    calls = []

    async def fake(query):
        calls.append(query)
        if len(calls) == 1:
            raise RuntimeError("upstream down")
        return {"standard": query, "expanded": []}

    monkeypatch.setattr(retrieval, "understand", fake)
    restore = ev._install_cached_understand()
    try:
        with pytest.raises(RuntimeError):
            await retrieval.understand("q1")
        assert await retrieval.understand("q1") == {"standard": "q1", "expanded": []}
        assert calls == ["q1", "q1"]
    finally:
        restore()


def test_install_cached_understand_restore_and_idempotent(monkeypatch):
    async def fake(query):
        return {"standard": query, "expanded": []}

    monkeypatch.setattr(retrieval, "understand", fake)
    restore = ev._install_cached_understand()
    wrapped = retrieval.understand
    assert wrapped is not fake
    restore2 = ev._install_cached_understand()   # 重复安装不叠加包装
    assert retrieval.understand is wrapped
    restore2()
    assert retrieval.understand is fake
    restore()   # restore 幂等
    assert retrieval.understand is fake


async def test_deterministic_isolates_single_point_failure(monkeypatch, tmp_path):
    # 单(策略,题)死亡(CancelledError 穿透 _try)→ 记账 + 缺 HITS 不 KeyError,报告仍产出
    samples = [
        {"id": "a1", "bucket": "A_policy", "query": "几天无理由退货",
         "expect_section": ["退货政策"], "expect_points": ["7 天无理由退货"], "should_refuse": False},
        {"id": "b1", "bucket": "B_model", "query": "多久发货",
         "expect_section": ["发货时效"], "expect_points": ["48 小时内发货"], "should_refuse": False},
    ]

    async def fake_search(query, strategy="hybrid_rerank", top_k=None, **kw):
        if strategy == "vector" and query == "几天无理由退货":
            raise asyncio.CancelledError()
        return [_hit(answer="支持7天无理由退货")]

    orig_understand = retrieval.understand
    monkeypatch.setattr(retrieval, "search_knowledge", fake_search)
    retrieval_d, coverage_d, HITS = await ev._deterministic(samples)
    assert retrieval.understand is orig_understand   # 确定性段结束须还原改写函数

    assert ("vector", "a1") not in HITS
    assert "retr task: CancelledError" in ev._errs
    # 失败题:recall/MRR 记 0,覆盖度 None 不计均值;其余题/策略不受影响
    assert retrieval_d["vector"]["A_policy"] == {"recall": 0.0, "mrr": 0.0, "n": 1}
    assert retrieval_d["vector"]["overall"]["recall"] == 0.0
    assert coverage_d["vector"]["A_policy"] is None
    assert coverage_d["vector"]["B_model"] == 0.0
    assert coverage_d["vector"]["overall"] == 0.0
    for st in ("bm25", "hybrid", "hybrid_rerank"):
        assert retrieval_d[st]["A_policy"]["recall"] == 1.0
        assert retrieval_d[st]["overall"]["recall"] == 0.5
        assert coverage_d[st]["A_policy"] == 1.0
        assert coverage_d[st]["overall"] == 0.5

    # 确定性段始终能落报告
    monkeypatch.setattr(ev, "_OUT_TXT", tmp_path / "r.txt")
    monkeypatch.setattr(ev, "_OUT_JSON", tmp_path / "r.json")
    ev._write_report(retrieval_d, coverage_d, None, {"errors": list(ev._errs)})
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["retrieval"]["vector"]["A_policy"]["recall"] == 0.0
    assert report["generation"] is None
