"""T13 评估脚本单测:纯确定性逻辑 + mock,不碰真实 Milvus/LLM/MySQL。"""
import asyncio
import json
import logging

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


# ---------- 段1 Recall@5 / Recall@10 / MRR(Ruling 21-4 + 24-4) ----------

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
    assert out["recall_at_5"] == 1.0
    assert out["recall_at_10"] == 1.0
    assert out["mrr"] == 0.5


def test_recall_mrr_no_hit_and_aggregation():
    samples = [{"id": "q1", "expect_section": ["退货政策"]},
               {"id": "q2", "expect_section": ["发货时效"]}]
    hits = {"q1": [_hit(section="售后/退货政策是什么")], "q2": []}
    out = ev._recall_mrr(samples, hits)
    assert out["recall_at_5"] == 0.5
    assert out["recall_at_10"] == 0.5
    assert out["mrr"] == 0.5  # (1.0 + 0) / 2
    assert ev._recall_mrr([samples[1]], {"q2": []})["mrr"] == 0.0


def test_recall_depths_separated_rank7():
    # 命中在第 7 位:@5 记 0、@10 记 1;MRR 仍按 top-10 内名次
    assert ev.RECALL_K == 5 and ev.K == 10
    samples = [{"id": "q1", "expect_section": ["退货政策"]}]
    hits = {"q1": [_hit(section=f"售前/无关{i}", hid=f"h{i}") for i in range(6)]
                  + [_hit(section="售后/退货政策是什么", hid="h7")]}
    out = ev._recall_mrr(samples, hits)
    assert out["recall_at_5"] == 0.0
    assert out["recall_at_10"] == 1.0
    assert out["mrr"] == pytest.approx(1 / 7)


def test_recall_multi_group_partial_credit_and_group_mean_mrr():
    # E_multi:两组证据只凑齐一组 → recall 0.5;MRR 按各组名次倒数取平均
    samples = [{"id": "e1", "expect_section": [],
                "expect_sections_all": [["MH-W40"], ["MH-W60"]]}]
    hits = {"e1": [_hit(section="商品规格手册 / 饮水机系列 / MH-W40 自动饮水机 Plus / 参数"),
                   _hit(section="售前/无关"), _hit(section="售前/无关2"),
                   _hit(section="商品规格手册 / 饮水机系列 / MH-W60 自动饮水机 Pro 不锈钢 / 参数")]}
    out = ev._recall_mrr(samples, hits)
    assert out["recall_at_5"] == 1.0 and out["recall_at_10"] == 1.0
    assert out["mrr"] == pytest.approx((1.0 + 0.25) / 2)   # 名次 1 和 4 → 0.625

    # 第二组彻底缺席 → recall 0.5,缺席组 MRR 项记 0
    hits = {"e1": hits["e1"][:1]}
    out = ev._recall_mrr(samples, hits)
    assert out["recall_at_5"] == 0.5 and out["recall_at_10"] == 0.5
    assert out["mrr"] == 0.5   # (1.0 + 0) / 2,不是"最后一块到位"口径


def test_recall_multi_group_depth5_partial():
    # 一组在前 5、另一组在第 6-10:@5 半分、@10 全分
    samples = [{"id": "e1", "expect_section": [],
                "expect_sections_all": [["退货政策"], ["保修说明"]]}]
    hits = {"e1": [_hit(section="售后/退货政策是什么")]
                  + [_hit(section=f"售前/无关{i}", hid=f"h{i}") for i in range(5)]
                  + [_hit(section="售后手册 / 保修说明", hid="h7")]}
    out = ev._recall_mrr(samples, hits)
    assert out["recall_at_5"] == 0.5
    assert out["recall_at_10"] == 1.0
    assert out["mrr"] == pytest.approx((1.0 + 1 / 7) / 2)
    assert out["n"] == 1


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
        {"id": "e1", "bucket": "E_multi", "query": "俩饮水机换芯周期各多久",
         "expect_section": [], "expect_sections_all": [["MH-W40"], ["MH-W60"]],
         "expect_points": ["每 3 周更换一次滤芯", "每 4 周更换一次滤芯"], "should_refuse": False},
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
        if query == "俩饮水机换芯周期各多久":
            return [_hit(section="商品规格手册 / 饮水机系列 / MH-W40 自动饮水机 Plus / 参数",
                         answer="建议每 3 周更换一次滤芯", hid="w40"),
                    _hit(section="商品规格手册 / 饮水机系列 / MH-W60 自动饮水机 Pro 不锈钢 / 参数",
                         answer="建议每 4 周更换一次滤芯", hid="w60")]
        return [_hit(section="售前/运费怎么算", answer="满99包邮")]

    monkeypatch.setattr(retrieval, "search_knowledge", fake_search)
    retrieval_d, coverage_d, HITS = await ev._deterministic(samples)

    # 每(策略,题)只检索一次;top_k=K、无 min_score;D 桶不进确定性段
    assert len(seen) == 4 * 4
    assert all(t[2] == ev.K and t[3] is None for t in seen)
    assert set(HITS) == {(st, s["id"]) for st in ev.STRATEGIES for s in samples[:4]}

    assert retrieval_d["vector"]["overall"]["recall_at_5"] == 0.0
    assert retrieval_d["vector"]["overall"]["recall_at_10"] == 0.0
    assert retrieval_d["vector"]["overall"]["mrr"] == 0.0
    for st in ("bm25", "hybrid", "hybrid_rerank"):
        r = retrieval_d[st]
        assert r["A_policy"]["recall_at_5"] == 1.0 and r["A_policy"]["mrr"] == 1.0
        assert r["B_model"]["recall_at_5"] == 0.0 and r["B_model"]["mrr"] == 0.0
        assert r["C_colloquial"]["recall_at_10"] == 1.0
        # E 桶两组全命中:recall 满分,MRR 按组平均 (1 + 1/2)/2
        assert r["E_multi"] == {"recall_at_5": 1.0, "recall_at_10": 1.0,
                                "mrr": 0.75, "n": 1}
        assert r["overall"]["recall_at_5"] == pytest.approx(3 / 4)
        assert r["overall"]["mrr"] == pytest.approx((1 + 0 + 1 + 0.75) / 4)
        c = coverage_d[st]
        assert c["A_policy"] == 1.0 and c["B_model"] == 0.0 and c["C_colloquial"] == 1.0
        assert c["E_multi"] == 1.0
        assert c["overall"] == pytest.approx(3 / 4)
    assert coverage_d["vector"]["overall"] == 0.0


# ---------- judge 宽容解析(Ruling 17 模式) ----------

def test_judge_chain_temperature_zero(monkeypatch):
    # T20 尺子去抖:两个裁判共用的 _judge_chain 显式 temperature=0
    seen = {}
    real = ev.get_chat_model

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    monkeypatch.setattr(ev, "get_chat_model", spy)
    ev._judge_chain(ev.FAITHFULNESS_PROMPT, ev.Faithfulness)
    assert seen["temperature"] == 0
    assert seen["extra_body"] == {"thinking": {"type": "disabled"}}

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


# ---------- meta(Ruling 22-1:模型配置 + 报告时知识库块数) ----------

def test_build_meta_models_and_n_chunks(monkeypatch):
    monkeypatch.setattr(ev.milvus_client, "get_client", lambda uri=None: object())
    monkeypatch.setattr(ev.milvus_client, "count", lambda client, **kw: 321)
    meta = ev._build_meta([{"id": "q1"}, {"id": "q2"}, {"id": "q3"}])
    assert meta["n_samples"] == 3
    assert meta["embed_model"] == ev.settings.embed_model
    assert meta["rerank_model"] == ev.settings.rerank_model
    # 裁判与生成同一个 chat 模型(自评自判是已知取舍)
    assert meta["judge_model"] == meta["chat_model"] == ev.settings.chat_model
    assert meta["n_chunks"] == 321
    assert meta["k"] == ev.K and meta["strategies"] == ev.STRATEGIES
    assert meta["recall_k"] == ev.RECALL_K == 5   # 召回口径与检索深度分开配
    assert meta["eval_file"] == str(ev.EVAL_FILE)
    assert "timestamp" in meta and meta["errors"] == []


def test_build_meta_milvus_down_gives_none(monkeypatch):
    def boom(uri=None):
        raise ConnectionError("Milvus 没起")
    monkeypatch.setattr(ev.milvus_client, "get_client", boom)
    meta = ev._build_meta([])
    assert meta["n_chunks"] is None   # 读不到不阻塞报告
    assert meta["n_samples"] == 0


def test_build_meta_count_failure_gives_none(monkeypatch):
    monkeypatch.setattr(ev.milvus_client, "get_client", lambda uri=None: object())
    def boom(client, **kw):
        raise RuntimeError("count 失败")
    monkeypatch.setattr(ev.milvus_client, "count", boom)
    assert ev._build_meta([])["n_chunks"] is None


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


async def test_cached_understand_failure_eviction_identity_guard(monkeypatch):
    # 失败逐出须验明正身:A 失败后立刻重试装入新任务,B 的失败收尾不得误逐新任务
    calls = []

    async def fake(query):
        calls.append(query)
        if len(calls) == 1:
            raise RuntimeError("upstream down")
        await asyncio.sleep(0.05)
        return {"standard": query, "expanded": []}

    monkeypatch.setattr(retrieval, "understand", fake)
    restore = ev._install_cached_understand()
    try:
        async def waiter():
            with pytest.raises(RuntimeError):
                await retrieval.understand("q1")
            return await retrieval.understand("q1")   # 失败后立刻重试

        a, b = await asyncio.gather(waiter(), waiter())
        assert a == b == {"standard": "q1", "expanded": []}
        # 只失败一次 + 重试一次:B 复用 A 装入的新任务,而不是再发起第三次真实调用
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
    assert retrieval_d["vector"]["A_policy"] == {"recall_at_5": 0.0, "recall_at_10": 0.0,
                                                 "mrr": 0.0, "n": 1}
    assert retrieval_d["vector"]["overall"]["recall_at_5"] == 0.0
    assert coverage_d["vector"]["A_policy"] is None
    assert coverage_d["vector"]["B_model"] == 0.0
    assert coverage_d["vector"]["overall"] == 0.0
    for st in ("bm25", "hybrid", "hybrid_rerank"):
        assert retrieval_d[st]["A_policy"]["recall_at_10"] == 1.0
        assert retrieval_d[st]["overall"]["recall_at_5"] == 0.5
        assert coverage_d[st]["A_policy"] == 1.0
        assert coverage_d[st]["overall"] == 0.5

    # 确定性段始终能落报告
    monkeypatch.setattr(ev, "_OUT_TXT", tmp_path / "r.txt")
    monkeypatch.setattr(ev, "_OUT_JSON", tmp_path / "r.json")
    ev._write_report(retrieval_d, coverage_d, None, {"errors": list(ev._errs)})
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["retrieval"]["vector"]["A_policy"]["recall_at_5"] == 0.0
    assert report["generation"] is None


# ---------- T19 台账落库(附加管理视图:写库失败只打一行日志,报告照落) ----------

def _gen_d(cases):
    return {"answer_coverage": {}, "faithfulness": {"hybrid_rerank": None},
            "faithfulness_cases": cases, "refusal_rate": None,
            "refusal_misses": [], "errors": []}


def _ledger_case(cid):
    return {"id": cid, "bucket": "A_policy", "query": "q", "strategy": "hybrid_rerank",
            "answer": "a", "reason": "r", "citations": [{"n": 1}]}


async def test_write_ledger_upserts_each_case_and_logs_recurred(monkeypatch):
    from app.db import repository   # 脚本侧是懒 import,测试替身打在仓储模块上
    calls = []

    async def fake_upsert(case, judge_model=None):
        calls.append((case, judge_model))
        return {"id": len(calls), "recurred": case["id"] == "a2"}
    monkeypatch.setattr(repository, "upsert_faith_case", fake_upsert)
    cases = [_ledger_case("a1"), _ledger_case("a2")]
    await ev._write_ledger(_gen_d(cases), {"chat_model": "glm-x"})
    assert [c for c, _ in calls] == cases          # 每例一次,顺序照报告
    assert all(jm == "glm-x" for _, jm in calls)   # judge_model 取 meta 的 chat 模型
    assert any("本轮判出 2 例,复发 1 例" in ln for ln in ev._LINES)


async def test_write_ledger_skips_when_generation_none(monkeypatch):
    from app.db import repository

    async def boom(case, judge_model=None):
        raise AssertionError("生成段未完成不该碰台账")
    monkeypatch.setattr(repository, "upsert_faith_case", boom)
    await ev._write_ledger(None, {"chat_model": "glm-x"})   # 不抛即过


async def test_write_ledger_failure_logs_one_warning(monkeypatch, caplog):
    from app.db import repository

    async def boom(case, judge_model=None):
        raise RuntimeError("MySQL 没起")
    monkeypatch.setattr(repository, "upsert_faith_case", boom)
    with caplog.at_level(logging.WARNING, logger="scripts.eval_ch04"):
        await ev._write_ledger(_gen_d([_ledger_case("a1")]), {"chat_model": "glm-x"})
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warns) == 1   # 整块失败只打一行,不逐例刷屏


async def test_main_report_lands_even_if_ledger_fails(monkeypatch, tmp_path):
    from app.db import repository
    gen = _gen_d([_ledger_case("a1")])

    async def fake_det(samples):
        return {}, {}, {}
    async def fake_gen(samples, hits):
        return gen
    async def boom(case, judge_model=None):
        raise RuntimeError("MySQL 没起")
    monkeypatch.setattr(ev, "_load", lambda: [{"id": "a1"}])
    monkeypatch.setattr(ev, "_deterministic", fake_det)
    monkeypatch.setattr(ev, "_generation", fake_gen)
    monkeypatch.setattr(ev, "_build_meta",
                        lambda samples: {"chat_model": "glm-x", "n_samples": 1, "errors": []})
    monkeypatch.setattr(ev, "_OUT_TXT", tmp_path / "r.txt")
    monkeypatch.setattr(ev, "_OUT_JSON", tmp_path / "r.json")
    monkeypatch.setattr(repository, "upsert_faith_case", boom)
    assert await ev.main() == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["generation"]["faithfulness_cases"] == gen["faithfulness_cases"]
    assert (tmp_path / "r.txt").exists()


async def test_main_writes_ledger_with_meta_chat_model(monkeypatch, tmp_path):
    from app.db import repository
    gen = _gen_d([_ledger_case("a1"), _ledger_case("a2")])
    calls = []

    async def fake_det(samples):
        return {}, {}, {}
    async def fake_gen(samples, hits):
        return gen
    async def fake_upsert(case, judge_model=None):
        calls.append((case["id"], judge_model))
        return {"id": len(calls), "recurred": False}
    monkeypatch.setattr(ev, "_load", lambda: [{"id": "a1"}, {"id": "a2"}])
    monkeypatch.setattr(ev, "_deterministic", fake_det)
    monkeypatch.setattr(ev, "_generation", fake_gen)
    monkeypatch.setattr(ev, "_build_meta",
                        lambda samples: {"chat_model": "glm-x", "n_samples": 2, "errors": []})
    monkeypatch.setattr(ev, "_OUT_TXT", tmp_path / "r.txt")
    monkeypatch.setattr(ev, "_OUT_JSON", tmp_path / "r.json")
    monkeypatch.setattr(repository, "upsert_faith_case", fake_upsert)
    assert await ev.main() == 0
    assert calls == [("a1", "glm-x"), ("a2", "glm-x")]
    # 汇总行进运行日志,随 txt 落盘
    assert "本轮判出 2 例,复发 0 例" in (tmp_path / "r.txt").read_text(encoding="utf-8")
