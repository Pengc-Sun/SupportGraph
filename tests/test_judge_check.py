"""T20 judge-check 回归工具:快照原样重放、处置→标准答案映射、失败不计一致。
确定性逻辑全 mock;筛选用 mewhelp_test(db_session_factory/db_clean)。"""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import scripts.judge_check as jc
from app.db import repository

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _cit(n=None, question="退货政策是什么", answer="支持 7 天无理由退货", hid="h1"):
    c = {"id": hid, "section_path": "商品与购物 FAQ / 退货政策是什么",
         "question": question, "answer": answer, "content_type": "faq"}
    if n is not None:
        c["n"] = n
    return c


def _row(eid, status, answer="答案[1]", citations=None):
    return SimpleNamespace(eval_id=eid, status=status, answer=answer,
                           citations=[_cit(n=1)] if citations is None else citations)


def _raw_result(parsed=None, args=None, tool_name="Tool"):
    raw = AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": "x"}]) \
        if args is not None else AIMessage(content="")
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


# ---------- 证据拼回:与 eval_ch04 逐字同格式,快照顺序原样不重排 ----------

async def test_assemble_evidence_verbatim_format_and_stored_order():
    citations = [_cit(n=1, question="q1", answer="a1"),
                 _cit(n=3, question="q3", answer="a3", hid="h3"),
                 _cit(n=2, question="q2", answer="a2", hid="h2")]
    expected = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    assert jc.assemble_evidence(citations) == expected
    assert jc.assemble_evidence(citations) == "[1] q1: a1\n[3] q3: a3\n[2] q2: a2"


async def test_assemble_evidence_missing_n_filled_by_index():
    citations = [_cit(question="q1", answer="a1"),
                 _cit(n=5, question="q2", answer="a2", hid="h2"),
                 _cit(question="q3", answer="a3", hid="h3")]
    assert jc.assemble_evidence(citations) == "[1] q1: a1\n[5] q2: a2\n[3] q3: a3"
    assert jc.assemble_evidence([{"n": None, "question": "q", "answer": "a"}]) == "[1] q: a"


async def test_assemble_evidence_empty_gives_placeholder():
    assert jc.assemble_evidence([]) == "(无证据)"


# ---------- 处置→标准答案映射 ----------

async def test_disposition_to_expected_mapping():
    assert jc.EXPECTED == {"已解决": False, "无需解决": True}   # 未解决不在映射里 → 跳过


# ---------- 能考个案的筛选:已处置 + citations 非空 ----------

async def test_load_cases_filters_status_and_citations(db_session_factory, db_clean):
    base = {"bucket": "A_policy", "query": "q", "strategy": "hybrid_rerank",
            "answer": "a", "reason": "r"}
    ids = {}
    for cid, citations in [("a1", [_cit(n=1)]), ("a2", [_cit(n=1)]), ("a3", [_cit(n=1)]),
                           ("a4", None), ("a5", [])]:
        ids[cid] = (await repository.upsert_faith_case(
            {**base, "id": cid, "citations": citations}, judge_model=None))["id"]
    await repository.set_faith_case_status(ids["a2"], "已解决", "确认编造,已修")
    await repository.set_faith_case_status(ids["a3"], "无需解决", "裁判误判")
    await repository.set_faith_case_status(ids["a4"], "已解决", "确认编造,已修")
    await repository.set_faith_case_status(ids["a5"], "无需解决", "裁判误判")
    cases = await jc.load_cases()
    assert [c.eval_id for c in cases] == ["a2", "a3"]   # a1 未解决、a4/a5 无快照,均跳过


# ---------- 裁判调用:temperature=0 + Ruling 17 宽容解析 ----------

async def test_judge_chain_temperature_zero(monkeypatch):
    seen = {}
    real = jc.get_chat_model

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    monkeypatch.setattr(jc, "get_chat_model", spy)
    jc._judge_chain()
    assert seen["temperature"] == 0
    assert seen["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_judge_once_tolerant_parse_ignores_tool_name(monkeypatch):
    class Chain:
        async def ainvoke(self, payload):
            return _raw_result(args={"faithful": False, "reason": "凭空数值"})
    monkeypatch.setattr(jc, "_judge_chain", lambda: Chain())
    r = await jc._judge_once({"evidence": "e", "answer": "a"})
    assert r is not None and r.faithful is False and r.reason == "凭空数值"


async def test_judge_once_parsed_and_garbage(monkeypatch):
    class OkChain:
        async def ainvoke(self, payload):
            return _raw_result(parsed=jc.Faithfulness(faithful=True, reason="ok"))
    monkeypatch.setattr(jc, "_judge_chain", lambda: OkChain())
    r = await jc._judge_once({"evidence": "e", "answer": "a"})
    assert r.faithful is True

    class BadChain:
        async def ainvoke(self, payload):
            return _raw_result(args={"faithful": "不是布尔"})
    monkeypatch.setattr(jc, "_judge_chain", lambda: BadChain())
    assert await jc._judge_once({"evidence": "e", "answer": "a"}) is None


# ---------- 回空重试一次;仍失败 → None(调用失败) ----------

async def test_judge_retries_once_on_empty_completion(monkeypatch):
    calls = []

    class Chain:
        async def ainvoke(self, payload):
            calls.append(payload)
            if len(calls) == 1:
                return _raw_result()   # 空 completion,解析成 None
            return _raw_result(parsed=jc.Faithfulness(faithful=True, reason="ok"))
    monkeypatch.setattr(jc, "_judge_chain", lambda: Chain())
    r = await jc.judge_answer("e", "a")
    assert r is not None and r.faithful is True
    assert calls == [{"evidence": "e", "answer": "a"}] * 2   # 恰好重试一次


async def test_judge_failure_after_one_retry(monkeypatch):
    calls = []

    class Chain:
        async def ainvoke(self, payload):
            calls.append(payload)
            return _raw_result()
    monkeypatch.setattr(jc, "_judge_chain", lambda: Chain())
    assert await jc.judge_answer("e", "a") is None
    assert len(calls) == 2   # 不无限重试


async def test_judge_exception_also_retried_once(monkeypatch):
    calls = []

    class Chain:
        async def ainvoke(self, payload):
            calls.append(payload)
            if len(calls) == 1:
                raise RuntimeError("upstream 502")
            return _raw_result(parsed=jc.Faithfulness(faithful=False, reason="编造"))
    monkeypatch.setattr(jc, "_judge_chain", lambda: Chain())
    r = await jc.judge_answer("e", "a")
    assert r is not None and r.faithful is False and len(calls) == 2

    class DeadChain:
        async def ainvoke(self, payload):
            raise RuntimeError("upstream down")
    monkeypatch.setattr(jc, "_judge_chain", lambda: DeadChain())
    assert await jc.judge_answer("e", "a") is None


# ---------- replay:逐条对照 + 一致率 + 退出码 ----------

async def test_replay_all_match_exit_0_and_verbatim_evidence(monkeypatch, capsys):
    seen = []

    async def fake_judge(evidence, answer):
        seen.append((evidence, answer))
        return jc.Faithfulness(faithful={"编造答案": False, "忠实答案": True}[answer], reason="依据")
    monkeypatch.setattr(jc, "judge_answer", fake_judge)
    rows = [_row("a1", "已解决", answer="编造答案"), _row("a2", "无需解决", answer="忠实答案")]
    assert await jc.replay(rows) == 0
    assert seen == [("[1] 退货政策是什么: 支持 7 天无理由退货", "编造答案"),
                    ("[1] 退货政策是什么: 支持 7 天无理由退货", "忠实答案")]
    out = capsys.readouterr().out
    assert "a1" in out and "a2" in out and "2/2" in out


async def test_replay_mismatch_exit_1_with_reason(monkeypatch, capsys):
    async def fake_judge(evidence, answer):
        return jc.Faithfulness(faithful=True, reason="误判为忠实")
    monkeypatch.setattr(jc, "judge_answer", fake_judge)
    rows = [_row("a1", "已解决"), _row("a2", "无需解决")]
    assert await jc.replay(rows) == 1   # a1 期望编造却判忠实 → 不一致
    out = capsys.readouterr().out
    assert "a1" in out and "不一致" in out and "误判为忠实" in out and "1/2" in out


async def test_replay_call_failure_excluded_from_denominator(monkeypatch, capsys):
    async def fake_judge(evidence, answer):
        if answer == "失败题":
            return None
        return jc.Faithfulness(faithful=False, reason="编造")
    monkeypatch.setattr(jc, "judge_answer", fake_judge)
    rows = [_row("a1", "已解决", answer="编造答案"), _row("a2", "已解决", answer="失败题")]
    assert await jc.replay(rows) == 0   # 失败不算不一致,1/1 全一致
    out = capsys.readouterr().out
    assert "调用失败" in out and "a2" in out and "1/1" in out


async def test_replay_zero_successful_comparisons_exit_1(monkeypatch, capsys):
    async def fake_judge(evidence, answer):
        return None
    monkeypatch.setattr(jc, "judge_answer", fake_judge)
    assert await jc.replay([_row("a1", "已解决")]) == 1
    assert "调用失败" in capsys.readouterr().out
    assert await jc.replay([]) == 1   # 没有可考个案也算失败
