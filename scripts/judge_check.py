"""ch04 裁判校准回归:台账已处置个案当回归集,快照原样重放(不重检索、不重生成,只换裁判)。
人工处置翻译成标准答案:已解决→该判编造,无需解决→该判忠实,未解决跳过。
运行:PYTHONPATH=. uv run python scripts/judge_check.py(make judge-check)
逐条对照 + 一致率;任一不一致或零成功对照 → 退出码 1。"""
import asyncio
import sys

from sqlalchemy import select

import app.db.base as db
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT
from app.db.models import FaithCase
from scripts.eval_ch04 import Faithfulness

EXPECTED = {"已解决": False, "无需解决": True}


def assemble_evidence(citations: list[dict]) -> str:
    """快照顺序原样(已首尾组装,不重排);格式与 eval_ch04 逐字一致;老快照缺 n 按序补。"""
    numbered = [{**c, "n": c["n"] if c.get("n") is not None else i + 1}
                for i, c in enumerate(citations)]
    return "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in numbered) or "(无证据)"


async def load_cases() -> list[FaithCase]:
    async with db.async_session() as s:
        rows = (await s.execute(
            select(FaithCase).where(FaithCase.status.in_(tuple(EXPECTED)))
            .order_by(FaithCase.id)
        )).scalars()
        return [r for r in rows if r.citations]


def _judge_chain():
    # 同 eval_ch04._judge_chain(Ruling 17 + T20):temperature=0 + 关 thinking + 宽容解析
    model = get_chat_model(temperature=0, extra_body={"thinking": {"type": "disabled"}})
    return FAITHFULNESS_PROMPT | model.with_structured_output(
        Faithfulness, method="function_calling", include_raw=True)


async def _judge_once(payload: dict) -> Faithfulness | None:
    out = await _judge_chain().ainvoke(payload)
    r = out.get("parsed")
    if r is None:
        tool_calls = getattr(out.get("raw"), "tool_calls", None) or []
        if tool_calls:
            try:
                r = Faithfulness.model_validate(tool_calls[0]["args"])   # 忽略错误工具名
            except Exception:
                r = None
    return r


async def judge_answer(evidence: str, answer: str) -> Faithfulness | None:
    """上游偶发回空 completion(解析成 None)/异常 → 重试一次;仍失败回 None(调用失败)。"""
    payload = {"evidence": evidence, "answer": answer}
    for _ in range(2):
        try:
            r = await _judge_once(payload)
        except Exception:
            r = None
        if r is not None:
            return r
    return None


def _label(v: bool) -> str:
    return "忠实" if v else "编造"


async def replay(cases: list) -> int:
    matched = mismatched = failed = 0
    for row in cases:
        expected = EXPECTED[row.status]
        r = await judge_answer(assemble_evidence(row.citations), row.answer)
        if r is None:
            failed += 1
            print(f"{row.eval_id} | 期望={_label(expected)} | 调用失败(重试后仍无结果,不计入一致率)")
            continue
        got = bool(r.faithful)
        ok = got == expected
        matched += ok
        mismatched += not ok
        print(f"{row.eval_id} | 期望={_label(expected)} | 判出={_label(got)} | "
              f"{'一致' if ok else '不一致'} | {r.reason or ''}")
    total = matched + mismatched
    rate = f"{matched}/{total}" if total else "0/0(无成功对照)"
    print(f"一致率:{rate} | 调用失败 {failed} 例 | 可考个案 {len(cases)} 例")
    return 1 if mismatched or total == 0 else 0


async def main() -> int:
    cases = await load_cases()
    print(f"judge-check:台账已处置且有证据快照的个案 {len(cases)} 例(temperature=0 重放)")
    return await replay(cases)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
