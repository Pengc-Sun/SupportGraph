import logging

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import SELF_CHECK_PROMPT

logger = logging.getLogger(__name__)


class SelfCheck(BaseModel):
    useful: bool = Field(description="证据是否足以回答")
    reason: str = Field(default="", description="判断依据")


def _chain():
    # DeepSeek 官方 deepseek-flash 默认 thinking,不支持强制 tool_choice / json_schema;
    # 结构化输出走 function_calling 需关 thinking(同 app/core/query_understanding.py)。
    # include_raw=True(Ruling 17):模型间歇输出错误工具名(Tool / \_Check),
    # langchain 按名匹配会解析失败,拿 raw 自行按 args 宽容解析。
    model = get_chat_model(extra_body={"thinking": {"type": "disabled"}})
    return SELF_CHECK_PROMPT | model.with_structured_output(
        SelfCheck, method="function_calling", include_raw=True
    )


async def check_sufficient(query: str, evidence_texts: list[str]) -> dict:
    """生成前证据充分性自评。扁平字段 useful/reason(避开上游嵌套数组 502)。

    fail-open(Ruling 16):上游异常或解析彻底失败时放行(useful=True)——
    机械 rerank 门已过,基础设施故障不构成证据不足;生成 prompt 的拒答规则+负面知识兜底。
    宽容解析(Ruling 17):parsed=None 时忽略工具名,直接校验 raw.tool_calls[0]["args"]。
    """
    evidence = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_texts)) or "(无证据)"
    try:
        out = await _chain().ainvoke({"query": query, "evidence": evidence})
        r: SelfCheck | None = out["parsed"]
        if r is None:
            raw = out.get("raw")
            tool_calls = getattr(raw, "tool_calls", None) or []
            if tool_calls:
                try:
                    r = SelfCheck.model_validate(tool_calls[0]["args"])
                except Exception:
                    logger.warning("证据自评 args 宽容解析失败: %s", query, exc_info=True)
    except Exception:
        logger.warning("证据自评上游异常,fail-open 放行: %s", query, exc_info=True)
        return {"useful": True, "reason": ""}
    if r is None:
        logger.warning("证据自评解析失败,fail-open 放行: %s", query)
        return {"useful": True, "reason": ""}
    return {"useful": bool(r.useful), "reason": r.reason or ""}
