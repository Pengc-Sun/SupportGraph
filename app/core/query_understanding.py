import logging

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class QueryRewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


def get_rewriter() -> Runnable:
    # DeepSeek 官方 deepseek-flash 默认 thinking,不支持强制 tool_choice / json_schema;
    # 结构化输出走 function_calling 需关 thinking(同 app/api/extract.py)。
    # include_raw=True(Ruling 17):模型间歇输出错误工具名,拿 raw 按 args 宽容解析。
    model = get_chat_model(extra_body={"thinking": {"type": "disabled"}})
    return QUERY_REWRITE_PROMPT | model.with_structured_output(
        QueryRewrite, method="function_calling", include_raw=True
    )


async def understand(query: str) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象),避开上游 502。

    宽容解析(Ruling 17):parsed=None 时忽略工具名,直接校验 raw.tool_calls[0]["args"];
    彻底失败或上游异常 → 降级用原 query,检索仍可继续。
    """
    try:
        out = await get_rewriter().ainvoke({"query": query})
        r: QueryRewrite | None = out["parsed"]
        if r is None:
            raw = out.get("raw")
            tool_calls = getattr(raw, "tool_calls", None) or []
            if tool_calls:
                try:
                    r = QueryRewrite.model_validate(tool_calls[0]["args"])
                except Exception:
                    logger.warning("Query 改写 args 宽容解析失败: %s", query, exc_info=True)
    except Exception:
        logger.warning("Query 改写上游异常,降级为原 query: %s", query, exc_info=True)
        return {"standard": query, "expanded": []}
    if r is None:
        logger.warning("Query 改写解析失败,降级为原 query: %s", query)
        return {"standard": query, "expanded": []}
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}
