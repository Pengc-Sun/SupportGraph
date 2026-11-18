import logging

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


def get_rewriter() -> Runnable:
    # DeepSeek 官方 deepseek-flash 默认 thinking,不支持强制 tool_choice / json_schema;
    # 结构化输出走 function_calling 需关 thinking(同 app/api/extract.py)。
    model = get_chat_model(extra_body={"thinking": {"type": "disabled"}})
    return QUERY_REWRITE_PROMPT | model.with_structured_output(_Rewrite, method="function_calling")


async def understand(query: str) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象),避开上游 502。"""
    r: _Rewrite | None = await get_rewriter().ainvoke({"query": query})
    if r is None:
        # with_structured_output 解析失败可能返回 None 而非抛异常;降级用原 query,检索仍可继续
        logger.warning("Query 改写解析失败,降级为原 query: %s", query)
        return {"standard": query, "expanded": []}
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}
