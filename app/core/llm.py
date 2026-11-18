from langchain_openai import ChatOpenAI

from app.config import settings

def get_chat_model(streaming: bool = False, extra_body: dict | None = None,
                   temperature: float = 0.3) -> ChatOpenAI:
    """直连聊天上游。地址、模型名、密钥全从 .env 来,换上游不动这里。
    temperature 默认 0.3(生成);裁判显式传 0 去抖(T20)。"""
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=temperature,
        extra_body=extra_body,
    )
