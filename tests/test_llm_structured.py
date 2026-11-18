"""T20 尺子去抖:get_chat_model 的 temperature 默认 0.3、显式透传(裁判传 0)。"""
from app.core.llm import get_chat_model


def test_default_temperature_stays_0_3():
    assert get_chat_model().temperature == 0.3


def test_explicit_temperature_passthrough():
    assert get_chat_model(temperature=0).temperature == 0.0
    assert get_chat_model(temperature=0.7).temperature == 0.7


def test_temperature_composes_with_streaming_and_extra_body():
    m = get_chat_model(streaming=True, extra_body={"thinking": {"type": "disabled"}},
                       temperature=0)
    assert m.temperature == 0.0 and m.streaming is True
    assert m.extra_body == {"thinking": {"type": "disabled"}}
