import pytest
from streamlit.testing.v1 import AppTest

from adl_smart_search.prompts import PromptConfigError
from adl_smart_search.query_expand import (
    build_translate_context_prompt,
    translate_context_to_ru,
)


class _Provider:
    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append((prompt, system or ""))
        return "ПЕРЕВОД"


def test_translate_context_uses_system_and_text_template():
    provider = _Provider()

    out = translate_context_to_ru(
        "Weight (at0004) | DV_QUANTITY",
        provider,
        system="SYSTEM",
        user_template="Текст:\n{text}",
    )

    assert out == "ПЕРЕВОД"
    assert provider.prompts[0][1] == "SYSTEM"
    assert "Weight (at0004) | DV_QUANTITY" in provider.prompts[0][0]


def test_translate_context_trims_long_text():
    provider = _Provider()

    translate_context_to_ru(
        "x" * 20_000,
        provider,
        system="SYSTEM",
        user_template="{text}",
        max_chars=2000,
    )

    assert len(provider.prompts[0][0]) <= 2000


def test_translate_context_requires_prompts():
    with pytest.raises(PromptConfigError):
        translate_context_to_ru("text", _Provider(), system="", user_template="{text}")

    with pytest.raises(PromptConfigError):
        translate_context_to_ru("text", _Provider(), system="SYSTEM", user_template="")

    with pytest.raises(PromptConfigError):
        translate_context_to_ru(
            "text", _Provider(), system="SYSTEM", user_template="Без плейсхолдера"
        )


def test_build_translate_context_prompt_keeps_other_braces():
    prompt = build_translate_context_prompt(
        'at0004 = <"x">', "Переведи: {text}"
    )

    assert prompt == 'Переведи: at0004 = <"x">'


_TRANSLATION_APP = """
from types import SimpleNamespace

import streamlit as st

from adl_smart_search.ui.search_tab import render_context_with_translation


class _Provider:
    def generate(self, prompt, system=None):
        st.session_state["calls"] = int(st.session_state.get("calls", 0)) + 1
        return "ПЕРЕВОД Weight (at0004)"


ctx = SimpleNamespace(llm_aux_provider=_Provider(), llm_setup_msg="")
render_context_with_translation(
    "Weight (at0004) | DV_QUANTITY",
    ["weight"],
    [],
    cache_key="test-context",
    ctx=ctx,
    prompts={"ui_translate": "SYSTEM", "ui_translate_user": "{text}"},
)
"""


def test_translation_button_caches_result_between_reruns():
    at = AppTest.from_string(_TRANSLATION_APP, default_timeout=30).run()
    assert [b.label for b in at.button] == ["Перевести на русский"]

    at.button[0].click().run()

    cache = at.session_state["ui_translation_cache"]
    assert len(cache) == 1
    assert "ПЕРЕВОД Weight (at0004)" in next(iter(cache.values()))
    assert at.session_state["calls"] == 1

    at.run()
    assert at.session_state["calls"] == 1

