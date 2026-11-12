from streamlit.testing.v1 import AppTest

GENERATION_APP = """
from adl_smart_search.config import AppConfig
from adl_smart_search.ui.sidebar import _render_corp_generation_settings

_render_corp_generation_settings(
    AppConfig(adl_dir="."),
    title="Параметры генерации",
    key_prefix="corp",
    raw_json="{}",
    from_catalog=True,
    catalog_model="catalog-model",
)
"""

# Порядок чекбоксов в блоке: temperature, max_tokens, top_p, top_k,
# enable_thinking, presence_penalty, repetition_penalty.
MAX_TOKENS_INDEX = 1


def test_manual_max_tokens_checkbox_survives_reruns():
    at = AppTest.from_string(GENERATION_APP, default_timeout=30).run()
    assert at.checkbox[MAX_TOKENS_INDEX].value is False

    at.checkbox[MAX_TOKENS_INDEX].check()
    at.run()

    assert at.checkbox[MAX_TOKENS_INDEX].value is True

    # Повторный rerun не должен сбрасывать ручную настройку для каталоговой модели.
    at.run()
    assert at.checkbox[MAX_TOKENS_INDEX].value is True
