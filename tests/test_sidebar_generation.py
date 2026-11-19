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

MAX_TOKENS_LABEL = "Задать свой лимит ответа (max_tokens)"


def _max_tokens_checkbox(at):
    return next(c for c in at.checkbox if c.label == MAX_TOKENS_LABEL)


def test_manual_max_tokens_checkbox_survives_reruns():
    at = AppTest.from_string(GENERATION_APP, default_timeout=30).run()
    assert _max_tokens_checkbox(at).value is False

    _max_tokens_checkbox(at).check()
    at.run()

    assert _max_tokens_checkbox(at).value is True

    # Повторный rerun не должен сбрасывать ручную настройку для каталоговой модели.
    at.run()
    assert _max_tokens_checkbox(at).value is True
