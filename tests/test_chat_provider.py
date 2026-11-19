import json

from streamlit.testing.v1 import AppTest

from adl_smart_search.config import AppConfig
from adl_smart_search.ui.settings import apply_llm_settings

APP = "app.py"


def _prepare(tmp_path, monkeypatch, *, chat_provider: str | None = None):
    monkeypatch.setenv("ADL_INDEX_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ADL_LOCAL_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("ADL_DEV_MODE", "0")
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("ADL_ENV_FILE", str(empty_env))
    if chat_provider is not None:
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "llm.json").write_text(
            json.dumps({"chat_provider": chat_provider}), encoding="utf-8"
        )


def test_apply_llm_settings_reads_chat_provider():
    cfg = AppConfig(adl_dir=".")

    apply_llm_settings(cfg, {"chat_provider": "gigachat"})

    assert cfg.chat_provider == "gigachat"


def test_default_provider_comes_from_llm_json(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch, chat_provider="gigachat")

    at = AppTest.from_file(APP, default_timeout=120).run()

    assert at.session_state["chat_provider"] == "gigachat"
    assert at.session_state["chat_provider_saved"] == "gigachat"
    assert at.session_state["chat_provider_choice"] == "gigachat"


def test_session_choice_survives_widget_cleanup(tmp_path, monkeypatch):
    """Радио не отрисовалось — Streamlit вычистил ключ; выбор не теряем."""
    _prepare(tmp_path, monkeypatch, chat_provider="corp_async")

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.session_state["chat_provider_saved"] = "gigachat"
    at.session_state["chat_provider_choice"] = None
    at.run()

    assert at.session_state["chat_provider_choice"] == "gigachat"
    # Значение по умолчанию из конфига при этом не меняется.
    assert at.session_state["chat_provider"] == "corp_async"
