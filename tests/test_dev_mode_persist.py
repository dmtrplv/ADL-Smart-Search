import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _prepare(tmp_path, monkeypatch):
    monkeypatch.setenv("ADL_INDEX_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ADL_LOCAL_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("ADL_DEV_MODE", "0")
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("ADL_ENV_FILE", str(empty_env))


def test_chat_provider_defaults_to_corp(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)

    at = AppTest.from_file(APP, default_timeout=120).run()

    assert at.session_state["chat_provider_choice"] == "corp_async"
    assert at.session_state["api_kind"] == "corp_async"


def test_dev_edit_does_not_write_config_until_save(tmp_path, monkeypatch):
    """Изменения живут в сессии: без «Сохранить» файл не появляется."""
    _prepare(tmp_path, monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.toggle[0].set_value(True).run()
    next(s for s in at.slider if s.label == "Вес точного совпадения").set_value(1.2).run()

    assert at.session_state["keyword_bonus"] == 1.2
    assert not (tmp_path / "config" / "settings.json").exists()


def test_save_search_block_writes_config(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.toggle[0].set_value(True).run()
    next(s for s in at.slider if s.label == "Вес точного совпадения").set_value(1.2).run()
    assert any("несохранённые" in caption.value for caption in at.caption)

    at.button(key="save_search_settings_btn").click().run()

    assert not any("несохранённые" in caption.value for caption in at.caption)
    saved = json.loads(
        (tmp_path / "config" / "settings.json").read_text(encoding="utf-8")
    )
    assert saved["keyword_bonus"] == 1.2


def test_reset_search_block_restores_config_values(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.json").write_text(
        json.dumps({"keyword_bonus": 0.9}), encoding="utf-8"
    )
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.toggle[0].set_value(True).run()
    next(s for s in at.slider if s.label == "Вес точного совпадения").set_value(1.2).run()
    at.button(key="reset_search_settings_btn").click().run()

    assert at.session_state["keyword_bonus"] == 0.9
    saved = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved["keyword_bonus"] == 0.9


def test_reset_all_button_restores_all_blocks(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.json").write_text(
        json.dumps({"keyword_bonus": 0.9}), encoding="utf-8"
    )
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.toggle[0].set_value(True).run()
    next(s for s in at.slider if s.label == "Вес точного совпадения").set_value(1.2).run()
    assert any("несохранённые" in caption.value for caption in at.caption)

    at.button(key="reset_all_settings_btn").click().run()

    assert at.session_state["keyword_bonus"] == 0.9
    saved = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved["keyword_bonus"] == 0.9
    assert not any("несохранённые" in caption.value for caption in at.caption)


def test_context_budget_saved_via_button(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.json").write_text(
        json.dumps({"context_tokens_archetypes": 1000, "context_tokens_fields": 1000}),
        encoding="utf-8",
    )
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.toggle[0].set_value(True).run()
    at.number_input(key="context_tokens_archetypes").set_value(10000).run()
    at.button(key="save_search_settings_btn").click().run()

    saved = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved["context_tokens_archetypes"] == 10000


def test_reset_search_to_factory_defaults(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.json").write_text(
        json.dumps({"keyword_bonus": 0.9}), encoding="utf-8"
    )
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.toggle[0].set_value(True).run()
    assert at.session_state["keyword_bonus"] == 0.9

    at.button(key="reset_search_defaults_btn").click().run()

    # Значение из кода, а не из конфига; файл не тронут и видно, что не сохранено.
    assert at.session_state["keyword_bonus"] == 0.5
    saved = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved["keyword_bonus"] == 0.9
    assert any("несохранённые" in caption.value for caption in at.caption)


def test_thinking_toggle_only_for_supporting_models(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    catalog = [
        {
            "model_name": "Plain",
            "provider": "p",
            "model": "m",
            "model_kwargs": {},
            "context_window": 8192,
        },
        {
            "model_name": "Thinker",
            "provider": "p",
            "model": "m2",
            "model_kwargs": {"chat_template_kwargs": {"enable_thinking": True}},
            "context_window": 8192,
        },
    ]
    (config_dir / "corp_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle[0].set_value(True).run()

    next(s for s in at.selectbox if s.label == "Модель").set_value("Plain").run()
    assert not any(t.label == "Размышления (thinking)" for t in at.toggle)
    assert at.session_state["corp_use_enable_thinking"] is False

    next(s for s in at.selectbox if s.label == "Модель").set_value("Thinker").run()
    thinking = next(t for t in at.toggle if t.label == "Размышления (thinking)")
    assert thinking.value is True
    assert at.session_state["corp_use_enable_thinking"] is True

    thinking.set_value(False).run()
    assert at.session_state["corp_enable_thinking"] is False
    assert at.session_state["corp_use_enable_thinking"] is True


def test_save_llm_block_writes_config(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.toggle[0].set_value(True).run()
    next(s for s in at.selectbox if s.label == "GigaChat model").set_value("GigaChat-2-Pro")
    at.button(key="save_llm_settings_btn").click().run()

    llm_file = tmp_path / "config" / "llm.json"
    assert llm_file.exists()
    assert json.loads(llm_file.read_text(encoding="utf-8"))["gigachat_model"] == "GigaChat-2-Pro"


def test_save_llm_block_keeps_enable_thinking_mode(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    catalog = [
        {
            "model_name": "Thinking Model",
            "provider": "test-provider",
            "model": "test-model",
            "model_kwargs": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "context_window": 8192,
        }
    ]
    (config_dir / "corp_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle[0].set_value(True).run()
    next(s for s in at.selectbox if s.label == "Модель").set_value("Thinking Model").run()
    at.button(key="save_llm_settings_btn").click().run()

    llm_file = tmp_path / "config" / "llm.json"
    assert llm_file.exists()
    saved = json.loads(llm_file.read_text(encoding="utf-8"))
    assert saved["corp_use_enable_thinking"] is True
    assert saved["corp_enable_thinking"] is False


def test_catalog_model_locks_model_fields_and_prefills_generation(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    catalog = [
        {
            "model_name": "Test Model",
            "provider": "test-provider",
            "model": "test-model",
            "model_kwargs": {
                "cluster": "http://cluster",
                "service_url": "test",
                "max_tokens": 2048,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 0.5,
                "repetition_penalty": 1.15,
                "chat_template_kwargs": {
                    "temperature": 0.4,
                    "enable_thinking": False,
                },
            },
            "context_window": 8192,
        }
    ]
    (config_dir / "corp_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle[0].set_value(True).run()
    next(s for s in at.selectbox if s.label == "Модель").set_value("Test Model").run()

    provider = next(s for s in at.selectbox if s.label == "API provider")
    api_model = next(t for t in at.text_input if t.label == "API model")
    assert provider.disabled is True
    assert api_model.disabled is True

    temp_use = next(c for c in at.checkbox if c.label == "Задать temperature")
    temp = next(s for s in at.slider if s.label == "temperature")
    assert temp_use.disabled is True
    assert temp_use.value is True
    assert temp.disabled is True
    assert temp.value == 0.4

    top_p = next(s for s in at.slider if s.label == "top_p")
    assert top_p.disabled is True
    assert top_p.value == 0.8

    thinking = next(t for t in at.toggle if t.label == "Размышления (thinking)")
    assert thinking.value is False
    assert thinking.disabled is False


def test_aux_model_list_mixes_aiedu_and_mosgpt(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    catalog = [
        {
            "model_name": "Main Aiedu",
            "provider": "aiedu-v2",
            "model": "/model",
            "model_kwargs": {"service_url": "main-aiedu"},
            "context_window": 8192,
        },
        {
            "model_name": "Aux Mosgpt",
            "provider": "mosgpt",
            "model": "aux-mosgpt",
            "model_kwargs": {
                "chat_template_kwargs": {
                    "temperature": 0.3,
                    "enable_thinking": False,
                },
                "top_p": 0.5,
            },
            "context_window": 8192,
        },
    ]
    (config_dir / "corp_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle[0].set_value(True).run()
    next(s for s in at.selectbox if s.label == "Модель").set_value("Main Aiedu").run()
    next(
        c for c in at.checkbox if c.label == "Отдельная модель для перевода и расширения"
    ).set_value(True).run()

    aux = next(s for s in at.selectbox if s.label == "Модель (перевод/расширение)")
    assert "Main Aiedu" in aux.options
    assert "Aux Mosgpt" in aux.options

    aux.set_value("Aux Mosgpt").run()
    assert at.session_state["corp_aux_provider"] == "mosgpt"
    assert at.session_state["corp_aux_use_temperature"] is True
    assert at.session_state["corp_aux_temperature"] == 0.3
    assert at.session_state["corp_aux_use_top_p"] is True
    assert at.session_state["corp_aux_top_p"] == 0.5
    assert at.session_state["corp_aux_use_enable_thinking"] is True
    assert at.session_state["corp_aux_enable_thinking"] is False


def test_catalog_locks_only_generation_params_present_in_json(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    catalog = [
        {
            "model_name": "Only Service URL",
            "provider": "aiedu-v2",
            "model": "/model",
            "model_kwargs": {"service_url": "qwen-test"},
            "context_window": 8192,
        }
    ]
    (config_dir / "corp_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle[0].set_value(True).run()
    next(s for s in at.selectbox if s.label == "Модель").set_value("Only Service URL").run()

    assert next(c for c in at.checkbox if c.label == "Задать temperature").disabled is False
    assert next(c for c in at.checkbox if c.label == "Задать top_p").disabled is False
    assert (
        next(c for c in at.checkbox if c.label == "Задать свой лимит ответа (max_tokens)").disabled
        is False
    )


def test_user_mode_does_not_write_config(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)

    AppTest.from_file(APP, default_timeout=120).run()

    assert not (tmp_path / "config" / "llm.json").exists()
    assert not (tmp_path / "config" / "settings.json").exists()


def test_admin_exe_dev_save_writes_repo_config(tmp_path, monkeypatch):
    """dev-режим exe пишет в тот же config/, что читают пользователи."""
    _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv("ADL_APP_MODE", "exe")
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.toggle[0].set_value(True).run()
    next(s for s in at.slider if s.label == "Вес точного совпадения").set_value(1.2).run()
    at.button(key="save_search_settings_btn").click().run()

    saved = json.loads(
        (tmp_path / "config" / "settings.json").read_text(encoding="utf-8")
    )
    assert saved["keyword_bonus"] == 1.2
    assert not (tmp_path / "local" / "config" / "settings.overrides.json").exists()


def test_exit_button_is_rendered(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)

    at = AppTest.from_file(APP, default_timeout=120).run()

    assert any(button.label == "Завершить работу" for button in at.button)
