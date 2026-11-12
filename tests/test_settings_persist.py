import json
import sys

import pytest

from adl_smart_search.config import AppConfig, from_env
from adl_smart_search.ui.settings import (
    LLM_SETTINGS_TYPES,
    SEARCH_SETTINGS_TYPES,
    apply_llm_settings,
    apply_search_settings,
    is_dev_mode,
    llm_settings_path,
    load_llm_settings,
    load_settings,
    save_llm_settings,
    save_settings,
    settings_path,
)


def test_search_settings_roundtrip(tmp_path):
    index_dir = str(tmp_path / "data")
    payload = {
        "adl_dir": r"C:\adl",
        "index_dir": index_dir,
        "ckm_adl_dir": r"C:\ckm",
        "auto_strictness": 0.4,
        "keyword_bonus": 0.8,
        "max_results": 12,
        "sim_floor": 0.25,
        "sim_floor_pure": 0.5,
        "enable_misuse_penalty": False,
    }
    save_settings(index_dir, payload)

    loaded = load_settings(index_dir)

    for key, value in payload.items():
        assert loaded[key] == value
    assert settings_path(index_dir).endswith("settings.json")


def test_apply_search_settings_overrides_cfg_and_coerces():
    cfg = AppConfig(adl_dir=".")
    apply_search_settings(
        cfg,
        {
            "auto_strictness": "0.33",
            "max_results": "7",
            "keyword_bonus": "0.75",
            "enable_misuse_penalty": "no",
            "sim_floor": "0.2",
        },
    )

    assert cfg.auto_strictness == 0.33
    assert cfg.max_results == 7
    assert cfg.keyword_bonus == 0.75
    assert cfg.enable_misuse_penalty is False
    assert cfg.sim_floor == 0.2


def test_apply_search_settings_ignores_bad_values():
    cfg = AppConfig(adl_dir=".")
    apply_search_settings(cfg, {"max_results": "not-a-number", "auto_strictness": 0.5})

    assert cfg.max_results == 10  # значение по умолчанию не испорчено
    assert cfg.auto_strictness == 0.5


def test_context_budget_settings_have_defaults_and_types():
    cfg = AppConfig(adl_dir=".")

    assert cfg.context_tokens_archetypes == 10000
    assert cfg.context_tokens_fields == 10000
    assert SEARCH_SETTINGS_TYPES["context_tokens_archetypes"] is int
    assert SEARCH_SETTINGS_TYPES["context_tokens_fields"] is int


def test_llm_settings_roundtrip(tmp_path):
    index_dir = str(tmp_path / "data")
    payload = {
        "llm_mode": "api",
        "api_kind": "gigachat",
        "api_base_url": "https://example.test/async/v1",
        "api_model": "test-model",
        "openai_base_url": "https://openai.test/v1",
        "openai_model": "openai-model",
        "corp_provider": "test-provider",
        "corp_use_temperature": True,
        "corp_temperature": 0.35,
        "corp_poll_interval_s": 0.5,
        "ollama_model": "test-ollama",
        "corp_model_name": "catalog-model",
        "corp_context_window": 16384,
        "gigachat_model": "GigaChat-2",
        "gigachat_scope": "GIGACHAT_API_PERS",
        "gigachat_temperature": 0.3,
        "gigachat_max_tokens": 2048,
        "gigachat_context_window": 32000,
    }

    assert save_llm_settings(index_dir, payload) is True
    assert load_llm_settings(index_dir) == payload
    assert llm_settings_path(index_dir).endswith("llm.json")
    # Повторная запись без изменений не трогает файл.
    assert save_llm_settings(index_dir, payload) is False


def test_llm_secrets_are_not_persisted(tmp_path):
    index_dir = str(tmp_path / "data")
    payload = {
        "api_kind": "corp_async",
        "api_model": "m",
        "api_key": "secret-key",
        "openai_api_key": "secret-openai-key",
        "gigachat_credentials": "secret-gigachat-key",
    }

    save_llm_settings(index_dir, payload)

    loaded = load_llm_settings(index_dir)
    assert "api_key" not in loaded
    assert "openai_api_key" not in loaded
    assert "gigachat_credentials" not in loaded
    assert loaded["api_model"] == "m"


def test_apply_llm_settings_overrides_cfg_and_coerces():
    cfg = AppConfig(adl_dir=".")
    apply_llm_settings(
        cfg,
        {
            "llm_mode": "ollama",
            "ollama_model": "m1",
            "corp_max_tokens": "512",
            "corp_use_max_tokens": "yes",
            "corp_temperature": "0.4",
            "gigachat_model": "GigaChat-2-Pro",
            "gigachat_temperature": "0.5",
            "gigachat_verify_ssl_certs": "no",
            "api_key": "ignored-secret",
        },
    )

    assert cfg.llm_mode == "ollama"
    assert cfg.ollama_model == "m1"
    assert cfg.corp_max_tokens == 512
    assert cfg.corp_use_max_tokens is True
    assert cfg.corp_temperature == 0.4
    assert cfg.gigachat_model == "GigaChat-2-Pro"
    assert cfg.gigachat_temperature == 0.5
    assert cfg.gigachat_verify_ssl_certs is False
    assert cfg.api_key == ""  # секреты берутся только из env
    assert cfg.api_model == ""


def test_apply_llm_settings_bumps_old_disabled_max_tokens():
    cfg = AppConfig(adl_dir=".")
    apply_llm_settings(cfg, {"corp_use_max_tokens": False, "corp_max_tokens": 800})

    assert cfg.corp_use_max_tokens is False
    assert cfg.corp_max_tokens == 8192


def test_apply_llm_settings_new_corp_params_and_types():
    cfg = AppConfig(adl_dir=".")
    apply_llm_settings(
        cfg,
        {
            "corp_use_top_p": "yes",
            "corp_top_p": "0.85",
            "corp_use_top_k": "true",
            "corp_top_k": "25",
            "corp_use_enable_thinking": "1",
            "corp_enable_thinking": "no",
            "corp_use_presence_penalty": "on",
            "corp_presence_penalty": "-0.5",
            "corp_use_repetition_penalty": "yes",
            "corp_repetition_penalty": "1.2",
        },
    )

    assert cfg.corp_use_top_p is True
    assert cfg.corp_top_p == 0.85
    assert cfg.corp_use_top_k is True
    assert cfg.corp_top_k == 25
    assert cfg.corp_use_enable_thinking is True
    assert cfg.corp_enable_thinking is False
    assert cfg.corp_use_presence_penalty is True
    assert cfg.corp_presence_penalty == -0.5
    assert cfg.corp_use_repetition_penalty is True
    assert cfg.corp_repetition_penalty == 1.2

    expected = {
        "corp_use_top_p",
        "corp_top_p",
        "corp_use_top_k",
        "corp_top_k",
        "corp_use_enable_thinking",
        "corp_enable_thinking",
        "corp_use_presence_penalty",
        "corp_presence_penalty",
        "corp_use_repetition_penalty",
        "corp_repetition_penalty",
    }
    assert expected.issubset(LLM_SETTINGS_TYPES)


def test_load_llm_settings_corrupt_file_returns_empty(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "llm.json").write_text("{not json", encoding="utf-8")

    assert load_llm_settings(str(data_dir)) == {}


def test_saved_llm_settings_do_not_override_env_secrets(tmp_path, monkeypatch):
    index_dir = str(tmp_path / "data")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    save_llm_settings(index_dir, {"api_key": "saved-key", "api_model": "m"})

    cfg = from_env(default_adl_dir="")
    apply_llm_settings(cfg, load_llm_settings(index_dir))

    assert cfg.api_key == "env-key"
    assert cfg.api_model == "m"
    assert "api_key" not in LLM_SETTINGS_TYPES
    assert "gigachat_credentials" not in LLM_SETTINGS_TYPES
    assert "openai_api_key" not in LLM_SETTINGS_TYPES
    assert {"api_kind", "api_model", "gigachat_model"}.issubset(LLM_SETTINGS_TYPES)
    assert {"openai_base_url", "openai_model"}.issubset(LLM_SETTINGS_TYPES)


def test_saved_settings_json_is_valid_utf8(tmp_path):
    index_dir = str(tmp_path / "data")
    save_settings(index_dir, {"adl_dir": r"C:\Проекты\ADL"})
    save_llm_settings(index_dir, {"api_model": "модель"})

    with open(settings_path(index_dir), encoding="utf-8") as f:
        assert json.load(f)["adl_dir"] == r"C:\Проекты\ADL"
    with open(llm_settings_path(index_dir), encoding="utf-8") as f:
        assert json.load(f)["api_model"] == "модель"


@pytest.mark.parametrize(
    ("env_value", "argv", "expected"),
    [
        (None, ["app.py"], False),
        ("1", ["app.py"], True),
        ("true", ["app.py"], True),
        ("0", ["app.py"], False),
        (None, ["app.py", "--dev"], True),
    ],
)
def test_is_dev_mode_from_env_or_argv(monkeypatch, env_value, argv, expected):
    if env_value is None:
        monkeypatch.delenv("ADL_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("ADL_DEV_MODE", env_value)
    monkeypatch.setattr(sys, "argv", argv)

    assert is_dev_mode() is expected
