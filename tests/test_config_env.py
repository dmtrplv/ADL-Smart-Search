from pathlib import Path

from adl_smart_search.config import (
    DEFAULT_AUTO_STRICTNESS,
    AppConfig,
    from_env,
)


def test_default_config_values():
    cfg = AppConfig(adl_dir=".")
    assert cfg.auto_strictness == DEFAULT_AUTO_STRICTNESS
    # Адреса corp_async в коде нет: он задаётся в config/llm.json (api_base_url).
    assert cfg.api_base_url == ""
    assert cfg.corp_catalog_path == "./config/corp_catalog.json"
    assert cfg.corp_providers == ""
    assert cfg.api_kind == ""
    assert cfg.chat_provider == "corp_async"
    assert cfg.gigachat_scope == "GIGACHAT_API_PERS"
    assert cfg.gigachat_model == "GigaChat-2"
    assert cfg.llm_mode == "api"
    assert cfg.openai_base_url == ""
    assert cfg.openai_model == ""


def test_from_env_reads_only_llm_secrets(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "corp-key")
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "gc-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    # Не-секретные настройки LLM из окружения больше не читаются:
    # они живут в data/llm.json (их сохраняет dev-панель).
    monkeypatch.setenv("GIGACHAT_SCOPE", "GIGACHAT_API_B2B")
    monkeypatch.setenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
    monkeypatch.setenv("GIGACHAT_CONTEXT_WINDOW", "64000")
    monkeypatch.setenv("LLM_API_MODEL", "corp-model")
    monkeypatch.setenv("LLM_API_KIND", "corp_async")
    monkeypatch.setenv("LLM_MODE", "ollama")

    cfg = from_env(default_adl_dir="")

    assert cfg.api_key == "corp-key"
    assert cfg.gigachat_credentials == "gc-key"
    assert cfg.openai_api_key == "oa-key"

    assert cfg.gigachat_scope == "GIGACHAT_API_PERS"
    assert cfg.gigachat_model == "GigaChat-2"
    assert cfg.gigachat_context_window == 32000
    assert cfg.api_model == ""
    assert cfg.llm_mode == "api"
    # Адрес и провайдер по умолчанию из окружения не подставляются.
    assert cfg.api_base_url == ""
    assert cfg.chat_provider == "corp_async"


def test_from_env_reads_new_keys(monkeypatch):
    monkeypatch.setenv("ADL_AUTO_STRICTNESS", "0.75")
    monkeypatch.setenv("ADL_CORP_CATALOG", "./corp.json")
    monkeypatch.setenv("ADL_CORP_PROVIDERS", "p1,p2")
    cfg = from_env(default_adl_dir="")
    assert cfg.auto_strictness == 0.75
    assert cfg.corp_catalog_path == "./corp.json"
    assert cfg.corp_providers == "p1,p2"
    assert cfg.api_base_url == ""


def test_from_env_uses_defaults_when_unset(monkeypatch):
    for k in ("ADL_AUTO_STRICTNESS", "ADL_CORP_CATALOG", "ADL_CORP_PROVIDERS"):
        monkeypatch.delenv(k, raising=False)
    cfg = from_env(default_adl_dir="")
    assert cfg.auto_strictness == DEFAULT_AUTO_STRICTNESS
    assert cfg.api_base_url == ""
    assert Path(cfg.corp_catalog_path) == Path("config") / "corp_catalog.json"
    assert cfg.corp_providers == ""
