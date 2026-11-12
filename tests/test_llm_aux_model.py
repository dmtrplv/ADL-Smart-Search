import json

from adl_smart_search.config import AppConfig
from adl_smart_search.ui.llm_provider import _provider_params


def _cfg(**overrides):
    cfg = AppConfig(adl_dir="")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_aux_disabled_uses_main_models():
    cfg = _cfg(
        gigachat_model="GigaChat-2",
        gigachat_aux_model="GigaChat-2-Pro",
        llm_aux_model_enabled=False,
    )

    assert _provider_params(cfg, aux=True)["gigachat_model"] == "GigaChat-2"


def test_aux_enabled_uses_aux_model():
    cfg = _cfg(
        gigachat_model="GigaChat-2",
        gigachat_aux_model="GigaChat-2-Pro",
        llm_aux_model_enabled=True,
    )

    assert _provider_params(cfg, aux=False)["gigachat_model"] == "GigaChat-2"
    assert _provider_params(cfg, aux=True)["gigachat_model"] == "GigaChat-2-Pro"


def test_aux_empty_falls_back_to_main():
    cfg = _cfg(
        gigachat_model="GigaChat-2",
        gigachat_aux_model="",
        llm_aux_model_enabled=True,
    )

    assert _provider_params(cfg, aux=True)["gigachat_model"] == "GigaChat-2"


def test_aux_keeps_the_same_provider_and_credentials():
    cfg = _cfg(
        api_kind="gigachat",
        gigachat_credentials="secret",
        gigachat_scope="GIGACHAT_API_B2B",
        gigachat_model="GigaChat-2",
        gigachat_aux_model="GigaChat-2-Pro",
        llm_aux_model_enabled=True,
    )

    main = _provider_params(cfg, aux=False)
    aux = _provider_params(cfg, aux=True)

    assert aux["api_kind"] == main["api_kind"]
    assert aux["gigachat_credentials"] == main["gigachat_credentials"]
    assert aux["gigachat_scope"] == main["gigachat_scope"]


def test_aux_model_applies_to_every_provider():
    cfg = _cfg(
        llm_aux_model_enabled=True,
        ollama_model="main-ollama",
        ollama_aux_model="aux-ollama",
        api_model="main-corp",
        api_aux_model="aux-corp",
        openai_model="main-openai",
        openai_aux_model="aux-openai",
    )

    aux = _provider_params(cfg, aux=True)

    assert aux["ollama_model"] == "aux-ollama"
    assert aux["api_model"] == "aux-corp"
    assert aux["openai_model"] == "aux-openai"


def test_aux_corp_kwargs_fall_back_to_main():
    cfg = _cfg(
        llm_aux_model_enabled=True,
        api_model="main-corp",
        api_aux_model="aux-corp",
        corp_model_kwargs_json='{"service_url": "main"}',
        corp_aux_model_kwargs_json="",
    )

    aux = _provider_params(cfg, aux=True)

    assert "main" in aux["corp_model_kwargs_json"]


def test_aux_corp_explicit_empty_kwargs_do_not_borrow_main():
    cfg = _cfg(
        llm_aux_model_enabled=True,
        api_model="main-corp",
        api_aux_model="aux-corp",
        corp_model_kwargs_json='{"service_url": "main"}',
        corp_aux_model_kwargs_json="{}",
    )

    aux = _provider_params(cfg, aux=True)

    assert "main" not in aux["corp_model_kwargs_json"]
    assert json.loads(aux["corp_model_kwargs_json"])["max_tokens"] == 8192


def test_aux_corp_can_use_another_provider():
    cfg = _cfg(
        api_kind="corp_async",
        llm_aux_model_enabled=True,
        corp_provider="aiedu-v2",
        api_model="/model",
        corp_aux_provider="mosgpt",
        api_aux_model="openqwen/model-v49",
    )

    main = _provider_params(cfg, aux=False)
    aux = _provider_params(cfg, aux=True)

    assert main["corp_provider"] == "aiedu-v2"
    assert aux["corp_provider"] == "mosgpt"
    assert aux["api_model"] == "openqwen/model-v49"


def test_aux_corp_provider_falls_back_to_main():
    cfg = _cfg(
        api_kind="corp_async",
        corp_provider="aiedu-v2",
        corp_aux_provider="",
    )

    assert _provider_params(cfg, aux=True)["corp_provider"] == "aiedu-v2"
