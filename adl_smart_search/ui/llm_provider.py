"""LLM-настройки в UI: каталог моделей, кэш провайдеров, дефолты сессии."""

import json
import logging
import os

import streamlit as st

from adl_smart_search.config import (
    DEFAULT_CORP_CATALOG_PATH,
    LEGACY_CORP_CATALOG_PATH,
)
from adl_smart_search.corp_catalog import (
    corp_catalog_by_name,
    resolve_corp_catalog_from_paths,
)
from adl_smart_search.embedder import get_default_embedder
from adl_smart_search.llm import build_provider
from adl_smart_search.ui.settings import LLM_UI_DEFAULTS

logger = logging.getLogger(__name__)

API_KIND_OPTIONS = ["gigachat", "corp_async", "openai_compatible"]
CHAT_PROVIDER_OPTIONS = ("gigachat", "corp_async")
CORP_API_KIND = "corp_async"
DEFAULT_API_KIND = "corp_async"
# Модели GigaChat API (PERS/B2B). Порядок — от лёгкой к самой мощной.
GIGACHAT_MODEL_OPTIONS = [
    "GigaChat-2",
    "GigaChat-2-Pro",
    "GigaChat-2-Max",
    "GigaChat-3-Lightning",
    "GigaChat-3-Pro",
    "GigaChat-3-Ultra",
]


def resolve_chat_provider(value: str | None) -> str:
    """Допустимый провайдер чата; всё остальное сводится к МЭШ.ИИ.Хаб."""
    return value if value in CHAT_PROVIDER_OPTIONS else DEFAULT_API_KIND


def selected_chat_provider() -> str:
    """Провайдер, выбранный радиокнопкой в чате.

    Выбор живёт только в session_state. При новом запуске приложения
    ``init_llm_session_defaults`` ставит ``corp_async``.
    """
    return resolve_chat_provider(st.session_state.get("chat_provider_choice"))


def load_ui_catalog(cfg) -> list[dict]:
    """Каталог для UI: env/файл → config/corp_catalog.json → data/ → legacy."""
    configured = getattr(cfg, "corp_catalog_path", "") or ""
    config_dir = getattr(cfg, "config_dir", "") or "./config"
    index_dir = getattr(cfg, "index_dir", "") or "./data"
    return resolve_corp_catalog_from_paths(
        configured,
        os.path.join(config_dir, "corp_catalog.json"),
        os.path.join(index_dir, "corp_catalog.json"),
        DEFAULT_CORP_CATALOG_PATH,
        LEGACY_CORP_CATALOG_PATH,
    )


def apply_corp_catalog_selection(catalog: list[dict]):
    """
    on_change callback: при выборе модели из каталога подставляем provider/model/model_kwargs/context_window.
    """
    name = (st.session_state.get("corp_model_name") or "").strip()
    if not name or name == "(вручную)":
        return
    item = corp_catalog_by_name(catalog).get(name)
    if not item:
        return
    new_provider = (item.get("provider") or "").strip()
    st.session_state["corp_provider"] = new_provider
    st.session_state["api_model"] = item.get("model", "") or ""
    kwargs = item.get("model_kwargs", {}) or {}
    st.session_state["corp_model_kwargs_json"] = json.dumps(kwargs, ensure_ascii=False)
    context_window = item.get("context_window")
    if isinstance(context_window, (int, float)) and context_window > 0:
        st.session_state["corp_context_window"] = int(context_window)


def apply_corp_aux_catalog_selection(catalog: list[dict]):
    """on_change: модель для перевода/расширения из каталога.

    Внутри corp_async можно смешивать aiedu-v2 и mosgpt: провайдер
    подставляется из выбранной модели автоматически.
    """
    name = (st.session_state.get("corp_aux_model_name") or "").strip()
    if not name or name == "(вручную)":
        st.session_state["corp_aux_provider"] = ""
        st.session_state.pop("corp_aux_model_warning", None)
        return
    item = corp_catalog_by_name(catalog).get(name)
    if not item:
        return
    st.session_state["corp_aux_provider"] = (item.get("provider") or "").strip()
    st.session_state["api_aux_model"] = item.get("model", "") or ""
    st.session_state["corp_aux_model_kwargs_json"] = json.dumps(
        item.get("model_kwargs", {}) or {}, ensure_ascii=False
    )
    st.session_state.pop("corp_aux_model_warning", None)


def safe_load_json_dict(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("safe_load_json_dict: invalid JSON, fallback to {}")
        return {}


def merged_ollama_options_json(cfg) -> str:
    base: dict = {}
    if getattr(cfg, "ollama_use_temperature", False):
        base["temperature"] = float(getattr(cfg, "ollama_temperature", cfg.ollama_temperature))
    if getattr(cfg, "ollama_use_top_p", False):
        base["top_p"] = float(getattr(cfg, "ollama_top_p", cfg.ollama_top_p))
    if getattr(cfg, "ollama_use_repeat_penalty", False):
        base["repeat_penalty"] = float(getattr(cfg, "ollama_repeat_penalty", cfg.ollama_repeat_penalty))
    if getattr(cfg, "ollama_use_num_predict", False):
        base["num_predict"] = int(getattr(cfg, "ollama_num_predict", cfg.ollama_num_predict))
    return json.dumps(base, ensure_ascii=False)


def _merged_corp_model_kwargs_json(cfg, raw: str, *, aux: bool = False) -> str:
    """
    Берём базовый JSON из UI и добавляем параметры генерации МЭШ.ИИ.Хаб.

    Для aux-модели используются поля ``corp_aux_*``. ``temperature`` и
    ``enable_thinking`` уходят внутрь ``chat_template_kwargs``, остальные
    параметры — на верхний уровень.

    ``max_tokens`` отправляем всегда: у некоторых провайдеров маленький лимит
    по умолчанию обрезает ответ. Явное значение в JSON или включённый чекбокс
    имеют приоритет.
    """
    prefix = "corp_aux_" if aux else "corp_"

    def value(name: str, default):
        return getattr(cfg, prefix + name, default)

    base: dict = {}
    raw = (raw or "").strip()
    if raw:
        try:
            base = json.loads(raw)
            if not isinstance(base, dict):
                base = {}
        except Exception:
            logger.warning("merged_corp_model_kwargs_json: invalid model_kwargs JSON, fallback to {}")
            base = {}

    max_tokens = int(value("max_tokens", 8192))
    if bool(value("use_max_tokens", False)) or "max_tokens" not in base:
        base["max_tokens"] = max_tokens

    # temperature и enable_thinking — только внутри chat_template_kwargs.
    chat_raw = base.get("chat_template_kwargs")
    had_chat_kwargs = isinstance(chat_raw, dict)
    chat_kwargs = dict(chat_raw) if had_chat_kwargs else {}
    if bool(value("use_temperature", False)):
        # Старое значение на верхнем уровне больше не отправляем.
        base.pop("temperature", None)
        chat_kwargs["temperature"] = float(value("temperature", 0.2))
    if bool(value("use_enable_thinking", False)):
        chat_kwargs["enable_thinking"] = bool(value("enable_thinking", False))
    if chat_kwargs or had_chat_kwargs:
        base["chat_template_kwargs"] = chat_kwargs

    if bool(value("use_top_p", False)):
        base["top_p"] = float(value("top_p", 0.9))
    if bool(value("use_top_k", False)):
        base["top_k"] = int(value("top_k", 40))
    if bool(value("use_presence_penalty", False)):
        base["presence_penalty"] = float(value("presence_penalty", 0.0))
    if bool(value("use_repetition_penalty", False)):
        base["repetition_penalty"] = float(value("repetition_penalty", 1.0))
    return json.dumps(base, ensure_ascii=False)


def merged_corp_model_kwargs_json(cfg) -> str:
    """model_kwargs основной модели corp_async."""
    return _merged_corp_model_kwargs_json(
        cfg, getattr(cfg, "corp_model_kwargs_json", "") or ""
    )


def merged_corp_aux_model_kwargs_json(cfg) -> str:
    """model_kwargs вспомогательной модели corp_async (перевод/расширение).

    Если отдельных kwargs нет (пустая строка), берём kwargs основной модели.
    Явный ``{}`` из каталога означает «своих параметров нет» и не подменяется
    kwargs основной модели — это важно при выборе aux-модели другого провайдера.
    """
    raw = getattr(cfg, "corp_aux_model_kwargs_json", "") or ""
    if not raw.strip():
        raw = getattr(cfg, "corp_model_kwargs_json", "") or ""
    return _merged_corp_model_kwargs_json(cfg, raw, aux=True)


def missing_llm_fields_message(cfg) -> str:
    mode = (getattr(cfg, "llm_mode", "") or "").strip().lower()
    if mode != "api":
        return ""
    api_kind = (getattr(cfg, "api_kind", DEFAULT_API_KIND) or DEFAULT_API_KIND).strip().lower()

    if api_kind in ("gigachat", "giga"):
        missing = []
        if not (getattr(cfg, "gigachat_credentials", "") or "").strip():
            missing.append("GigaChat credentials (GIGACHAT_CREDENTIALS)")
        if not (getattr(cfg, "gigachat_model", "") or "").strip():
            missing.append("GigaChat model")
        if not missing:
            return ""
        return "Для GigaChat заполните: " + ", ".join(missing) + "."

    if api_kind in ("openai", "openai_compatible", "compatible"):
        missing = []
        if not (getattr(cfg, "openai_base_url", "") or "").strip():
            missing.append("OpenAI base url")
        if not (getattr(cfg, "openai_api_key", "") or "").strip():
            missing.append("OpenAI API key (OPENAI_API_KEY)")
        if not (getattr(cfg, "openai_model", "") or "").strip():
            missing.append("OpenAI model")
        if not missing:
            return ""
        return "Для OpenAI-совместимого API заполните: " + ", ".join(missing) + "."

    api_base_url = (getattr(cfg, "api_base_url", "") or "").strip()
    api_key = (getattr(cfg, "api_key", "") or "").strip()
    api_model = (getattr(cfg, "api_model", "") or "").strip()

    missing = []
    if not api_base_url:
        missing.append("API base url")
    if not api_key:
        missing.append("API key")
    if not api_model:
        missing.append("API model")
    if api_kind in ("corp_async", "corporate_async", "async"):
        corp_provider = (getattr(cfg, "corp_provider", "") or "").strip()
        if not corp_provider:
            missing.append("Corp provider")
    if not missing:
        return ""
    return "Для LLM по API заполните: " + ", ".join(missing) + "."


@st.cache_resource
def get_embedder(model_name: str):
    return get_default_embedder(model_name=model_name)


@st.cache_resource
def _aux_model(cfg, main_key: str, aux_key: str) -> str:
    """Имя модели для перевода/расширения: aux, если задана, иначе основная."""
    main = str(getattr(cfg, main_key, "") or "")
    if not bool(getattr(cfg, "llm_aux_model_enabled", False)):
        return main
    aux = str(getattr(cfg, aux_key, "") or "").strip()
    return aux or main


def _provider_params(cfg, *, aux: bool) -> dict:
    """Параметры build_provider.

    Для ``aux`` можно выбрать модель другого корпоративного провайдера
    (например, mosgpt при основной aiedu-v2): провайдер берётся из
    ``corp_aux_provider``. Ключи, URL и параметры генерации остаются общими.
    """

    def model(main_key: str, aux_key: str) -> str:
        if aux:
            return _aux_model(cfg, main_key, aux_key)
        return str(getattr(cfg, main_key, "") or "")

    corp_provider = getattr(cfg, "corp_provider", "") or ""
    if aux:
        corp_provider = getattr(cfg, "corp_aux_provider", "") or corp_provider

    return {
        "mode": getattr(cfg, "llm_mode", "") or "api",
        "ollama_model": model("ollama_model", "ollama_aux_model"),
        "ollama_base_url": getattr(cfg, "ollama_base_url", "") or "",
        "ollama_options": safe_load_json_dict(merged_ollama_options_json(cfg)),
        "api_base_url": getattr(cfg, "api_base_url", "") or "",
        "api_key": getattr(cfg, "api_key", "") or "",
        "api_model": model("api_model", "api_aux_model"),
        "openai_base_url": getattr(cfg, "openai_base_url", "") or "",
        "openai_api_key": getattr(cfg, "openai_api_key", "") or "",
        "openai_model": model("openai_model", "openai_aux_model"),
        "api_kind": getattr(cfg, "api_kind", DEFAULT_API_KIND) or DEFAULT_API_KIND,
        "corp_provider": corp_provider,
        "corp_model_kwargs_json": (
            merged_corp_aux_model_kwargs_json(cfg)
            if aux
            else merged_corp_model_kwargs_json(cfg)
        ),
        "corp_poll_timeout_s": int(
            getattr(cfg, "corp_poll_timeout_s", cfg.corp_poll_timeout_s) or cfg.corp_poll_timeout_s
        ),
        "corp_poll_interval_s": float(
            getattr(cfg, "corp_poll_interval_s", cfg.corp_poll_interval_s) or cfg.corp_poll_interval_s
        ),
        "gigachat_credentials": getattr(cfg, "gigachat_credentials", "") or "",
        "gigachat_scope": getattr(cfg, "gigachat_scope", "GIGACHAT_API_PERS") or "GIGACHAT_API_PERS",
        "gigachat_model": model("gigachat_model", "gigachat_aux_model"),
        "gigachat_base_url": getattr(cfg, "gigachat_base_url", "") or "",
        "gigachat_ca_bundle_file": getattr(cfg, "gigachat_ca_bundle_file", "") or "",
        "gigachat_verify_ssl_certs": bool(getattr(cfg, "gigachat_verify_ssl_certs", True)),
        "gigachat_timeout_s": float(getattr(cfg, "gigachat_timeout_s", 120.0) or 120.0),
        "gigachat_temperature": float(getattr(cfg, "gigachat_temperature", 0.2) or 0.2),
        "gigachat_max_tokens": int(getattr(cfg, "gigachat_max_tokens", 4096) or 4096),
    }


@st.cache_resource
def get_llm_provider_cached(role: str, params_json: str):
    """Кэш провайдеров: role разделяет основную и вспомогательную модели.

    Разные экземпляры важны не только для модели, но и для статистики:
    ``usage_total``/``last_finish_reason`` перевода и расширения не должны
    смешиваться со статистикой ответа.
    """
    return build_provider(**json.loads(params_json))


def _get_provider(cfg, *, role: str, aux: bool):
    params = _provider_params(cfg, aux=aux)
    return get_llm_provider_cached(
        role,
        json.dumps(params, sort_keys=True, ensure_ascii=False),
    )


def get_llm_provider(cfg):
    """Провайдер для ответов ИИ."""
    return _get_provider(cfg, role="answer", aux=False)


def get_aux_llm_provider(cfg):
    """Провайдер для перевода запроса и предложения расширений."""
    return _get_provider(cfg, role="aux", aux=True)


def init_llm_session_defaults(cfg, persisted: dict | None = None) -> None:
    """
    UI state: фиксируем значения LLM-настроек в session_state,
    чтобы они не сбрасывались при переключении режимов.

    ``persisted`` — данные из data/llm.json: ими заполняются поля, которых нет в cfg
    (выбранная модель каталога и размер контекста).
    """
    persisted = persisted or {}
    llm_defaults = {
        # LLM mode: по умолчанию API (Оллама доступна только в dev-режиме через radio).
        "llm_mode": (getattr(cfg, "llm_mode", "") or "api"),
        "ollama_model": (cfg.ollama_model or ""),
        "ollama_base_url": getattr(cfg, "ollama_base_url", "") or "",
        "ollama_use_temperature": bool(getattr(cfg, "ollama_use_temperature", False)),
        "ollama_temperature": float(getattr(cfg, "ollama_temperature", cfg.ollama_temperature)),
        "ollama_use_top_p": bool(getattr(cfg, "ollama_use_top_p", False)),
        "ollama_top_p": float(getattr(cfg, "ollama_top_p", cfg.ollama_top_p)),
        "ollama_use_repeat_penalty": bool(getattr(cfg, "ollama_use_repeat_penalty", False)),
        "ollama_repeat_penalty": float(getattr(cfg, "ollama_repeat_penalty", cfg.ollama_repeat_penalty)),
        "ollama_use_num_predict": bool(getattr(cfg, "ollama_use_num_predict", False)),
        "ollama_num_predict": int(getattr(cfg, "ollama_num_predict", cfg.ollama_num_predict)),
        # API. Провайдер выбирается радиокнопкой в чате, по умолчанию — corp_async.
        "api_kind": DEFAULT_API_KIND,
        "chat_provider_choice": DEFAULT_API_KIND,
        "api_base_url": (cfg.api_base_url or cfg.api_base_url_default),
        "api_key": (cfg.api_key or ""),
        "api_model": (cfg.api_model or ""),
        # OpenAI-совместимый API — отдельные настройки (без данных МЭШ)
        "openai_base_url": getattr(cfg, "openai_base_url", "") or "",
        "openai_api_key": getattr(cfg, "openai_api_key", "") or "",
        "openai_model": getattr(cfg, "openai_model", "") or "",
        # GigaChat
        "gigachat_credentials": getattr(cfg, "gigachat_credentials", "") or "",
        "gigachat_scope": getattr(cfg, "gigachat_scope", "GIGACHAT_API_PERS") or "GIGACHAT_API_PERS",
        "gigachat_model": getattr(cfg, "gigachat_model", "GigaChat-2") or "GigaChat-2",
        "gigachat_base_url": getattr(cfg, "gigachat_base_url", "") or "",
        "gigachat_ca_bundle_file": getattr(cfg, "gigachat_ca_bundle_file", "") or "",
        "gigachat_verify_ssl_certs": bool(getattr(cfg, "gigachat_verify_ssl_certs", True)),
        "gigachat_timeout_s": float(getattr(cfg, "gigachat_timeout_s", 120.0) or 120.0),
        "gigachat_temperature": float(getattr(cfg, "gigachat_temperature", 0.2) or 0.2),
        "gigachat_max_tokens": int(getattr(cfg, "gigachat_max_tokens", 4096) or 4096),
        "gigachat_context_window": int(getattr(cfg, "gigachat_context_window", 32000) or 32000),
        # corp_async extras
        "corp_model_name": str(persisted.get("corp_model_name") or LLM_UI_DEFAULTS["corp_model_name"]),
        "corp_aux_model_name": str(
            persisted.get("corp_aux_model_name") or LLM_UI_DEFAULTS["corp_aux_model_name"]
        ),
        # провайдер по умолчанию убираем: пользователь выбирает модель или провайдера явно
        "corp_provider": getattr(cfg, "corp_provider", "") or "",
        "corp_use_temperature": bool(getattr(cfg, "corp_use_temperature", False)),
        "corp_temperature": float(getattr(cfg, "corp_temperature", cfg.corp_temperature)),
        "corp_use_max_tokens": bool(getattr(cfg, "corp_use_max_tokens", False)),
        "corp_max_tokens": int(getattr(cfg, "corp_max_tokens", cfg.corp_max_tokens)),
        "corp_use_top_p": bool(getattr(cfg, "corp_use_top_p", False)),
        "corp_top_p": float(getattr(cfg, "corp_top_p", cfg.corp_top_p)),
        "corp_use_top_k": bool(getattr(cfg, "corp_use_top_k", False)),
        "corp_top_k": int(getattr(cfg, "corp_top_k", cfg.corp_top_k)),
        "corp_use_enable_thinking": bool(getattr(cfg, "corp_use_enable_thinking", False)),
        "corp_enable_thinking": bool(getattr(cfg, "corp_enable_thinking", cfg.corp_enable_thinking)),
        "corp_use_presence_penalty": bool(getattr(cfg, "corp_use_presence_penalty", False)),
        "corp_presence_penalty": float(
            getattr(cfg, "corp_presence_penalty", cfg.corp_presence_penalty)
        ),
        "corp_use_repetition_penalty": bool(getattr(cfg, "corp_use_repetition_penalty", False)),
        "corp_repetition_penalty": float(
            getattr(cfg, "corp_repetition_penalty", cfg.corp_repetition_penalty)
        ),
        "corp_aux_use_temperature": bool(getattr(cfg, "corp_aux_use_temperature", False)),
        "corp_aux_temperature": float(getattr(cfg, "corp_aux_temperature", 0.2)),
        "corp_aux_use_max_tokens": bool(getattr(cfg, "corp_aux_use_max_tokens", False)),
        "corp_aux_max_tokens": int(getattr(cfg, "corp_aux_max_tokens", 8192)),
        "corp_aux_use_top_p": bool(getattr(cfg, "corp_aux_use_top_p", False)),
        "corp_aux_top_p": float(getattr(cfg, "corp_aux_top_p", 0.9)),
        "corp_aux_use_top_k": bool(getattr(cfg, "corp_aux_use_top_k", False)),
        "corp_aux_top_k": int(getattr(cfg, "corp_aux_top_k", 40)),
        "corp_aux_use_enable_thinking": bool(getattr(cfg, "corp_aux_use_enable_thinking", False)),
        "corp_aux_enable_thinking": bool(getattr(cfg, "corp_aux_enable_thinking", False)),
        "corp_aux_use_presence_penalty": bool(
            getattr(cfg, "corp_aux_use_presence_penalty", False)
        ),
        "corp_aux_presence_penalty": float(getattr(cfg, "corp_aux_presence_penalty", 0.0)),
        "corp_aux_use_repetition_penalty": bool(
            getattr(cfg, "corp_aux_use_repetition_penalty", False)
        ),
        "corp_aux_repetition_penalty": float(
            getattr(cfg, "corp_aux_repetition_penalty", 1.0)
        ),
        "corp_poll_timeout_s": int(getattr(cfg, "corp_poll_timeout_s", cfg.corp_poll_timeout_s)),
        "corp_poll_interval_s": float(getattr(cfg, "corp_poll_interval_s", cfg.corp_poll_interval_s)),
        "corp_model_kwargs_json": getattr(cfg, "corp_model_kwargs_json", "") or "{}",
        # Отдельная модель для перевода и расширения (dev). Внутри corp_async
        # провайдер можно не совпадать с основной моделью (aiedu-v2 ↔ mosgpt).
        "llm_aux_model_enabled": bool(getattr(cfg, "llm_aux_model_enabled", False)),
        "ollama_aux_model": getattr(cfg, "ollama_aux_model", "") or "",
        "api_aux_model": getattr(cfg, "api_aux_model", "") or "",
        "openai_aux_model": getattr(cfg, "openai_aux_model", "") or "",
        "corp_aux_provider": getattr(cfg, "corp_aux_provider", "") or "",
        "corp_aux_model_kwargs_json": getattr(cfg, "corp_aux_model_kwargs_json", "") or "",
        "gigachat_aux_model": getattr(cfg, "gigachat_aux_model", "") or "",
        # context_window (tokens) текущей модели из каталога; дефолт — 8k (conservative).
        "corp_context_window": int(persisted.get("corp_context_window") or 0)
        or LLM_UI_DEFAULTS["corp_context_window"],
    }
    for k, v in llm_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Если ключ уже был создан раньше, но остался пустым — подставим дефолт.
    if not (st.session_state.get("api_base_url") or "").strip():
        st.session_state["api_base_url"] = cfg.api_base_url_default
    # Защита от битых/старых значений в session_state или llm.json.
    if st.session_state.get("llm_mode") not in ("ollama", "api"):
        st.session_state["llm_mode"] = "api"
    if st.session_state.get("api_kind") not in API_KIND_OPTIONS:
        st.session_state["api_kind"] = DEFAULT_API_KIND
    if st.session_state.get("chat_provider_choice") not in CHAT_PROVIDER_OPTIONS:
        st.session_state["chat_provider_choice"] = DEFAULT_API_KIND
