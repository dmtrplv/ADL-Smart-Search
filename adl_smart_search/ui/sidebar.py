"""Настройки разработчика (индексация, поиск, LLM) — блок над разделами."""

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import streamlit as st

from adl_smart_search.config import AppConfig, from_env
from adl_smart_search.corp_catalog import catalog_providers
from adl_smart_search.index_build_info import save_kind_build_info
from adl_smart_search.index_cache import clear_index_cache
from adl_smart_search.index_files import (
    CKM_SOURCE,
    LOCAL_SOURCE,
    index_filenames,
    index_kind_files_exist,
    source_index_dir,
)
from adl_smart_search.prompts import (
    PROMPT_LABELS,
    PROMPT_PART_KEYS,
    PromptConfigError,
    load_prompts,
    prompts_path,
    save_prompts,
)
from adl_smart_search.ui.context import prompts_from_session
from adl_smart_search.ui.indexing import (
    build_archetype_index_with_progress,
    build_field_index_with_progress,
    delete_archetype_index_files,
    delete_field_index_files,
)
from adl_smart_search.ui.llm_provider import (
    CHAT_PROVIDER_OPTIONS,
    CORP_API_KIND,
    GIGACHAT_MODEL_OPTIONS,
    apply_corp_aux_catalog_selection,
    apply_corp_catalog_selection,
    load_ui_catalog,
    safe_load_json_dict,
    selected_chat_provider,
)
from adl_smart_search.ui.settings import (
    LLM_SETTINGS_TYPES,
    LLM_UI_DEFAULTS,
    SEARCH_SETTINGS_TYPES,
    index_last_built_label,
    load_llm_settings,
    load_settings,
    save_llm_settings,
    save_settings,
)

logger = logging.getLogger(__name__)


def corp_provider_options(cfg) -> list[str]:
    """Список провайдеров: ENV ADL_CORP_PROVIDERS или из каталога (пустой = ручной ввод)."""
    if (getattr(cfg, "corp_providers", "") or "").strip():
        return [""] + [p.strip() for p in cfg.corp_providers.split(",") if p.strip()]
    return [""] + catalog_providers(load_ui_catalog(cfg))


@dataclass
class SidebarState:
    adl_dir: str
    index_dir: str
    ckm_adl_dir: str
    auto_strictness: float
    keyword_bonus: float
    phrase_bonus: float
    misuse_penalty: float
    weak_match_penalty: float
    enable_misuse_penalty: bool
    enable_weak_match_penalty: bool
    max_results: int
    sim_floor: float
    sim_floor_pure: float
    llm_mode: str
    ollama_model: str
    ollama_base_url: str
    api_kind: str
    chat_provider: str
    api_base_url: str
    api_key: str
    api_model: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    corp_provider: str
    corp_model_kwargs_json: str
    corp_poll_timeout_s: int
    corp_poll_interval_s: float
    corp_use_temperature: bool
    corp_temperature: float
    corp_use_max_tokens: bool
    corp_max_tokens: int
    corp_use_top_p: bool
    corp_top_p: float
    corp_use_top_k: bool
    corp_top_k: int
    corp_use_enable_thinking: bool
    corp_enable_thinking: bool
    corp_use_presence_penalty: bool
    corp_presence_penalty: float
    corp_use_repetition_penalty: bool
    corp_repetition_penalty: float
    corp_aux_use_temperature: bool
    corp_aux_temperature: float
    corp_aux_use_max_tokens: bool
    corp_aux_max_tokens: int
    corp_aux_use_top_p: bool
    corp_aux_top_p: float
    corp_aux_use_top_k: bool
    corp_aux_top_k: int
    corp_aux_use_enable_thinking: bool
    corp_aux_enable_thinking: bool
    corp_aux_use_presence_penalty: bool
    corp_aux_presence_penalty: float
    corp_aux_use_repetition_penalty: bool
    corp_aux_repetition_penalty: float
    gigachat_credentials: str
    gigachat_scope: str
    gigachat_model: str
    gigachat_base_url: str
    gigachat_ca_bundle_file: str
    gigachat_verify_ssl_certs: bool
    gigachat_timeout_s: float
    gigachat_temperature: float
    gigachat_max_tokens: int
    gigachat_context_window: int
    llm_aux_model_enabled: bool
    ollama_aux_model: str
    api_aux_model: str
    openai_aux_model: str
    corp_aux_provider: str
    corp_aux_model_kwargs_json: str
    gigachat_aux_model: str

    def apply_to_cfg(self, cfg: AppConfig) -> None:
        cfg.adl_dir = self.adl_dir
        cfg.preferred_lang = cfg.preferred_lang or "ru"
        cfg.index_dir = self.index_dir
        cfg.ckm_adl_dir = self.ckm_adl_dir
        cfg.keyword_bonus = self.keyword_bonus
        cfg.phrase_bonus = self.phrase_bonus
        cfg.misuse_penalty = self.misuse_penalty
        cfg.weak_match_penalty = self.weak_match_penalty
        cfg.enable_misuse_penalty = self.enable_misuse_penalty
        cfg.enable_weak_match_penalty = self.enable_weak_match_penalty
        cfg.max_results = int(self.max_results)
        cfg.sim_floor = float(self.sim_floor)
        cfg.sim_floor_pure = float(self.sim_floor_pure)
        cfg.simi_pref_eps = float(os.getenv("ADL_SIMI_PREF_EPS", str(cfg.simi_pref_eps)))
        cfg.llm_mode = self.llm_mode
        cfg.ollama_model = self.ollama_model
        cfg.ollama_base_url = self.ollama_base_url
        cfg.ollama_use_temperature = bool(st.session_state.get("ollama_use_temperature", cfg.ollama_use_temperature))
        cfg.ollama_temperature = float(st.session_state.get("ollama_temperature", cfg.ollama_temperature))
        cfg.ollama_use_top_p = bool(st.session_state.get("ollama_use_top_p", cfg.ollama_use_top_p))
        cfg.ollama_top_p = float(st.session_state.get("ollama_top_p", cfg.ollama_top_p))
        cfg.ollama_use_repeat_penalty = bool(
            st.session_state.get("ollama_use_repeat_penalty", cfg.ollama_use_repeat_penalty)
        )
        cfg.ollama_repeat_penalty = float(st.session_state.get("ollama_repeat_penalty", cfg.ollama_repeat_penalty))
        cfg.ollama_use_num_predict = bool(st.session_state.get("ollama_use_num_predict", cfg.ollama_use_num_predict))
        cfg.ollama_num_predict = int(st.session_state.get("ollama_num_predict", cfg.ollama_num_predict))
        cfg.api_base_url = self.api_base_url
        cfg.api_key = self.api_key
        cfg.api_model = self.api_model
        cfg.openai_base_url = self.openai_base_url
        cfg.openai_api_key = self.openai_api_key
        cfg.openai_model = self.openai_model
        cfg.api_kind = self.api_kind
        cfg.chat_provider = self.chat_provider
        cfg.corp_provider = self.corp_provider
        cfg.corp_model_kwargs_json = self.corp_model_kwargs_json
        cfg.corp_poll_timeout_s = int(self.corp_poll_timeout_s)
        cfg.corp_poll_interval_s = float(self.corp_poll_interval_s)
        cfg.corp_use_temperature = bool(self.corp_use_temperature)
        cfg.corp_temperature = float(self.corp_temperature)
        cfg.corp_use_max_tokens = bool(self.corp_use_max_tokens)
        cfg.corp_max_tokens = int(self.corp_max_tokens)
        cfg.corp_use_top_p = bool(self.corp_use_top_p)
        cfg.corp_top_p = float(self.corp_top_p)
        cfg.corp_use_top_k = bool(self.corp_use_top_k)
        cfg.corp_top_k = int(self.corp_top_k)
        cfg.corp_use_enable_thinking = bool(self.corp_use_enable_thinking)
        cfg.corp_enable_thinking = bool(self.corp_enable_thinking)
        cfg.corp_use_presence_penalty = bool(self.corp_use_presence_penalty)
        cfg.corp_presence_penalty = float(self.corp_presence_penalty)
        cfg.corp_use_repetition_penalty = bool(self.corp_use_repetition_penalty)
        cfg.corp_repetition_penalty = float(self.corp_repetition_penalty)
        cfg.corp_aux_use_temperature = bool(self.corp_aux_use_temperature)
        cfg.corp_aux_temperature = float(self.corp_aux_temperature)
        cfg.corp_aux_use_max_tokens = bool(self.corp_aux_use_max_tokens)
        cfg.corp_aux_max_tokens = int(self.corp_aux_max_tokens)
        cfg.corp_aux_use_top_p = bool(self.corp_aux_use_top_p)
        cfg.corp_aux_top_p = float(self.corp_aux_top_p)
        cfg.corp_aux_use_top_k = bool(self.corp_aux_use_top_k)
        cfg.corp_aux_top_k = int(self.corp_aux_top_k)
        cfg.corp_aux_use_enable_thinking = bool(self.corp_aux_use_enable_thinking)
        cfg.corp_aux_enable_thinking = bool(self.corp_aux_enable_thinking)
        cfg.corp_aux_use_presence_penalty = bool(self.corp_aux_use_presence_penalty)
        cfg.corp_aux_presence_penalty = float(self.corp_aux_presence_penalty)
        cfg.corp_aux_use_repetition_penalty = bool(self.corp_aux_use_repetition_penalty)
        cfg.corp_aux_repetition_penalty = float(self.corp_aux_repetition_penalty)
        cfg.gigachat_credentials = self.gigachat_credentials
        cfg.gigachat_scope = self.gigachat_scope
        cfg.gigachat_model = self.gigachat_model
        cfg.gigachat_base_url = self.gigachat_base_url
        cfg.gigachat_ca_bundle_file = self.gigachat_ca_bundle_file
        cfg.gigachat_verify_ssl_certs = bool(self.gigachat_verify_ssl_certs)
        cfg.gigachat_timeout_s = float(self.gigachat_timeout_s)
        cfg.gigachat_temperature = float(self.gigachat_temperature)
        cfg.gigachat_max_tokens = int(self.gigachat_max_tokens)
        cfg.gigachat_context_window = int(self.gigachat_context_window)
        cfg.llm_aux_model_enabled = bool(self.llm_aux_model_enabled)
        cfg.ollama_aux_model = self.ollama_aux_model
        cfg.api_aux_model = self.api_aux_model
        cfg.openai_aux_model = self.openai_aux_model
        cfg.corp_aux_provider = self.corp_aux_provider
        cfg.corp_aux_model_kwargs_json = self.corp_aux_model_kwargs_json
        cfg.gigachat_aux_model = self.gigachat_aux_model


def browse_dir(session_key: str = "adl_dir", description: str = "Выберите папку") -> None:
    # Важно:
    # - tkinter диалоги часто падают в Streamlit с "main thread is not in main loop"
    #   (callback выполняется не в GUI main thread).
    # - Поэтому на Windows используем FolderBrowserDialog через отдельный pwsh-процесс.
    try:
        path = ""
        if sys.platform.startswith("win"):
            ps = rf"""
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = "{description}"
$dlg.ShowNewFolderButton = $false
if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
  Write-Output $dlg.SelectedPath
}}
"""
            p = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if p.returncode == 0:
                path = (p.stdout or "").strip()
        if path:
            st.session_state[session_key] = path
            return
    except Exception as e:
        st.warning(f"Не удалось открыть диалог выбора папки: {e}")
        logger.warning("browse_dir: folder dialog failed: %s", e)


def browse_adl_dir() -> None:
    browse_dir("adl_dir", "Выберите папку с .adl")


def _reload_merged_indexes(cfg: AppConfig, adl_dir: str, index_dir: str, ckm_adl_dir: str):
    """После пересборки сбрасывает общий кэш индексов.

    Сами данные загрузятся в app.py на следующем rerun; в session_state копий
    больше нет, поэтому старый набор освобождается.
    """
    clear_index_cache()
    return None


def _record_build_info(index_root: str, source: str, kind: str, cfg: AppConfig, embed_dim: int = 0) -> None:
    from adl_smart_search.index_store import INDEX_FORMAT_VERSION, PARSER_VERSION

    save_kind_build_info(
        index_root,
        source,
        kind,
        parser_version=PARSER_VERSION,
        index_format_version=INDEX_FORMAT_VERSION,
        embed_model=cfg.embed_model_name,
        embed_dim=embed_dim,
    )


def _rebuild_kind(
    adl_dir: str,
    index_dir: str,
    kind: str,
    cfg: AppConfig,
    source: str = LOCAL_SOURCE,
    full: bool = False,
    record_build_info: bool = True,
):
    """Пересобирает индекс одного вида (archetype/field) с прогресс-баром.

    full=False — инкрементально (файлы не удаляются, пересчитываются только изменившиеся .adl).
    full=True — полная пересборка: файлы индекса сначала удаляются.
    """
    if full:
        if kind == "archetype":
            delete_archetype_index_files(index_dir)
        else:
            delete_field_index_files(index_dir)
    if kind == "archetype":
        docs, vectors = build_archetype_index_with_progress(
            adl_dir=adl_dir,
            preferred_lang=cfg.preferred_lang or "ru",
            index_dir=index_dir,
            embed_model_name=cfg.embed_model_name,
            source=source,
        )
    else:
        docs, vectors = build_field_index_with_progress(
            adl_dir=adl_dir,
            preferred_lang=cfg.preferred_lang or "ru",
            index_dir=index_dir,
            embed_model_name=cfg.embed_model_name,
            source=source,
        )
    if record_build_info:
        embed_dim = int(getattr(vectors, "shape", (0, 0))[1]) if getattr(vectors, "ndim", 0) == 2 else 0
        _record_build_info(os.path.dirname(index_dir), source, kind, cfg, embed_dim)
    return docs, vectors


# (подпись строки, подпись кнопки-иконки)
INDEX_ROWS = (
    ("Локальная база — архетипы", LOCAL_SOURCE, "archetype", "Индексировать архетипы локальной базы"),
    ("Локальная база — поля", LOCAL_SOURCE, "field", "Индексировать поля локальной базы"),
    ("CKM — архетипы", CKM_SOURCE, "archetype", "Индексировать архетипы CKM"),
    ("CKM — поля", CKM_SOURCE, "field", "Индексировать поля CKM"),
)


@st.dialog("Полная пересборка индексов", width="small")
def _full_rebuild_dialog() -> None:
    st.write(
        "Индексы архетипов и полей для локальной базы и CKM будут собраны заново. "
        "Это может занять несколько минут — не закрывайте вкладку. "
        "Старые индексы останутся на месте, пока новые не будут готовы."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Отмена", use_container_width=True):
            st.session_state["full_rebuild_open"] = False
            st.rerun()
    with c2:
        if st.button("Пересобрать", type="primary", use_container_width=True):
            st.session_state["full_rebuild_confirm"] = True
            st.session_state["full_rebuild_open"] = False
            st.rerun()


def _run_full_rebuild(cfg: AppConfig, adl_dir: str, ckm_adl_dir: str, index_dir: str) -> bool:
    """Полная пересборка всех индексов; старые файлы заменяются только после успеха."""
    st.toast("Полная пересборка индексов запущена", icon="⏳")
    ok = True
    local_dir = (adl_dir or "").strip()
    ckm_dir = (ckm_adl_dir or "").strip()
    plan = (
        (LOCAL_SOURCE, local_dir, source_index_dir(index_dir, LOCAL_SOURCE),
         bool(local_dir) and os.path.isdir(local_dir)),
        (CKM_SOURCE, ckm_dir, source_index_dir(index_dir, CKM_SOURCE),
         bool(ckm_dir) and os.path.isdir(ckm_dir)),
    )
    for src_name, src_adl, idx_dir, src_ok in plan:
        if not src_ok:
            st.info(f"Источник {src_name}: полная пересборка пропущена (нет папки с .adl).")
            continue
        os.makedirs(idx_dir, exist_ok=True)
        tmp_dir = os.path.join(idx_dir, ".rebuild_tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        dims: dict[str, int] = {}
        try:
            for kind in ("archetype", "field"):
                title = next(t for t, s, k, _ in INDEX_ROWS if s == src_name and k == kind)
                with st.spinner(f"Полная пересборка: {title}..."):
                    docs, vectors = _rebuild_kind(
                        src_adl,
                        tmp_dir,
                        kind,
                        cfg,
                        source=src_name,
                        full=False,
                        record_build_info=False,
                    )
                dims[kind] = (
                    int(getattr(vectors, "shape", (0, 0))[1])
                    if getattr(vectors, "ndim", 0) == 2
                    else 0
                )
                st.success(f"Готово: {title} = {len(docs)}")
            # Старые файлы не трогаем, пока оба новых индекса не собраны.
            for kind in ("archetype", "field"):
                for fn in index_filenames(kind):
                    src_path = os.path.join(tmp_dir, fn)
                    dst_path = os.path.join(idx_dir, fn)
                    if os.path.exists(src_path):
                        os.replace(src_path, dst_path)
                _record_build_info(index_dir, src_name, kind, cfg, dims.get(kind, 0))
        except Exception as e:
            logger.exception("full rebuild failed for %s", src_name)
            ok = False
            st.error(
                f"Полная пересборка {src_name} не удалась: {e}. "
                "Проверьте индекс и при необходимости повторите."
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    _reload_merged_indexes(cfg, adl_dir, index_dir, ckm_adl_dir)
    return ok


def _render_index_error_highlight(err_local: bool, err_ckm: bool) -> None:
    """Красная подсветка обязательных полей папок, если индекс не запущен из-за пустой папки."""
    rules = []
    for key, on in (("adl_dir", err_local), ("ckm_adl_dir", err_ckm)):
        if not on:
            continue
        rules.append(
            f".st-key-{key} [data-testid='stBaseInput']{{border:2px solid #ff4b4b !important;"
            f"border-radius:6px;box-shadow:0 0 0 1px rgba(255,75,75,.35) !important;}}"
        )
    if rules:
        st.markdown(f"<style>{' '.join(rules)}</style>", unsafe_allow_html=True)


def _render_gigachat_settings(cfg: AppConfig):
    """Поля GigaChat. Ключ авторизации берётся из .env и не сохраняется в llm.json."""
    st.caption(
        "Ключ авторизации храните в `.env` (`GIGACHAT_CREDENTIALS`). "
        "Поле ниже — только для текущей сессии."
    )
    credentials = st.text_input(
        "GigaChat ключ авторизации",
        value=st.session_state.get("gigachat_credentials", cfg.gigachat_credentials),
        type="password",
        key="gigachat_credentials",
        autocomplete="off",
        help="Долгоживущий ключ из личного кабинета GigaChat API.",
    )
    scope_options = ["GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"]
    saved_scope = st.session_state.get("gigachat_scope", cfg.gigachat_scope) or scope_options[0]
    scope = st.selectbox(
        "GigaChat scope",
        options=scope_options,
        index=scope_options.index(saved_scope) if saved_scope in scope_options else 0,
        key="gigachat_scope",
        help="PERS — физлицо (freemium), B2B/CORP — ИП/юрлицо.",
    )
    model_options = list(GIGACHAT_MODEL_OPTIONS)
    saved_model = st.session_state.get("gigachat_model", cfg.gigachat_model) or model_options[0]
    if saved_model not in model_options:
        # Сохранённое значение из прошлых версий не теряем.
        model_options.append(saved_model)
    model = st.selectbox(
        "GigaChat model",
        options=model_options,
        index=model_options.index(saved_model) if saved_model in model_options else 0,
        key="gigachat_model",
        help="Lite — GigaChat-2; далее Pro/Max/Lightning/Ultra.",
    )
    aux_model = st.session_state.get("gigachat_aux_model", "") or ""
    if bool(st.session_state.get("llm_aux_model_enabled", False)):
        aux_options = list(model_options)
        saved_aux = aux_model or model
        if saved_aux not in aux_options:
            aux_options.append(saved_aux)
        aux_model = st.selectbox(
            "GigaChat model (перевод/расширение)",
            options=aux_options,
            index=aux_options.index(saved_aux),
            key="gigachat_aux_model",
            help="Для перевода запроса и предложения расширений. Провайдер тот же.",
        )
    verify_ssl = st.checkbox(
        "Проверять SSL-сертификат",
        value=bool(st.session_state.get("gigachat_verify_ssl_certs", cfg.gigachat_verify_ssl_certs)),
        key="gigachat_verify_ssl_certs",
        help="Для Python нужен корневой сертификат Минцифры. Отключение — только для отладки.",
    )
    ca_bundle = st.text_input(
        "CA-сертификат Минцифры",
        value=st.session_state.get("gigachat_ca_bundle_file", cfg.gigachat_ca_bundle_file),
        key="gigachat_ca_bundle_file",
        help="Путь к Russian_Trusted_Root_CA.cer. Пусто — ищется в папке sertificates рядом с приложением.",
        autocomplete="off",
    )
    with st.expander("GigaChat: дополнительно", expanded=False):
        base_url = st.text_input(
            "Base URL (опционально)",
            value=st.session_state.get("gigachat_base_url", cfg.gigachat_base_url),
            key="gigachat_base_url",
            help="Пусто = https://api.giga.chat/v1",
            autocomplete="off",
        )
        timeout_s = st.number_input(
            "Таймаут запроса (с)",
            min_value=10.0,
            max_value=600.0,
            value=float(st.session_state.get("gigachat_timeout_s", cfg.gigachat_timeout_s)),
            step=10.0,
            key="gigachat_timeout_s",
        )
        temperature = st.slider(
            "GigaChat temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state.get("gigachat_temperature", cfg.gigachat_temperature)),
            step=0.05,
            key="gigachat_temperature",
        )
        max_tokens = st.slider(
            "GigaChat max_tokens",
            min_value=128,
            max_value=8192,
            value=int(st.session_state.get("gigachat_max_tokens", cfg.gigachat_max_tokens)),
            step=128,
            key="gigachat_max_tokens",
        )
        context_window = st.slider(
            "Контекст для ИИ (токенов)",
            min_value=8000,
            max_value=128000,
            value=int(st.session_state.get("gigachat_context_window", cfg.gigachat_context_window)),
            step=8000,
            key="gigachat_context_window",
        )
    return {
        "gigachat_credentials": credentials,
        "gigachat_scope": scope,
        "gigachat_model": model,
        "gigachat_base_url": base_url,
        "gigachat_ca_bundle_file": ca_bundle,
        "gigachat_verify_ssl_certs": verify_ssl,
        "gigachat_timeout_s": timeout_s,
        "gigachat_temperature": temperature,
        "gigachat_max_tokens": max_tokens,
        "gigachat_context_window": context_window,
        "gigachat_aux_model": aux_model,
    }


CORP_GEN_KEYS = (
    "corp_use_temperature",
    "corp_temperature",
    "corp_use_max_tokens",
    "corp_max_tokens",
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
)
CORP_AUX_GEN_KEYS = tuple(f"corp_aux_{key[len('corp_'):]}" for key in CORP_GEN_KEYS)


def _corp_generation_defaults(cfg: AppConfig, prefix: str) -> dict:
    defaults = {
        "temperature": 0.2,
        "max_tokens": 8192,
        "top_p": 0.9,
        "top_k": 40,
        "enable_thinking": False,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    out: dict = {}
    for name, default in defaults.items():
        use_key = f"{prefix}_use_{name}"
        val_key = f"{prefix}_{name}"
        out[use_key] = bool(st.session_state.get(use_key, getattr(cfg, use_key, False)))
        fallback = getattr(cfg, val_key, default)
        value = st.session_state.get(val_key, fallback)
        if isinstance(default, bool):
            out[val_key] = bool(value)
        elif isinstance(default, int):
            out[val_key] = int(value)
        else:
            out[val_key] = float(value)
    return out


def _render_thinking_toggle(prefix: str, raw_json: str, label: str) -> dict:
    """Явный переключатель thinking — только для моделей с его поддержкой.

    Поддержка определяется по ``chat_template_kwargs.enable_thinking`` в
    ``model_kwargs`` (обычно приходит из ``corp_catalog.json``). Если ключа нет,
    переключатель не отрисовывается и параметр не отправляется.
    """
    chat = safe_load_json_dict(raw_json).get("chat_template_kwargs")
    supported = isinstance(chat, dict) and isinstance(chat.get("enable_thinking"), bool)
    use_key = f"{prefix}_use_enable_thinking"
    value_key = f"{prefix}_enable_thinking"
    if not supported:
        st.session_state[use_key] = False
        st.session_state[value_key] = False
        return {use_key: False, value_key: False}
    if value_key not in st.session_state:
        st.session_state[value_key] = bool(chat.get("enable_thinking"))
    enabled = st.toggle(
        label,
        key=value_key,
        help=(
            "Модель поддерживает режим размышлений: в corp_catalog.json задан "
            "chat_template_kwargs.enable_thinking. Отправляется как этот же параметр."
        ),
    )
    st.session_state[use_key] = True
    return {use_key: True, value_key: bool(enabled)}


def _render_corp_generation_settings(
    cfg: AppConfig,
    *,
    title: str,
    key_prefix: str,
    raw_json: str,
    from_catalog: bool,
    catalog_model: str = "",
) -> dict:
    """Параметры генерации для основной или aux-модели corp_async.

    Блокируются только параметры, которые уже заданы в ``model_kwargs``
    (например, из corp_catalog). Остальные можно настроить вручную.
    """
    data = safe_load_json_dict(raw_json)
    chat = data.get("chat_template_kwargs")
    if not isinstance(chat, dict):
        chat = {}

    sync_marker = f"{key_prefix}_catalog_sync_model"
    if from_catalog:
        # Заблокированные значения берем из JSON, отсутствующие — оставляем
        # пользователю (но сбрасываем старый «включено», чтобы не тянуть
        # параметры от предыдущей модели). Синхронизируем только при смене
        # модели: иначе ручные правки чекбоксов сбрасывались бы на каждом rerun.
        if st.session_state.get(sync_marker) != catalog_model:
            def _sync_num(name: str, source: dict, key: str, cast) -> None:
                present = key in source and isinstance(source.get(key), (int, float))
                st.session_state[f"{key_prefix}_use_{name}"] = present
                if present:
                    st.session_state[f"{key_prefix}_{name}"] = cast(source[key])

            _sync_num("temperature", chat, "temperature", float)
            _sync_num("max_tokens", data, "max_tokens", int)
            _sync_num("top_p", data, "top_p", float)
            _sync_num("top_k", data, "top_k", int)
            _sync_num("presence_penalty", data, "presence_penalty", float)
            _sync_num("repetition_penalty", data, "repetition_penalty", float)
            thinking_present = isinstance(chat.get("enable_thinking"), bool)
            st.session_state[f"{key_prefix}_use_enable_thinking"] = thinking_present
            st.session_state[f"{key_prefix}_enable_thinking"] = (
                bool(chat.get("enable_thinking")) if thinking_present else False
            )
            st.session_state[sync_marker] = catalog_model
    else:
        st.session_state.pop(sync_marker, None)

    locked = {
        "temperature": from_catalog and isinstance(chat.get("temperature"), (int, float)),
        "max_tokens": from_catalog and isinstance(data.get("max_tokens"), (int, float)),
        "top_p": from_catalog and isinstance(data.get("top_p"), (int, float)),
        "top_k": from_catalog and isinstance(data.get("top_k"), (int, float)),
        "presence_penalty": from_catalog
        and isinstance(data.get("presence_penalty"), (int, float)),
        "repetition_penalty": from_catalog
        and isinstance(data.get("repetition_penalty"), (int, float)),
    }

    def _session(name: str, default):
        return st.session_state.get(f"{key_prefix}_{name}", default)

    with st.expander(title, expanded=False):
        if from_catalog:
            st.caption(
                "Параметры, заданные в corp_catalog, заблокированы. "
                "Остальные можно настроить вручную."
            )

        use_temperature = st.checkbox(
            "Задать temperature",
            key=f"{key_prefix}_use_temperature",
            help="Передаётся внутри chat_template_kwargs.",
            disabled=locked["temperature"],
        )
        temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(_session("temperature", getattr(cfg, f"{key_prefix}_temperature", cfg.corp_temperature))),
            step=0.05,
            disabled=locked["temperature"] or not use_temperature,
            key=f"{key_prefix}_temperature",
        )
        use_max_tokens = st.checkbox(
            "Задать свой лимит ответа (max_tokens)",
            key=f"{key_prefix}_use_max_tokens",
            help="По умолчанию отправляется 8192 токена, чтобы ответы не обрывались.",
            disabled=locked["max_tokens"],
        )
        max_tokens = st.slider(
            "max_tokens",
            min_value=128,
            max_value=16384,
            value=int(_session("max_tokens", getattr(cfg, f"{key_prefix}_max_tokens", cfg.corp_max_tokens))),
            step=128,
            disabled=locked["max_tokens"] or not use_max_tokens,
            help=(
                "Сколько токенов разрешить на ответ. "
                "Если ответы обрываются — увеличьте (например, 8192)."
            ),
            key=f"{key_prefix}_max_tokens",
        )
        use_top_p = st.checkbox(
            "Задать top_p",
            key=f"{key_prefix}_use_top_p",
            help="Nucleus sampling: доля вероятностной массы для выбора токенов.",
            disabled=locked["top_p"],
        )
        top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=float(_session("top_p", getattr(cfg, f"{key_prefix}_top_p", cfg.corp_top_p))),
            step=0.05,
            disabled=locked["top_p"] or not use_top_p,
            key=f"{key_prefix}_top_p",
        )
        use_top_k = st.checkbox(
            "Задать top_k",
            key=f"{key_prefix}_use_top_k",
            help="Ограничивает выбор k наиболее вероятными токенами.",
            disabled=locked["top_k"],
        )
        top_k = st.slider(
            "top_k",
            min_value=0,
            max_value=200,
            value=int(_session("top_k", getattr(cfg, f"{key_prefix}_top_k", cfg.corp_top_k))),
            step=1,
            disabled=locked["top_k"] or not use_top_k,
            key=f"{key_prefix}_top_k",
        )
        use_presence_penalty = st.checkbox(
            "Задать presence_penalty",
            key=f"{key_prefix}_use_presence_penalty",
            help=(
                "Штраф за появление новых тем: положительное значение "
                "поощряет новые темы, отрицательное — повторение."
            ),
            disabled=locked["presence_penalty"],
        )
        presence_penalty = st.slider(
            "presence_penalty",
            min_value=-2.0,
            max_value=2.0,
            value=float(
                _session(
                    "presence_penalty",
                    getattr(cfg, f"{key_prefix}_presence_penalty", cfg.corp_presence_penalty),
                )
            ),
            step=0.1,
            disabled=locked["presence_penalty"] or not use_presence_penalty,
            key=f"{key_prefix}_presence_penalty",
        )
        use_repetition_penalty = st.checkbox(
            "Задать repetition_penalty",
            key=f"{key_prefix}_use_repetition_penalty",
            help="Больше 1.0 уменьшает повторы, меньше 1.0 — поощряет их.",
            disabled=locked["repetition_penalty"],
        )
        repetition_penalty = st.slider(
            "repetition_penalty",
            min_value=0.5,
            max_value=2.0,
            value=float(
                _session(
                    "repetition_penalty",
                    getattr(cfg, f"{key_prefix}_repetition_penalty", cfg.corp_repetition_penalty),
                )
            ),
            step=0.05,
            disabled=locked["repetition_penalty"] or not use_repetition_penalty,
            key=f"{key_prefix}_repetition_penalty",
        )

    return {
        f"{key_prefix}_use_temperature": bool(use_temperature),
        f"{key_prefix}_temperature": float(temperature),
        f"{key_prefix}_use_max_tokens": bool(use_max_tokens),
        f"{key_prefix}_max_tokens": int(max_tokens),
        f"{key_prefix}_use_top_p": bool(use_top_p),
        f"{key_prefix}_top_p": float(top_p),
        f"{key_prefix}_use_top_k": bool(use_top_k),
        f"{key_prefix}_top_k": int(top_k),
        f"{key_prefix}_use_presence_penalty": bool(use_presence_penalty),
        f"{key_prefix}_presence_penalty": float(presence_penalty),
        f"{key_prefix}_use_repetition_penalty": bool(use_repetition_penalty),
        f"{key_prefix}_repetition_penalty": float(repetition_penalty),
    }



def _render_api_base_key_model(
    cfg: AppConfig,
    catalog: list[dict],
    provider_options: list[str],
    *,
    kind: str = CORP_API_KIND,
):
    """
    Поля подключения для API-режима.

    У corp_async (МЭШ.ИИ.Хаб) и OpenAI-совместимого API отдельные настройки,
    чтобы данные МЭШ не подставлялись в OpenAI-режим.
    Возвращает dict с api_base_url, api_key, api_model, corp_provider,
    openai_base_url, openai_api_key, openai_model.
    """
    out = {
        "api_base_url": st.session_state.get("api_base_url", ""),
        "api_key": st.session_state.get("api_key", ""),
        "api_model": st.session_state.get("api_model", ""),
        "api_aux_model": st.session_state.get("api_aux_model", ""),
        "corp_provider": st.session_state.get("corp_provider", "") or "",
        "corp_aux_provider": st.session_state.get("corp_aux_provider", "") or "",
        "corp_aux_model_kwargs_json": st.session_state.get("corp_aux_model_kwargs_json", "{}") or "{}",
        "openai_base_url": st.session_state.get("openai_base_url", ""),
        "openai_api_key": st.session_state.get("openai_api_key", ""),
        "openai_model": st.session_state.get("openai_model", ""),
        "openai_aux_model": st.session_state.get("openai_aux_model", ""),
    }

    if kind == "openai_compatible":
        out["openai_base_url"] = st.text_input(
            "API base url",
            value=st.session_state.get("openai_base_url", ""),
            help="Напр.: https://host.example/v1",
            key="openai_base_url",
            autocomplete="openai_base_url",
        )
        out["openai_api_key"] = st.text_input(
            "API key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            help="Храните в .env (`OPENAI_API_KEY`); поле — только для текущей сессии.",
            key="openai_api_key",
            autocomplete="off",
        )
        out["openai_model"] = st.text_input(
            "API model",
            value=st.session_state.get("openai_model", ""),
            key="openai_model",
            autocomplete="openai_model",
        )
        if bool(st.session_state.get("llm_aux_model_enabled", False)):
            out["openai_aux_model"] = st.text_input(
                "API model (перевод/расширение)",
                value=(
                    st.session_state.get("openai_aux_model", "")
                    or st.session_state.get("openai_model", "")
                ),
                key="openai_aux_model",
                autocomplete="openai_aux_model",
                help="Модель для перевода запроса и расширения. Пусто = основная модель.",
            )
        return out

    # corp_async (МЭШ.ИИ.Хаб)
    out["api_base_url"] = st.text_input(
        "API base url",
        value=st.session_state.get("api_base_url", ""),
        help="Полный URL async endpoint, напр.: https://.../async/v1",
        key="api_base_url",
        autocomplete="api_base_url",
    )
    out["api_key"] = st.text_input(
        "API key",
        value=st.session_state.get("api_key", ""),
        type="password",
        help="Храните в .env (`LLM_API_KEY`); поле — только для текущей сессии.",
        key="api_key",
        autocomplete="off",
    )

    catalog_names = [x.get("model_name") for x in (catalog or []) if x.get("model_name")]
    saved_model_name = (st.session_state.get("corp_model_name") or "").strip()
    if (
        saved_model_name
        and saved_model_name != "(вручную)"
        and saved_model_name not in catalog_names
    ):
        catalog_names.append(saved_model_name)
    model_options = ["(вручную)"] + catalog_names
    st.selectbox(
        "Модель",
        options=model_options,
        index=model_options.index(saved_model_name) if saved_model_name in model_options else 0,
        help=(
            "Модель из corp_catalog.json. Провайдер, API model и параметры "
            "подставятся из каталога; model_kwargs можно поправить вручную."
        ),
        key="corp_model_name",
        on_change=apply_corp_catalog_selection,
        args=(catalog,),
    )
    catalog_model_names = {
        str(x.get("model_name")) for x in (catalog or []) if x.get("model_name")
    }
    selected_model_name = (st.session_state.get("corp_model_name") or "").strip()
    from_catalog = selected_model_name in catalog_model_names
    out["corp_provider"] = st.selectbox(
        "API provider",
        options=provider_options,
        index=(
            provider_options.index(st.session_state.get("corp_provider", ""))
            if st.session_state.get("corp_provider", "") in provider_options
            else 0
        ),
        help=(
            "Берётся из corp_catalog.json для выбранной модели. "
            "Редактируется только в режиме «(вручную)»."
        ),
        key="corp_provider",
        disabled=from_catalog,
    )
    out["api_model"] = st.text_input(
        "API model",
        value=st.session_state.get("api_model", ""),
        help=(
            "Берётся из corp_catalog.json. Редактируется только в режиме «(вручную)». "
            "Для corp_async отправляется как attributes.model."
        ),
        key="api_model",
        autocomplete="api_model",
        disabled=from_catalog,
    )
    main_kwargs_json = st.text_area(
        "model_kwargs (JSON)",
        value=st.session_state.get("corp_model_kwargs_json", "{}") or "{}",
        height=120,
        help=(
            "Базовый JSON (cluster/service_url). temperature и "
            "enable_thinking добавляются в chat_template_kwargs, "
            "остальные параметры — на верхний уровень."
        ),
        key="corp_model_kwargs_json",
    )
    out["corp_model_kwargs_json"] = main_kwargs_json
    out.update(
        _render_corp_generation_settings(
            cfg,
            title="Параметры генерации",
            key_prefix="corp",
            raw_json=main_kwargs_json,
            from_catalog=from_catalog,
            catalog_model=selected_model_name,
        )
    )
    out.update(
        _render_thinking_toggle("corp", main_kwargs_json, "Размышления (thinking)")
    )
    if bool(st.session_state.get("llm_aux_model_enabled", False)):
        st.markdown("**Модель для перевода и расширения**")
        aux_catalog_names = [
            x.get("model_name") for x in (catalog or []) if x.get("model_name")
        ]
        saved_aux_name = (st.session_state.get("corp_aux_model_name") or "").strip()
        if saved_aux_name and saved_aux_name not in aux_catalog_names:
            # Модель исчезла из каталога — сбрасываем на основную.
            st.session_state["corp_aux_model_name"] = "(вручную)"
            saved_aux_name = "(вручную)"
            st.session_state["api_aux_model"] = st.session_state.get("api_model", "")
            st.session_state["corp_aux_provider"] = ""
            st.session_state["corp_aux_model_kwargs_json"] = st.session_state.get(
                "corp_model_kwargs_json", "{}"
            )
        aux_options = ["(вручную)"] + aux_catalog_names
        st.selectbox(
            "Модель (перевод/расширение)",
            options=aux_options,
            index=aux_options.index(saved_aux_name) if saved_aux_name in aux_options else 0,
            key="corp_aux_model_name",
            on_change=apply_corp_aux_catalog_selection,
            args=(catalog,),
            help=(
                "Можно выбрать модель любого провайдера МЭШ.ИИ.Хаб "
                "(aiedu-v2 или mosgpt): провайдер подставится автоматически."
            ),
        )
        aux_from_catalog = saved_aux_name in {
            str(x.get("model_name")) for x in (catalog or []) if x.get("model_name")
        }
        out["api_aux_model"] = st.text_input(
            "API model (перевод/расширение)",
            value=(
                st.session_state.get("api_aux_model", "")
                or st.session_state.get("api_model", "")
            ),
            key="api_aux_model",
            autocomplete="api_aux_model",
            help=(
                "Берётся из corp_catalog.json. Редактируется только в режиме "
                "«(вручную)». Пусто = основная модель."
            ),
            disabled=aux_from_catalog,
        )
        out["corp_aux_provider"] = st.session_state.get("corp_aux_provider", "") or ""
        aux_kwargs_json = st.text_area(
            "model_kwargs (JSON, перевод/расширение)",
            value=(
                st.session_state.get("corp_aux_model_kwargs_json", "")
                or st.session_state.get("corp_model_kwargs_json", "{}")
                or "{}"
            ),
            height=90,
            key="corp_aux_model_kwargs_json",
            help="Пусто = как у основной модели.",
        )
        out["corp_aux_model_kwargs_json"] = aux_kwargs_json
        out.update(
            _render_corp_generation_settings(
                cfg,
                title="Параметры генерации (перевод/расширение)",
                key_prefix="corp_aux",
                raw_json=aux_kwargs_json,
                from_catalog=aux_from_catalog,
                catalog_model=saved_aux_name,
            )
        )
        out.update(
            _render_thinking_toggle(
                "corp_aux",
                aux_kwargs_json,
                "Размышления (thinking) — перевод/расширение",
            )
        )
    return out


def _state_from_config(cfg: AppConfig, settings: dict) -> SidebarState:
    """Состояние без виджетов — для пользовательского режима (значения из env/settings)."""
    return SidebarState(
        adl_dir=(settings.get("adl_dir") or cfg.adl_dir or ""),
        index_dir=(cfg.index_dir or settings.get("index_dir") or ""),
        ckm_adl_dir=(settings.get("ckm_adl_dir") or cfg.ckm_adl_dir or ""),
        auto_strictness=float(settings.get("auto_strictness", cfg.auto_strictness) or cfg.auto_strictness),
        keyword_bonus=float(cfg.keyword_bonus),
        phrase_bonus=float(cfg.phrase_bonus),
        misuse_penalty=float(cfg.misuse_penalty),
        weak_match_penalty=float(cfg.weak_match_penalty),
        enable_misuse_penalty=bool(cfg.enable_misuse_penalty),
        enable_weak_match_penalty=bool(cfg.enable_weak_match_penalty),
        max_results=int(cfg.max_results),
        sim_floor=float(cfg.sim_floor),
        sim_floor_pure=float(cfg.sim_floor_pure),
        llm_mode=cfg.llm_mode,
        ollama_model=cfg.ollama_model,
        ollama_base_url=cfg.ollama_base_url,
        api_kind=cfg.api_kind,
        chat_provider=cfg.chat_provider,
        api_base_url=cfg.api_base_url,
        api_key=cfg.api_key,
        api_model=cfg.api_model,
        openai_base_url=cfg.openai_base_url,
        openai_api_key=cfg.openai_api_key,
        openai_model=cfg.openai_model,
        corp_provider=cfg.corp_provider,
        corp_model_kwargs_json=cfg.corp_model_kwargs_json,
        corp_poll_timeout_s=int(cfg.corp_poll_timeout_s),
        corp_poll_interval_s=float(cfg.corp_poll_interval_s),
        corp_use_temperature=bool(cfg.corp_use_temperature),
        corp_temperature=float(cfg.corp_temperature),
        corp_use_max_tokens=bool(cfg.corp_use_max_tokens),
        corp_max_tokens=int(cfg.corp_max_tokens),
        corp_use_top_p=bool(cfg.corp_use_top_p),
        corp_top_p=float(cfg.corp_top_p),
        corp_use_top_k=bool(cfg.corp_use_top_k),
        corp_top_k=int(cfg.corp_top_k),
        corp_use_enable_thinking=bool(cfg.corp_use_enable_thinking),
        corp_enable_thinking=bool(cfg.corp_enable_thinking),
        corp_use_presence_penalty=bool(cfg.corp_use_presence_penalty),
        corp_presence_penalty=float(cfg.corp_presence_penalty),
        corp_use_repetition_penalty=bool(cfg.corp_use_repetition_penalty),
        corp_repetition_penalty=float(cfg.corp_repetition_penalty),
        corp_aux_use_temperature=bool(cfg.corp_aux_use_temperature),
        corp_aux_temperature=float(cfg.corp_aux_temperature),
        corp_aux_use_max_tokens=bool(cfg.corp_aux_use_max_tokens),
        corp_aux_max_tokens=int(cfg.corp_aux_max_tokens),
        corp_aux_use_top_p=bool(cfg.corp_aux_use_top_p),
        corp_aux_top_p=float(cfg.corp_aux_top_p),
        corp_aux_use_top_k=bool(cfg.corp_aux_use_top_k),
        corp_aux_top_k=int(cfg.corp_aux_top_k),
        corp_aux_use_enable_thinking=bool(cfg.corp_aux_use_enable_thinking),
        corp_aux_enable_thinking=bool(cfg.corp_aux_enable_thinking),
        corp_aux_use_presence_penalty=bool(cfg.corp_aux_use_presence_penalty),
        corp_aux_presence_penalty=float(cfg.corp_aux_presence_penalty),
        corp_aux_use_repetition_penalty=bool(cfg.corp_aux_use_repetition_penalty),
        corp_aux_repetition_penalty=float(cfg.corp_aux_repetition_penalty),
        gigachat_credentials=cfg.gigachat_credentials,
        gigachat_scope=cfg.gigachat_scope,
        gigachat_model=cfg.gigachat_model,
        gigachat_base_url=cfg.gigachat_base_url,
        gigachat_ca_bundle_file=cfg.gigachat_ca_bundle_file,
        gigachat_verify_ssl_certs=bool(cfg.gigachat_verify_ssl_certs),
        gigachat_timeout_s=float(cfg.gigachat_timeout_s),
        gigachat_temperature=float(cfg.gigachat_temperature),
        gigachat_max_tokens=int(cfg.gigachat_max_tokens),
        gigachat_context_window=int(cfg.gigachat_context_window),
        llm_aux_model_enabled=bool(getattr(cfg, "llm_aux_model_enabled", False)),
        ollama_aux_model=getattr(cfg, "ollama_aux_model", "") or "",
        api_aux_model=getattr(cfg, "api_aux_model", "") or "",
        openai_aux_model=getattr(cfg, "openai_aux_model", "") or "",
        corp_aux_provider=getattr(cfg, "corp_aux_provider", "") or "",
        corp_aux_model_kwargs_json=getattr(cfg, "corp_aux_model_kwargs_json", "") or "{}",
        gigachat_aux_model=getattr(cfg, "gigachat_aux_model", "") or "",
    )


_BLOCK_INDEX_KEYS = ("adl_dir", "ckm_adl_dir")
_LLM_UI_KEYS = ("corp_model_name", "corp_aux_model_name", "corp_context_window")
_PROMPT_SESSION_KEYS = (
    "system_prompt",
    *(f"prompt_{key}" for key in PROMPT_PART_KEYS),
)


def _dev_flash(kind: str, text: str) -> None:
    st.session_state["_dev_settings_msg"] = (kind, text)


def _render_dev_flash() -> None:
    data = st.session_state.pop("_dev_settings_msg", None)
    if not data:
        return
    kind, text = data
    if kind == "error":
        st.error(text)
    else:
        st.success(text)


def _save_settings_block(
    cfg: AppConfig, keys: tuple[str, ...], message: str
) -> None:
    """Записывает ключи одного блока в config/settings.json."""
    config_dir = cfg.config_dir or "."
    payload = dict(load_settings(config_dir))
    for key in keys:
        if key in st.session_state:
            payload[key] = st.session_state[key]
    save_settings(config_dir, payload)
    _dev_flash("ok", message)


def _save_llm_block(cfg: AppConfig) -> None:
    config_dir = cfg.config_dir or "."
    payload = dict(load_llm_settings(config_dir))
    for key in (*LLM_SETTINGS_TYPES, *_LLM_UI_KEYS):
        if key in st.session_state:
            payload[key] = st.session_state[key]
    payload["corp_model_name"] = (
        payload.get("corp_model_name") or LLM_UI_DEFAULTS["corp_model_name"]
    )
    payload["corp_aux_model_name"] = (
        payload.get("corp_aux_model_name") or LLM_UI_DEFAULTS["corp_aux_model_name"]
    )
    payload["corp_context_window"] = (
        int(payload.get("corp_context_window") or 0)
        or LLM_UI_DEFAULTS["corp_context_window"]
    )
    save_llm_settings(config_dir, payload)
    _dev_flash("ok", "Настройки LLM сохранены")


def _save_prompts_block(cfg: AppConfig) -> None:
    try:
        save_prompts(cfg.config_dir or ".", prompts_from_session())
    except PromptConfigError as e:
        _dev_flash("error", str(e))
        return
    except Exception:
        logger.exception("cannot save prompts")
        _dev_flash("error", "Не удалось сохранить промпты.")
        return
    _dev_flash("ok", "Промпты сохранены")


def _reset_settings_block(cfg: AppConfig, keys: tuple[str, ...], message: str) -> None:
    """Сбрасывает черновик блока к значениям конфига (файл не меняется)."""
    values = load_settings(cfg.config_dir)
    for key in keys:
        if key in values:
            st.session_state[key] = values[key]
            continue
        default = getattr(cfg, key, None)
        if default is None:
            st.session_state.pop(key, None)
        else:
            st.session_state[key] = default
    _dev_flash("ok", message)


def _reset_llm_block(cfg: AppConfig) -> None:
    values = load_llm_settings(cfg.config_dir)
    for key in LLM_SETTINGS_TYPES:
        if key in values:
            st.session_state[key] = values[key]
        else:
            default = getattr(cfg, key, None)
            if default is not None:
                st.session_state[key] = default
    for key in _LLM_UI_KEYS:
        st.session_state[key] = values.get(key, LLM_UI_DEFAULTS[key])
    _dev_flash("ok", "Настройки LLM сброшены к конфигу")


def _reset_prompts_block(cfg: AppConfig) -> bool:
    try:
        values = load_prompts(cfg.config_dir or ".")
    except PromptConfigError as e:
        _dev_flash("error", str(e))
        return False
    st.session_state["system_prompt"] = values.get(
        "system_prompt", st.session_state.get("system_prompt", "")
    )
    for key in PROMPT_PART_KEYS:
        st.session_state[f"prompt_{key}"] = values.get(key, "")
    return True


def _capture_baseline(keys) -> dict:
    return {key: st.session_state.get(key) for key in keys}


def _update_baseline(name: str, keys) -> None:
    data = st.session_state.setdefault("_dev_baseline", {})
    data[name] = _capture_baseline(keys)


def _block_is_dirty(name: str, keys) -> bool:
    data = st.session_state.get("_dev_baseline") or {}
    base = data.get(name)
    if base is None:
        return False
    for key in keys:
        if key not in st.session_state:
            continue
        current = st.session_state.get(key)
        if str(current) != str(base.get(key)):
            return True
    return False


def _render_block_actions(
    *,
    label: str,
    dirty: bool,
    save_key: str,
    reset_key: str,
    on_save,
    on_reset,
    save_args: tuple,
    reset_args: tuple,
) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "Сохранить",
            key=save_key,
            use_container_width=True,
            on_click=on_save,
            args=save_args,
        )
    with c2:
        st.button(
            "Сбросить",
            key=reset_key,
            use_container_width=True,
            on_click=on_reset,
            args=reset_args,
        )
    if dirty:
        st.caption(
            '<span style="color:#d13438;font-weight:600">'
            f"• {label}: есть несохранённые изменения"
            "</span>",
            unsafe_allow_html=True,
        )


def _save_search_action(cfg: AppConfig) -> None:
    _save_settings_block(
        cfg, tuple(SEARCH_SETTINGS_TYPES), "Настройки поиска сохранены"
    )
    _update_baseline("search", tuple(SEARCH_SETTINGS_TYPES))


def _save_index_action(cfg: AppConfig) -> None:
    _save_settings_block(cfg, _BLOCK_INDEX_KEYS, "Пути индексации сохранены")
    _update_baseline("index", _BLOCK_INDEX_KEYS)


def _save_all_action(cfg: AppConfig) -> None:
    _save_search_action(cfg)
    _save_index_action(cfg)
    _save_llm_block(cfg)
    _update_baseline("llm", (*LLM_SETTINGS_TYPES, *_LLM_UI_KEYS))
    _save_prompts_block(cfg)
    _update_baseline("prompts", _PROMPT_SESSION_KEYS)
    _dev_flash("ok", "Все настройки сохранены")


def _reset_search_action(cfg: AppConfig) -> None:
    _reset_settings_block(
        cfg, tuple(SEARCH_SETTINGS_TYPES), "Настройки поиска сброшены к конфигу"
    )
    _update_baseline("search", tuple(SEARCH_SETTINGS_TYPES))


def _search_factory_defaults(cfg: AppConfig) -> dict:
    """Значения поиска из кода/окружения — без учёта config/settings.json."""
    base = from_env(default_adl_dir=cfg.adl_dir)
    return {key: getattr(base, key) for key in SEARCH_SETTINGS_TYPES}


def _reset_search_defaults_action(cfg: AppConfig) -> None:
    """Черновик поиска → заводские значения из кода (в файл не пишет)."""
    for key, value in _search_factory_defaults(cfg).items():
        st.session_state[key] = value
    _dev_flash(
        "ok",
        "Настройки поиска сброшены к заводским. Нажмите «Сохранить», чтобы записать их в конфиг.",
    )


def _reset_index_action(cfg: AppConfig) -> None:
    _reset_settings_block(cfg, _BLOCK_INDEX_KEYS, "Пути индексации сброшены к конфигу")
    _update_baseline("index", _BLOCK_INDEX_KEYS)


def _reset_llm_action(cfg: AppConfig) -> None:
    _reset_llm_block(cfg)
    _update_baseline("llm", (*LLM_SETTINGS_TYPES, *_LLM_UI_KEYS))


def _reset_prompts_action(cfg: AppConfig) -> None:
    if _reset_prompts_block(cfg):
        _update_baseline("prompts", _PROMPT_SESSION_KEYS)
        _dev_flash("ok", "Промпты сброшены к конфигу")


def _reset_all_action(cfg: AppConfig) -> None:
    """Черновики всех блоков → значения из config/. Файлы не меняются."""
    _reset_settings_block(cfg, tuple(SEARCH_SETTINGS_TYPES), "")
    _reset_settings_block(cfg, _BLOCK_INDEX_KEYS, "")
    _reset_llm_block(cfg)
    _update_baseline("search", tuple(SEARCH_SETTINGS_TYPES))
    _update_baseline("index", _BLOCK_INDEX_KEYS)
    _update_baseline("llm", (*LLM_SETTINGS_TYPES, *_LLM_UI_KEYS))
    if not _reset_prompts_block(cfg):
        _dev_flash(
            "error",
            "Остальные настройки сброшены к конфигу, промпты сбросить не удалось.",
        )
        return
    _update_baseline("prompts", _PROMPT_SESSION_KEYS)
    _dev_flash("ok", "Все настройки сброшены к конфигу")


# Приблизительные размеры поля промпта: ширина сайдбара ~45 символов,
# строка ~20 px. Подстройка статическая — при следующем открытии настроек.
_PROMPT_CHARS_PER_LINE = 45
_PROMPT_LINE_HEIGHT_PX = 20
_PROMPT_FIELD_MIN_HEIGHT = 120
_PROMPT_FIELD_MAX_HEIGHT = 1000

# Точная подгонка высоты полей промптов под содержимое: ширина сайдбара
# меняется, поэтому статическая оценка может промахнуться. Скрипт отрабатывает
# при открытии настроек и при раскрытии блока «Промпты».
_PROMPT_FIELDS_JS = """
<style>html,body{margin:0;padding:0;height:0;overflow:hidden}</style>
<script>
setTimeout(() => {
  try {
    const doc = window.parent.document;
    const box = doc.querySelector('.st-key-prompt_fields');
    if (!box) return;
    const fit = () => {
      for (const ta of box.querySelectorAll('textarea')) {
        // height:auto даёт минимум в 2 строки, поэтому меряем с нулевой высотой.
        ta.style.height = '0px';
        const need = ta.scrollHeight;
        if (!need) continue;
        ta.style.height = Math.min(need + 2, __MAX__) + 'px';
      }
    };
    fit();
    const details = box.closest('details');
    if (details && !details.__adlPromptFit) {
      details.__adlPromptFit = true;
      details.addEventListener('toggle', () => {
        if (details.open) setTimeout(fit, 30);
      });
    }
  } catch (e) {}
}, 120);
</script>
"""


def prompt_field_height(text: str) -> int:
    """Высота text_area под текущий текст промпта (без динамического слежения)."""
    lines = 0
    for line in (text or "").splitlines() or [""]:
        lines += max(1, -(-len(line) // _PROMPT_CHARS_PER_LINE))
    lines = max(1, lines)
    height = _PROMPT_FIELD_MIN_HEIGHT + (lines - 1) * _PROMPT_LINE_HEIGHT_PX
    return max(_PROMPT_FIELD_MIN_HEIGHT, min(_PROMPT_FIELD_MAX_HEIGHT, height))


def render_dev_settings(
    cfg: AppConfig,
    settings: dict,
    dev_mode: bool,
) -> SidebarState:
    """Настройки разработчика в свёрнутом блоке над разделами.

    В пользовательском режиме виджеты не рисуются: значения берутся из env/settings.
    В dev-режиме «Сохранить» пишет в ``config/`` текущего окружения: папку проекта
    при запуске из Python или репозиторий, выбранный при запуске exe.

    Изменения виджетов живут в ``session_state`` как черновик и применяются сразу,
    но в файлы попадают только по кнопке «Сохранить» (или «Сохранить всё»).
    """
    if not dev_mode:
        return _state_from_config(cfg, settings)

    catalog = load_ui_catalog(cfg)
    provider_options = corp_provider_options(cfg)
    # Сохранённый провайдер может отсутствовать в каталоге на этой машине — не теряем его.
    saved_provider = (st.session_state.get("corp_provider") or "").strip()
    if saved_provider and saved_provider not in provider_options:
        provider_options.append(saved_provider)

    if dev_mode:
        done = st.session_state.pop("full_rebuild_done", None)
        if done == "ok":
            st.success("Полная пересборка индексов завершена.")
        elif done == "error":
            st.error(
                "Полная пересборка индексов завершилась с ошибкой. "
                "Подробности в logs/app.log."
            )
    if dev_mode and st.session_state.get("full_rebuild_confirm"):
        st.info(
            "Полная пересборка индексов запущена. Это может занять несколько минут — "
            "не закрывайте вкладку."
        )

    with st.expander("Настройки (dev)", expanded=False):
        _render_dev_flash()
        st.caption(
            "Режим разработчика · изменения применяются сразу, но записываются "
            "в config/ репозитория (или проекта) только по кнопке «Сохранить»."
        )
        c_save, c_reset = st.columns(2)
        with c_save:
            st.button(
                "Сохранить всё",
                use_container_width=True,
                key="save_all_settings_btn",
                help="Записать текущие значения всех блоков в конфиг.",
                on_click=_save_all_action,
                args=(cfg,),
            )
        with c_reset:
            st.button(
                "Сбросить всё",
                use_container_width=True,
                key="reset_all_settings_btn",
                help=(
                    "Вернуть значения всех блоков к config/. Файлы не меняются; "
                    "чтобы записать, нажмите «Сохранить всё»."
                ),
                on_click=_reset_all_action,
                args=(cfg,),
            )

        if dev_mode:
            with st.expander("Индексация", expanded=False):
                st.markdown("**Локальная база архетипов**")
                c_a1, c_a2 = st.columns([5, 2])
                with c_a1:
                    adl_dir = st.text_input(
                        "Папка с .adl",
                        value=(settings.get("adl_dir") or cfg.adl_dir or ""),
                        label_visibility="collapsed",
                        autocomplete="adl_dir",
                        key="adl_dir",
                    )
                with c_a2:
                    st.button("Обзор", use_container_width=True, key="browse_adl_dir_btn", on_click=browse_adl_dir)

                st.markdown("**Архетипы CKM**")
                c_c1, c_c2 = st.columns([5, 2])
                with c_c1:
                    ckm_adl_dir = st.text_input(
                        "Папка CKM",
                        value=(settings.get("ckm_adl_dir") or cfg.ckm_adl_dir or ""),
                        help="Зеркало международной базы архетипов (например, C:/CKM-mirror/local/archetypes).",
                        label_visibility="collapsed",
                        autocomplete="ckm_adl_dir",
                        key="ckm_adl_dir",
                    )
                with c_c2:
                    st.button(
                        "Обзор",
                        use_container_width=True,
                        key="browse_ckm_dir_btn",
                        on_click=browse_dir,
                        args=("ckm_adl_dir", "Выберите папку CKM"),
                    )

                # Убираем выбор языка: работаем только с русскими терминами
                if cfg.repo_root:
                    index_dir = cfg.index_dir
                    st.caption(f"Папка индексов: `{index_dir}` (из подключённого репозитория)")
                else:
                    index_dir = st.text_input(
                        "Папка индексов",
                        value=(settings.get("index_dir") or cfg.index_dir),
                        help="Корень дерева индексов: внутри будут подпапки local/ и ckm/ (обычно: ./data).",
                        autocomplete="index_dir",
                        key="index_dir",
                    )

                local_dir = (adl_dir or "").strip()
                ckm_dir_s = (ckm_adl_dir or "").strip()
                local_folder_ok = bool(local_dir) and os.path.isdir(local_dir)
                ckm_folder_ok = bool(ckm_dir_s) and os.path.isdir(ckm_dir_s)
                local_idx_dir = source_index_dir(index_dir or "./data", LOCAL_SOURCE)
                ckm_idx_dir = source_index_dir(index_dir or "./data", CKM_SOURCE)

                st.markdown("**Индексы**")
                row_clicks: dict[tuple[str, str], bool] = {}
                c_local, c_ckm = st.columns(2)
                for src_name, src_label, src_col in (
                    (LOCAL_SOURCE, "Локальная база", c_local),
                    (CKM_SOURCE, "CKM", c_ckm),
                ):
                    with src_col:
                        st.markdown(f"**{src_label}**")
                        for kind, kind_label in (("archetype", "Архетипы"), ("field", "Поля")):
                            idx_dir = local_idx_dir if src_name == LOCAL_SOURCE else ckm_idx_dir
                            _, _, _, tip = next(
                                row for row in INDEX_ROWS if row[1] == src_name and row[2] == kind
                            )
                            c_btn, c_txt = st.columns([1, 8], vertical_alignment="center")
                            with c_btn:
                                row_clicks[(src_name, kind)] = st.button(
                                    "",
                                    icon=":material/refresh:",
                                    help=f"{tip} (инкрементально)",
                                    key=f"idx_{src_name}_{kind}_btn",
                                    use_container_width=True,
                                )
                            with c_txt:
                                st.caption(f"{kind_label}: {index_last_built_label(idx_dir, kind)}")

                b_all_col, b_full_col = st.columns([1, 1])
                with b_all_col:
                    b_index_all = st.button(
                        "Индексировать всё",
                        use_container_width=True,
                        key="idx_all_btn",
                        type="primary",
                    )
                with b_full_col:
                    b_full_rebuild = st.button(
                        "Полная пересборка…",
                        use_container_width=True,
                        key="idx_full_rebuild_btn",
                    )

                if st.session_state.get("full_rebuild_confirm"):
                    st.session_state["full_rebuild_confirm"] = False
                    st.session_state.pop("full_rebuild_done", None)
                    st.info(
                        "Полная пересборка индексов запущена. Это может занять несколько минут — "
                        "не закрывайте вкладку."
                    )
                    ok = _run_full_rebuild(cfg, adl_dir, ckm_adl_dir, index_dir)
                    st.session_state["full_rebuild_done"] = "ok" if ok else "error"
                    st.rerun()

                idx_err_local = False
                idx_err_ckm = False

                for title, src_name, kind, _tip in INDEX_ROWS:
                    if not row_clicks.get((src_name, kind)):
                        continue
                    if src_name == LOCAL_SOURCE:
                        src_adl, src_ok = local_dir, local_folder_ok
                        idx_dir = local_idx_dir
                    else:
                        src_adl, src_ok = ckm_dir_s, ckm_folder_ok
                        idx_dir = ckm_idx_dir
                    if not src_ok:
                        if src_name == LOCAL_SOURCE:
                            st.error("Укажите папку с .adl (локальная база), чтобы индексировать данные.")
                            idx_err_local = True
                        else:
                            st.error("Укажите папку CKM (.adl), чтобы индексировать данные CKM.")
                            idx_err_ckm = True
                        continue
                    if kind == "field" and not index_kind_files_exist(idx_dir, "archetype"):
                        st.warning(f"Сначала постройте индекс архетипов ({title}): он нужен как основа.")
                        continue
                    with st.spinner(f"Индексация: {title}..."):
                        docs, _ = _rebuild_kind(src_adl, idx_dir, kind, cfg, source=src_name)
                    _reload_merged_indexes(cfg, adl_dir, index_dir, ckm_adl_dir)
                    st.success(f"Готово: {title} = {len(docs)}")

                if idx_err_local or idx_err_ckm:
                    _render_index_error_highlight(idx_err_local, idx_err_ckm)

                if b_index_all:
                    any_built = False
                    for src_name, src_adl, src_ok, idx_dir in (
                        (LOCAL_SOURCE, local_dir, local_folder_ok, local_idx_dir),
                        (CKM_SOURCE, ckm_dir_s, ckm_folder_ok, ckm_idx_dir),
                    ):
                        if not src_ok:
                            st.info(
                                f"Источник {src_name}: пропущен (не указана папка с .adl)."
                            )
                            continue
                        for kind in ("archetype", "field"):
                            title = next(t for t, s, k, _ in INDEX_ROWS if s == src_name and k == kind)
                            with st.spinner(f"Индексация: {title}..."):
                                docs, _ = _rebuild_kind(src_adl, idx_dir, kind, cfg, source=src_name)
                            any_built = True
                            st.success(f"Готово: {title} = {len(docs)}")
                    if any_built:
                        _reload_merged_indexes(cfg, adl_dir, index_dir, ckm_adl_dir)

                if b_full_rebuild:
                    st.session_state["full_rebuild_open"] = True

                _render_block_actions(
                    label="Индексация",
                    dirty=_block_is_dirty("index", _BLOCK_INDEX_KEYS),
                    save_key="save_index_settings_btn",
                    reset_key="reset_index_settings_btn",
                    on_save=_save_index_action,
                    on_reset=_reset_index_action,
                    save_args=(cfg,),
                    reset_args=(cfg,),
                )
        else:
            adl_dir = (settings.get("adl_dir") or cfg.adl_dir or "")
            index_dir = (settings.get("index_dir") or cfg.index_dir or "")
            ckm_adl_dir = (settings.get("ckm_adl_dir") or cfg.ckm_adl_dir or "")

        if dev_mode:
            with st.expander("Поиск", expanded=False):
                st.markdown("**Отбор результатов**")
                auto_strictness = st.slider(
                    "Строгость выдачи",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(settings.get("auto_strictness", cfg.auto_strictness) or cfg.auto_strictness),
                    step=0.05,
                    help="Выше = меньше результатов (жёстче порог по score). Ниже = больше результатов.",
                    key="auto_strictness",
                )
                max_results = st.slider(
                    "Максимум результатов",
                    min_value=3,
                    max_value=20,
                    value=int(cfg.max_results),
                    step=1,
                    help="Верхняя граница длины выдачи (в любом режиме — общий лимит).",
                    key="max_results",
                )
                sim_floor = st.slider(
                    "Порог схожести (есть совпадение)",
                    min_value=0.0,
                    max_value=0.8,
                    value=float(cfg.sim_floor),
                    step=0.05,
                    help="Документы с похожестью ниже порога не показываются (защита от мусора).",
                    key="sim_floor",
                )
                sim_floor_pure = st.slider(
                    "Порог схожести (только смысл)",
                    min_value=0.0,
                    max_value=0.9,
                    value=float(cfg.sim_floor_pure),
                    step=0.05,
                    help=(
                        "Порог для документов без буквальных совпадений: "
                        "чистая семантика требует более высокой схожести."
                    ),
                    key="sim_floor_pure",
                )

                st.markdown("**Веса ранжирования**")
                keyword_bonus = st.slider(
                    "Вес точного совпадения",
                    min_value=0.0,
                    max_value=1.5,
                    value=float(cfg.keyword_bonus),
                    step=0.05,
                    help=(
                        "Вклад буквальных совпадений (BM25) в score. "
                        "score = sim + вес × нормированный BM25. 0 = только смысл."
                    ),
                    key="keyword_bonus",
                )
                phrase_bonus = st.slider(
                    "Вес словосочетания",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg.phrase_bonus),
                    step=0.05,
                    help=(
                        "Бонус, если слова запроса стоят рядом как словосочетание "
                        "(допускается одно вставное слово). 0 = не учитывать."
                    ),
                    key="phrase_bonus",
                )
                misuse_penalty = st.slider(
                    "Вес штрафа misuse",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg.misuse_penalty),
                    step=0.05,
                    help=(
                        "Понижение приоритета, если слова запроса совпали с блоком "
                        "«недопустимое использование». Множитель: 0.5 = −50% score."
                    ),
                    key="misuse_penalty",
                )
                weak_match_penalty = st.slider(
                    "Вес штрафа за слабое совпадение",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg.weak_match_penalty),
                    step=0.05,
                    help="Понижение приоритета, если слова запроса нашлись только в имени файла/якоре.",
                    key="weak_match_penalty",
                )
                enable_misuse_penalty = st.checkbox(
                    "Включить misuse-штраф",
                    value=bool(cfg.enable_misuse_penalty),
                    key="enable_misuse_penalty",
                )
                enable_weak_match_penalty = st.checkbox(
                    "Включить штраф за слабое совпадение",
                    value=bool(cfg.enable_weak_match_penalty),
                    key="enable_weak_match_penalty",
                )
                _render_block_actions(
                    label="Поиск",
                    dirty=_block_is_dirty("search", tuple(SEARCH_SETTINGS_TYPES)),
                    save_key="save_search_settings_btn",
                    reset_key="reset_search_settings_btn",
                    on_save=_save_search_action,
                    on_reset=_reset_search_action,
                    save_args=(cfg,),
                    reset_args=(cfg,),
                )
                st.button(
                    "Сбросить к заводским настройкам",
                    use_container_width=True,
                    key="reset_search_defaults_btn",
                    help=(
                        "Вернуть пороги, веса и лимиты к значениям из кода. "
                        "Действует сразу в текущей сессии; чтобы записать в конфиг, "
                        "нажмите «Сохранить»."
                    ),
                    on_click=_reset_search_defaults_action,
                    args=(cfg,),
                )

        with st.expander("LLM", expanded=False):
            corp_model_kwargs_json = st.session_state.get("corp_model_kwargs_json", "{}") or "{}"
            corp_poll_timeout_s = int(st.session_state.get("corp_poll_timeout_s", cfg.corp_poll_timeout_s))
            corp_poll_interval_s = float(st.session_state.get("corp_poll_interval_s", cfg.corp_poll_interval_s))
            corp_gen = _corp_generation_defaults(cfg, "corp")
            corp_aux_gen = _corp_generation_defaults(cfg, "corp_aux")
            gigachat_credentials = st.session_state.get("gigachat_credentials", cfg.gigachat_credentials)
            gigachat_scope = st.session_state.get("gigachat_scope", cfg.gigachat_scope)
            gigachat_model = st.session_state.get("gigachat_model", cfg.gigachat_model)
            gigachat_base_url = st.session_state.get("gigachat_base_url", cfg.gigachat_base_url)
            gigachat_ca_bundle_file = st.session_state.get(
                "gigachat_ca_bundle_file", cfg.gigachat_ca_bundle_file
            )
            gigachat_verify_ssl_certs = bool(
                st.session_state.get("gigachat_verify_ssl_certs", cfg.gigachat_verify_ssl_certs)
            )
            gigachat_timeout_s = float(st.session_state.get("gigachat_timeout_s", cfg.gigachat_timeout_s))
            gigachat_temperature = float(
                st.session_state.get("gigachat_temperature", cfg.gigachat_temperature)
            )
            gigachat_max_tokens = int(st.session_state.get("gigachat_max_tokens", cfg.gigachat_max_tokens))
            gigachat_context_window = int(
                st.session_state.get("gigachat_context_window", cfg.gigachat_context_window)
            )
            openai_base_url = st.session_state.get(
                "openai_base_url", getattr(cfg, "openai_base_url", "") or ""
            )
            openai_api_key = st.session_state.get(
                "openai_api_key", getattr(cfg, "openai_api_key", "") or ""
            )
            openai_model = st.session_state.get("openai_model", getattr(cfg, "openai_model", "") or "")

            # index= (а не только key) — виджет впервые появляется вместе с Dev-блоком.
            llm_mode = st.radio(
                "Режим",
                options=["api", "ollama"],
                index=0 if st.session_state.get("llm_mode") != "ollama" else 1,
                horizontal=True,
                format_func=lambda x: "API" if x == "api" else "Ollama",
                key="llm_mode",
            )
            st.radio(
                "Провайдер по умолчанию",
                options=list(CHAT_PROVIDER_OPTIONS),
                format_func=lambda kind: {"gigachat": "ГигаЧат", "corp_async": "МЭШ.ИИ.Хаб"}[kind],
                horizontal=True,
                key="chat_provider",
                help=(
                    "С каким провайдером запускается новая сессия (config/llm.json: "
                    "chat_provider). Радиокнопка в чате переключает провайдера только "
                    "в текущей сессии и это значение не меняет."
                ),
            )
            llm_aux_model_enabled = st.checkbox(
                "Отдельная модель для перевода и расширения",
                value=bool(st.session_state.get("llm_aux_model_enabled", False)),
                key="llm_aux_model_enabled",
                help=(
                    "Перевод запроса для CKM и предложение расширений могут идти "
                    "через более простую модель. Провайдер у обеих моделей общий."
                ),
            )
            ollama_aux_model = st.session_state.get("ollama_aux_model", "")
            api_aux_model = st.session_state.get("api_aux_model", "")
            openai_aux_model = st.session_state.get("openai_aux_model", "")
            corp_aux_provider = st.session_state.get("corp_aux_provider", "") or ""
            corp_aux_model_kwargs_json = (
                st.session_state.get("corp_aux_model_kwargs_json", "{}") or "{}"
            )
            gigachat_aux_model = st.session_state.get("gigachat_aux_model", "")

            if llm_mode == "ollama":
                ollama_model = st.text_input(
                    "Ollama model",
                    value=st.session_state.get("ollama_model", ""),
                    key="ollama_model",
                    autocomplete="ollama_model",
                )
                if llm_aux_model_enabled:
                    ollama_aux_model = st.text_input(
                        "Ollama model (перевод/расширение)",
                        value=st.session_state.get("ollama_aux_model", ""),
                        key="ollama_aux_model",
                        autocomplete="ollama_aux_model",
                        help="Пусто = использовать основную модель.",
                    )
                ollama_base_url = st.text_input(
                    "Ollama base url",
                    value=st.session_state.get("ollama_base_url", ""),
                    key="ollama_base_url",
                    help="Пусто = локально. Удалённо: http://<IP>:11434",
                    autocomplete="ollama_base_url",
                )
                st.markdown("**Параметры генерации (Ollama, опционально)**")
                st.checkbox(
                    "Задать temperature",
                    value=bool(st.session_state.get("ollama_use_temperature", cfg.ollama_use_temperature)),
                    key="ollama_use_temperature",
                )
                st.slider(
                    "temperature",
                    min_value=0.0,
                    max_value=1.5,
                    value=float(st.session_state.get("ollama_temperature", cfg.ollama_temperature)),
                    step=0.05,
                    disabled=not bool(st.session_state.get("ollama_use_temperature", cfg.ollama_use_temperature)),
                    key="ollama_temperature",
                )
                st.checkbox(
                    "Задать top_p",
                    value=bool(st.session_state.get("ollama_use_top_p", cfg.ollama_use_top_p)),
                    key="ollama_use_top_p",
                )
                st.slider(
                    "top_p",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("ollama_top_p", cfg.ollama_top_p)),
                    step=0.05,
                    disabled=not bool(st.session_state.get("ollama_use_top_p", cfg.ollama_use_top_p)),
                    key="ollama_top_p",
                )
                st.checkbox(
                    "Задать repeat_penalty",
                    value=bool(st.session_state.get("ollama_use_repeat_penalty", cfg.ollama_use_repeat_penalty)),
                    key="ollama_use_repeat_penalty",
                )
                st.slider(
                    "repeat_penalty",
                    min_value=0.8,
                    max_value=2.0,
                    value=float(st.session_state.get("ollama_repeat_penalty", cfg.ollama_repeat_penalty)),
                    step=0.05,
                    disabled=not bool(
                        st.session_state.get("ollama_use_repeat_penalty", cfg.ollama_use_repeat_penalty)
                    ),
                    key="ollama_repeat_penalty",
                )
                st.checkbox(
                    "Ограничить длину (num_predict)",
                    value=bool(st.session_state.get("ollama_use_num_predict", cfg.ollama_use_num_predict)),
                    key="ollama_use_num_predict",
                )
                st.slider(
                    "num_predict",
                    min_value=64,
                    max_value=4096,
                    value=int(st.session_state.get("ollama_num_predict", cfg.ollama_num_predict)),
                    step=64,
                    disabled=not bool(st.session_state.get("ollama_use_num_predict", cfg.ollama_use_num_predict)),
                    key="ollama_num_predict",
                )
                api_kind = selected_chat_provider(cfg)
                corp_model_kwargs_json = st.session_state.get("corp_model_kwargs_json", "{}") or "{}"
                api_base_url = st.session_state.get("api_base_url", "")
                api_key = st.session_state.get("api_key", "")
                api_model = st.session_state.get("api_model", "")
                corp_provider = st.session_state.get("corp_provider", "") or ""
            else:
                # API mode: настройки обоих провайдеров редактируются всегда,
                # активного выбирает радиокнопка в чате.
                st.markdown("**МЭШ.ИИ.Хаб**")
                corp_fields = _render_api_base_key_model(
                    cfg, catalog, provider_options, kind="corp_async"
                )
                corp_model_kwargs_json = corp_fields.get(
                    "corp_model_kwargs_json", corp_model_kwargs_json
                )
                corp_gen.update(
                    {key: corp_fields[key] for key in corp_gen if key in corp_fields}
                )
                corp_aux_gen.update(
                    {key: corp_fields[key] for key in corp_aux_gen if key in corp_fields}
                )
                cpt1, cpt2 = st.columns([1, 1])
                with cpt1:
                    corp_poll_timeout_s = st.number_input(
                        "poll timeout (s)",
                        min_value=10,
                        max_value=600,
                        value=int(
                            st.session_state.get("corp_poll_timeout_s", cfg.corp_poll_timeout_s)
                        ),
                        key="corp_poll_timeout_s",
                    )
                with cpt2:
                    corp_poll_interval_s = st.number_input(
                        "poll interval (s)",
                        min_value=0.2,
                        max_value=10.0,
                        value=float(
                            st.session_state.get("corp_poll_interval_s", cfg.corp_poll_interval_s)
                        ),
                        key="corp_poll_interval_s",
                    )

                st.markdown("**ГигаЧат**")
                gc = _render_gigachat_settings(cfg)
                gigachat_credentials = gc["gigachat_credentials"]
                gigachat_scope = gc["gigachat_scope"]
                gigachat_model = gc["gigachat_model"]
                gigachat_base_url = gc["gigachat_base_url"]
                gigachat_ca_bundle_file = gc["gigachat_ca_bundle_file"]
                gigachat_verify_ssl_certs = gc["gigachat_verify_ssl_certs"]
                gigachat_timeout_s = gc["gigachat_timeout_s"]
                gigachat_temperature = gc["gigachat_temperature"]
                gigachat_max_tokens = gc["gigachat_max_tokens"]
                gigachat_context_window = gc["gigachat_context_window"]
                gigachat_aux_model = gc["gigachat_aux_model"]

                with st.expander("OpenAI-совместимый API (только dev)", expanded=False):
                    openai_fields = _render_api_base_key_model(
                        cfg, catalog, provider_options, kind="openai_compatible"
                    )

                api_fields = {**corp_fields, **openai_fields}
                api_base_url = api_fields["api_base_url"]
                api_key = api_fields["api_key"]
                api_model = api_fields["api_model"]
                api_aux_model = api_fields.get("api_aux_model", "")
                corp_provider = api_fields["corp_provider"]
                corp_aux_provider = api_fields.get("corp_aux_provider", "") or ""
                corp_aux_model_kwargs_json = api_fields.get(
                    "corp_aux_model_kwargs_json", "{}"
                )
                openai_base_url = api_fields["openai_base_url"]
                openai_api_key = api_fields["openai_api_key"]
                openai_model = api_fields["openai_model"]
                openai_aux_model = api_fields.get("openai_aux_model", "")
                api_kind = selected_chat_provider(cfg)

                ollama_model = st.session_state.get("ollama_model", "")
                ollama_base_url = st.session_state.get("ollama_base_url", "")

            st.markdown("**Контекст для ИИ**")
            st.number_input(
                "Бюджет контекста: архетипы (токены)",
                min_value=1000,
                max_value=200000,
                step=1000,
                value=int(
                    st.session_state.get(
                        "context_tokens_archetypes",
                        getattr(cfg, "context_tokens_archetypes", 10000),
                    )
                ),
                key="context_tokens_archetypes",
                help=(
                    "Сколько токенов отводить на ADL-контекст во вкладке «Архетипы». "
                    "10 000 ≈ 30 000 символов. Больше — больше архетипов, "
                    "но выше риск шума и дольше ответ."
                ),
            )
            st.number_input(
                "Бюджет контекста: поля (токены)",
                min_value=1000,
                max_value=200000,
                step=1000,
                value=int(
                    st.session_state.get(
                        "context_tokens_fields",
                        getattr(cfg, "context_tokens_fields", 10000),
                    )
                ),
                key="context_tokens_fields",
                help=(
                    "Сколько токенов отводить на ADL-контекст во вкладке «Поля». "
                    "10 000 ≈ 30 000 символов. Поля короткие, поэтому их "
                    "обычно помещается много."
                ),
            )
            _render_block_actions(
                label="LLM",
                dirty=_block_is_dirty(
                    "llm", (*LLM_SETTINGS_TYPES, *_LLM_UI_KEYS)
                ),
                save_key="save_llm_settings_btn",
                reset_key="reset_llm_settings_btn",
                on_save=_save_llm_block,
                on_reset=_reset_llm_action,
                save_args=(cfg,),
                reset_args=(cfg,),
            )

        with st.expander("Промпты", expanded=False):
            st.caption(
                f"Все промпты хранятся в `{prompts_path(cfg.config_dir)}`. "
                "Изменения применяются сразу, а в файл попадают по кнопке «Сохранить». "
                "Пустые обязательные поля не сохраняются."
            )
            with st.container(key="prompt_fields"):
                st.text_area(
                    "Основной системный промпт",
                    value=st.session_state.get("system_prompt", ""),
                    height=prompt_field_height(st.session_state.get("system_prompt", "")),
                    key="system_prompt",
                    help=(
                        "Базовые правила ответа, включая правила для вопросов вне темы. "
                        "Можно использовать {query} — подставится поисковый запрос."
                    ),
                )
                for part_key in PROMPT_PART_KEYS:
                    part_value = st.session_state.get(f"prompt_{part_key}", "")
                    st.text_area(
                        PROMPT_LABELS[part_key],
                        value=part_value,
                        height=prompt_field_height(part_value),
                        key=f"prompt_{part_key}",
                    )
            # Точная подгонка высоты под содержимое: ширина сайдбара меняется,
            # поэтому статическая оценка в prompt_field_height может промахнуться.
            st.session_state["_prompt_fit_seq"] = (
                int(st.session_state.get("_prompt_fit_seq") or 0) + 1
            )
            st.iframe(
                _PROMPT_FIELDS_JS.replace("__MAX__", str(_PROMPT_FIELD_MAX_HEIGHT))
                + f"\n<!-- prompts-{st.session_state['_prompt_fit_seq']} -->",
                height=1,
            )
            _render_block_actions(
                label="Промпты",
                dirty=_block_is_dirty("prompts", _PROMPT_SESSION_KEYS),
                save_key="save_prompts_btn",
                reset_key="reset_prompts_btn",
                on_save=_save_prompts_block,
                on_reset=_reset_prompts_action,
                save_args=(cfg,),
                reset_args=(cfg,),
            )

        # Снимок значений для индикатора «есть несохранённые изменения».
    # Делается после отрисовки виджетов: иначе на первом прогоне сессии
    # ключей ещё нет и индикатор ложно срабатывал бы.
    baseline = st.session_state.setdefault("_dev_baseline", {})
    baseline.setdefault("search", _capture_baseline(tuple(SEARCH_SETTINGS_TYPES)))
    baseline.setdefault("index", _capture_baseline(_BLOCK_INDEX_KEYS))
    baseline.setdefault(
        "llm", _capture_baseline((*LLM_SETTINGS_TYPES, *_LLM_UI_KEYS))
    )
    baseline.setdefault("prompts", _capture_baseline(_PROMPT_SESSION_KEYS))

    state = SidebarState(
        adl_dir=adl_dir,
        index_dir=index_dir,
        ckm_adl_dir=(ckm_adl_dir or "").strip(),
        auto_strictness=float(auto_strictness),
        keyword_bonus=float(keyword_bonus),
        phrase_bonus=float(phrase_bonus),
        misuse_penalty=float(misuse_penalty),
        weak_match_penalty=float(weak_match_penalty),
        enable_misuse_penalty=bool(enable_misuse_penalty),
        enable_weak_match_penalty=bool(enable_weak_match_penalty),
        max_results=int(max_results),
        sim_floor=float(sim_floor),
        sim_floor_pure=float(sim_floor_pure),
        llm_mode=llm_mode,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        api_kind=api_kind,
        chat_provider=str(st.session_state.get("chat_provider") or cfg.chat_provider or ""),
        api_base_url=api_base_url,
        api_key=api_key,
        api_model=api_model,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        corp_provider=corp_provider,
        corp_model_kwargs_json=corp_model_kwargs_json,
        corp_poll_timeout_s=int(corp_poll_timeout_s),
        corp_poll_interval_s=float(corp_poll_interval_s),
        **corp_gen,
        **corp_aux_gen,
        gigachat_credentials=gigachat_credentials,
        gigachat_scope=gigachat_scope,
        gigachat_model=gigachat_model,
        gigachat_base_url=gigachat_base_url,
        gigachat_ca_bundle_file=gigachat_ca_bundle_file,
        gigachat_verify_ssl_certs=bool(gigachat_verify_ssl_certs),
        gigachat_timeout_s=float(gigachat_timeout_s),
        gigachat_temperature=float(gigachat_temperature),
        gigachat_max_tokens=int(gigachat_max_tokens),
        gigachat_context_window=int(gigachat_context_window),
        llm_aux_model_enabled=bool(llm_aux_model_enabled),
        ollama_aux_model=ollama_aux_model,
        api_aux_model=api_aux_model,
        openai_aux_model=openai_aux_model,
        corp_aux_provider=corp_aux_provider,
        corp_aux_model_kwargs_json=corp_aux_model_kwargs_json,
        gigachat_aux_model=gigachat_aux_model,
    )

    # Настройки сохранены в конфиг? Нет: изменения живут в session_state
    # как черновик и пишутся в файлы только по кнопке «Сохранить».

    # Полная пересборка: диалог подтверждения (сама пересборка — в блоке «Индексация»).
    if st.session_state.get("full_rebuild_open"):
        st.session_state["full_rebuild_open"] = False
        _full_rebuild_dialog()

    return state
