"""Shared state for the archetype/field tabs."""

from dataclasses import dataclass
from typing import Any

import streamlit as st

from adl_smart_search.prompts import PROMPT_PART_KEYS


@dataclass
class TabContext:
    dev_mode: bool
    cfg: Any
    adl_dir: str
    index_dir: str
    preferred_lang: str
    sources: list
    source_labels: dict
    docs: list | None
    vectors: Any
    field_docs: list | None
    field_vectors: Any
    embedder: Any
    llm_provider: Any
    llm_aux_provider: Any
    llm_setup_msg: str
    auto_strictness: float
    keyword_bonus: float
    misuse_penalty: float
    weak_match_penalty: float
    enable_misuse_penalty: bool
    enable_weak_match_penalty: bool
    max_results: int
    sim_floor: float
    sim_floor_pure: float
    sim_floor_pure_cross: float
    sim_low_confidence: float


def init_session_defaults(prompts: dict | None = None) -> None:
    """Общие сессионные ключи приложения (LLM-настройки — в llm_provider.init_llm_session_defaults)."""
    prompts = prompts or {}
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if "last_top_indices" not in st.session_state:
        st.session_state["last_top_indices"] = []
    if "last_match_debug" not in st.session_state:
        st.session_state["last_match_debug"] = {}
    if "last_selector_raw" not in st.session_state:
        st.session_state["last_selector_raw"] = ""
    if "last_selections" not in st.session_state:
        st.session_state["last_selections"] = []
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = ""
    if "last_error_details" not in st.session_state:
        st.session_state["last_error_details"] = ""
    if not str(st.session_state.get("system_prompt") or "").strip():
        st.session_state["system_prompt"] = str(prompts.get("system_prompt") or "")
    for part_key in PROMPT_PART_KEYS:
        state_key = f"prompt_{part_key}"
        # Пустое значение заполняем из prompts.json (например, если файл
        # появился позже). Непустые правки из dev-панели не трогаем.
        if not str(st.session_state.get(state_key) or "").strip():
            st.session_state[state_key] = str(prompts.get(part_key) or "")
    if "arch_dialog_messages" not in st.session_state:
        st.session_state["arch_dialog_messages"] = []
    if "field_dialog_messages" not in st.session_state:
        st.session_state["field_dialog_messages"] = []
    if "arch_chat_messages" not in st.session_state:
        st.session_state["arch_chat_messages"] = []
    if "field_chat_messages" not in st.session_state:
        st.session_state["field_chat_messages"] = []
    if "arch_rag_context" not in st.session_state:
        st.session_state["arch_rag_context"] = ""
    if "arch_rag_query" not in st.session_state:
        st.session_state["arch_rag_query"] = ""
    if "arch_rag_selection_key" not in st.session_state:
        st.session_state["arch_rag_selection_key"] = ""
    if "field_rag_context" not in st.session_state:
        st.session_state["field_rag_context"] = ""
    if "field_rag_query" not in st.session_state:
        st.session_state["field_rag_query"] = ""
    if "field_rag_selection_key" not in st.session_state:
        st.session_state["field_rag_selection_key"] = ""
    if "arch_query" not in st.session_state:
        # После смены вкладки состояние виджета очищается — возвращаем
        # последний введённый запрос, иначе поле выглядит пустым.
        st.session_state["arch_query"] = st.session_state.get("arch_query_saved", "")
    if "field_query" not in st.session_state:
        st.session_state["field_query"] = st.session_state.get("field_query_saved", "")


def prompts_from_session() -> dict[str, str]:
    """Текущие промпты из session_state (учитывает правки dev-панели)."""
    out = {"system_prompt": str(st.session_state.get("system_prompt") or "")}
    for part_key in PROMPT_PART_KEYS:
        out[part_key] = str(st.session_state.get(f"prompt_{part_key}") or "")
    return out
