"""Раздел «Поиск архетипов» — тонкая обёртка над общим движком поиска."""

import os
import re

import streamlit as st

from adl_smart_search.context_builder import context_tokens_to_max_chars
from adl_smart_search.rag import build_compact_context_for_archetypes
from adl_smart_search.ui.context import TabContext
from adl_smart_search.ui.search_tab import (
    SearchTabSpec,
    highlight_matches_html,
    render_search_tab,
    render_source_badge,
)

ARCH_STALE_WARNING = (
    "Контекст диалога устарел: изменился выбор архетипов. "
    "Следующее сообщение будет отправлено с новым контекстом."
)

# Жёсткий предохранитель. Реальный объём задаёт бюджет токенов в dev-панели
# (context_tokens_archetypes), поэтому здесь достаточно большого потолка.
MAX_CONTEXT_ARCHETYPES = 20
# Нарезка внутри архетипа: меньше строк и блоков — меньше токенов на каждый запрос.
ARCH_CONTEXT_MAX_LINES = 80
ARCH_CONTEXT_VECTOR_TOP_K = 8


def _archetype_class_from_file(fn: str) -> str:
    m = re.match(r"(?i)openEHR-EHR-([A-Z_]+)\.", fn or "")
    return (m.group(1).upper() if m else "")


def _render_name(doc: dict, md: dict, dev_mode: bool, fn: str, source: str = "local") -> None:
    render_source_badge(source)
    st.code(fn, language=None)


def _render_details(doc: dict, md: dict, fn: str) -> None:
    with st.expander("Показать метаданные и структуру архетипа", expanded=False):
        st.html(
            highlight_matches_html(
                doc.get("full_context", "")[:12000],
                (md or {}).get("matched_unique") or [],
            )
        )


def _archetype_label(doc: dict) -> str:
    """Короткая подпись архетипа (для подсказок «ничего не найдено»)."""
    fn = os.path.basename(str(doc.get("file") or ""))
    if fn.lower().endswith(".adl"):
        fn = fn[:-4]
    return fn or "(без имени)"


ARCH_SPEC = SearchTabSpec(
    prefix="arch",
    top_caption="Поиск архетипов",
    query_help="Одно или несколько слов",
    filter_label="Фильтр: класс архетипа",
    filter_help="Можно выбрать несколько",
    filter_options=[
        "ACTION",
        "CLUSTER",
        "COMPOSITION",
        "EVALUATION",
        "INSTRUCTION",
        "OBSERVATION",
        "SECTION",
        "ADMIN_ENTRY",
    ],
    filter_key="arch_class_filter",
    filter_matches=lambda selected, doc: _archetype_class_from_file(doc.get("file", "")) in set(selected),
    vocab_key="arch_token_vocab",
    min_k=3,
    result_count_label="Показано найденных архетипов:",
    pick_all_label="Выбрать все архетипы",
    make_pick_key=lambda doc, idx: f"arch_pick_{doc.get('file', '')}",
    render_result_name=_render_name,
    render_result_details=_render_details,
    result_label=_archetype_label,
    default_question_key="default_question_archetypes",
    build_context=lambda query, selected_docs, *, embedder=None, query_en="", expanded_query=None: (
        build_compact_context_for_archetypes(
            query=query,
            selected_docs=selected_docs,
            include_metadata=True,
            max_fields_lines=ARCH_CONTEXT_MAX_LINES,
            max_chars=context_tokens_to_max_chars(
                st.session_state.get("context_tokens_archetypes", 10000)
            ),
            embedder=embedder,
            query_en=query_en,
            expanded_query=expanded_query,
            vector_top_k=ARCH_CONTEXT_VECTOR_TOP_K,
            max_documents=MAX_CONTEXT_ARCHETYPES,
        )
    ),
    select_first_warning="Сначала выберите хотя бы один архетип в результатах поиска",
    stale_warning=ARCH_STALE_WARNING,
    context_tokens_key="context_tokens_archetypes",
    index_kind="archetype",
)


def render_archetype_tab(ctx: TabContext) -> None:
    render_search_tab(ctx, ARCH_SPEC, ctx.docs, ctx.vectors)