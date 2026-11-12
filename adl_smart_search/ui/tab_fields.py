"""Раздел «Поиск полей» — тонкая обёртка над общим движком поиска."""

import os
import re

import streamlit as st

from adl_smart_search.context_builder import context_tokens_to_max_chars
from adl_smart_search.rag import build_compact_context_for_fields
from adl_smart_search.ui.context import TabContext
from adl_smart_search.ui.search_tab import (
    SearchTabSpec,
    highlight_matches_html,
    render_search_tab,
    render_source_badge,
)

FIELD_STALE_WARNING = (
    "Контекст диалога устарел: изменился выбор полей. "
    "Следующее сообщение будет отправлено с новым контекстом."
)

# Жёсткий предохранитель. Реальный объём задаёт бюджет токенов в dev-панели
# (context_tokens_fields), поэтому здесь достаточно большого потолка.
MAX_CONTEXT_FIELDS = 100

DV_TYPE_OPTIONS = [
    "DV_BOOLEAN",
    "DV_CODED_TEXT",
    "DV_COUNT",
    "DV_DATE",
    "DV_DATE_TIME",
    "DV_DURATION",
    "DV_IDENTIFIER",
    "DV_MULTIMEDIA",
    "DV_ORDINAL",
    "DV_PARSABLE",
    "DV_PROPORTION",
    "DV_QUANTITY",
    "DV_TEXT",
    "DV_TIME",
    "DV_URI",
    "DV_EHR_URI",
    "DV_STATE",
    "DV_INTERVAL",
]


def _value_types_from_field_doc(d: dict) -> set[str]:
    vt = d.get("value_types")
    if isinstance(vt, list) and vt:
        return {str(x) for x in vt if str(x).startswith("DV_")}
    txt = (d.get("text") or "")
    return set(re.findall(r"(DV_[A-Z_]+)", txt))


def _extract_values_and_units(d: dict) -> tuple[list[str], list[str]]:
    """
    Достаёт:
    - значения: из строки "Значения: ..." в d["text"]
    - единицы: из фрагментов "Число (<units>)"
    """
    txt = (d.get("text") or "")
    values: list[str] = []
    units: list[str] = []
    if txt:
        m = re.search(r"(?im)^\s*Значения:\s*(.+?)\s*$", txt)
        if m:
            raw = m.group(1).strip()
            if raw:
                parts = [p.strip() for p in raw.split(";") if p.strip()]
                values = list(dict.fromkeys(parts))[:20]
                for u in re.findall(r"Число\s*\(([^)]+)\)", raw):
                    uu = u.strip()
                    if uu:
                        units.append(uu)
        if not units:
            for u in re.findall(r"Число\s*\(([^)]+)\)", txt):
                uu = u.strip()
                if uu:
                    units.append(uu)
    units = list(dict.fromkeys(units))[:12]
    return values, units


def _render_name(doc: dict, md: dict, dev_mode: bool, fn: str, source: str = "local") -> None:
    path = doc.get("path") or doc.get("field", "")
    at_code = doc.get("at_code", "")
    render_source_badge(source)
    st.code(fn, language=None)
    field_name = (doc.get("field") or "").strip()
    if not field_name:
        # fallback: последний сегмент path (если path задан)
        field_name = (str(path).split("/")[-1] if path else "").strip()
    vt = doc.get("value_types")
    vt_list = [str(x) for x in vt] if isinstance(vt, list) else []
    dv_types = [x for x in vt_list if x.startswith("DV_")]
    rm_type = doc.get("rm_type") or ""
    type_s = ", ".join(dv_types) if dv_types else (rm_type or "—")
    values, units = _extract_values_and_units(doc)
    extras = []
    if units:
        extras.append("ед.: " + ", ".join(units))
    if values:
        extras.append("значения: " + "; ".join(values[:8]))
    extra_s = (" | " + " | ".join(extras)) if extras else ""
    field_line = f"{field_name or '—'} ({at_code}) [{type_s}]{extra_s}"
    st.code(field_line, language=None)
    st.code(str(path), language=None)


def _render_details(doc: dict, md: dict, fn: str) -> None:
    with st.expander("Показать описание поля", expanded=False):
        st.html(
            highlight_matches_html(
                (doc.get("text") or "")[:8000],
                (md or {}).get("matched_unique") or [],
            )
        )


def _field_label(doc: dict) -> str:
    """Короткая подпись поля (для подсказок «ничего не найдено»)."""
    field_name = (doc.get("field") or "").strip()
    if not field_name:
        path = str(doc.get("path") or "")
        field_name = (path.split("/")[-1] if path else "").strip()
    at_code = doc.get("at_code") or ""
    fn = os.path.basename(str(doc.get("file") or ""))
    if fn.lower().endswith(".adl"):
        fn = fn[:-4]
    head = field_name or "(поле без имени)"
    if at_code:
        head = f"{head} ({at_code})"
    return f"{head} — {fn}" if fn else head


def _build_field_context(ctx: TabContext, query: str, selected: list) -> str:
    archetype_docs_by_file = {str(d.get("file") or ""): d for d in (ctx.docs or [])}
    return build_compact_context_for_fields(
        query=query,
        selected_fields=selected,
        max_items=MAX_CONTEXT_FIELDS,
        max_chars=context_tokens_to_max_chars(
            st.session_state.get("context_tokens_fields", 10000)
        ),
        include_containers=False,
        archetype_docs_by_file=archetype_docs_by_file,
    )


def _make_field_spec(ctx: TabContext) -> SearchTabSpec:
    return SearchTabSpec(
        prefix="field",
        top_caption="Поиск полей в архетипах",
        query_help="Одно или несколько слов",
        filter_label="Фильтр: тип данных поля (DV_*)",
        filter_help=(
        "Можно выбрать несколько. Поле попадает в результаты, если у него есть "
        "хотя бы один из выбранных типов."
    ),
        filter_options=DV_TYPE_OPTIONS,
        filter_key="field_dv_types_filter",
        filter_matches=lambda selected, doc: bool(_value_types_from_field_doc(doc) & set(selected)),
        vocab_key="field_token_vocab",
        min_k=5,
        result_count_label="Показано найденных полей:",
        pick_all_label="Выбрать все поля",
        make_pick_key=lambda doc, idx: f"field_pick_{idx}_{doc.get('file', '')}_{doc.get('at_code', '')}",
        render_result_name=_render_name,
        render_result_details=_render_details,
        result_label=_field_label,
        default_question_key="default_question_fields",
        build_context=lambda query, selected, embedder=None, query_en="", expanded_query=None: _build_field_context(
            ctx, query, selected
        ),
    select_first_warning="Сначала выберите хотя бы одно поле в результатах поиска",
    stale_warning=FIELD_STALE_WARNING,
    # Тексты полей короткие (заголовки), поэтому кросс-языковая схожесть у них
    # завышена: порог выше, чем для длинных текстов архетипов, иначе лезет мусор.
    sim_floor_pure_cross=0.50,
    context_tokens_key="context_tokens_fields",
    index_kind="field",
)


def render_fields_tab(ctx: TabContext) -> None:
    render_search_tab(ctx, _make_field_spec(ctx), ctx.field_docs, ctx.field_vectors)