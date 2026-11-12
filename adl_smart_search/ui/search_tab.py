"""Общий движок раздела поиска (используется «Поиск по архетипам» и «Поиск по полям»).

Раньше оба раздела почти полностью дублировали друг друга (~400–500 строк каждый).
Вся общая логика — форма поиска, расширение запроса ИИ, фильтр, индексация,
подсчёт и авто-выдача, рендер результатов с выбором, вопрос к ИИ, диалог
уточнений — вынесена сюда и переиспользуется через SearchTabSpec.
"""

import html
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import streamlit as st

from adl_smart_search.index_store import ensure_embedding_dim_compatible
from adl_smart_search.llm import LLMError, chat_with_continuation
from adl_smart_search.orchestrator import make_llm_dialog
from adl_smart_search.prompts import (
    apply_placeholders,
    compose_system_prompt,
    format_default_question,
    prompt_text,
)
from adl_smart_search.query_expand import (
    apply_manual_expansions,
    looks_latin,
    suggest_query_expansions,
    translate_query_to_en_mixed,
)
from adl_smart_search.rag import (
    _stem_prefix,
    compute_scores_multilingual,
    is_low_confidence,
    make_rag_selection_key,
    pure_floor_by_doc_lang,
    reorder_picked_with_simi_preference,
)
from adl_smart_search.ui.context import TabContext, prompts_from_session
from adl_smart_search.ui.dialog import (
    append_chat_message,
    clear_chat_session,
    get_chat_messages,
    rag_selection_stale,
    render_chat_messages,
    render_pending_chat_message,
    render_usage_caption,
    report_llm_error,
    send_followup,
    store_last_usage,
)
from adl_smart_search.ui.llm_provider import (
    CHAT_PROVIDER_OPTIONS,
    selected_chat_provider,
)
from adl_smart_search.ui.settings import select_indices_by_relevance
from adl_smart_search.ui.utils import filter_docs, short_token_hint

logger = logging.getLogger(__name__)

# «Показать ещё»: шаг выдачи и общий потолок (включая первичный отбор).
_SHOW_MORE_STEP = 5
_SHOW_MORE_LIMIT = 50

# Сколько вариантов расширения запроса показываем и берём из ответа ИИ.
_EXPAND_TERMS_LIMIT = 5

# Автоскролл истории чата: при первом показе и при добавлении сообщений.
# iframe имеет height=1, поэтому у внутреннего документа обязательно убираем
# отступы и прокрутку — иначе браузер рисует в нём полосу прокрутки,
# которая выглядит как короткий «обрубок» справа от чата.
_CHAT_SCROLL_JS = """
<style>html,body{margin:0;padding:0;height:0;overflow:hidden}</style>
<script>
setTimeout(() => {
  try {
    const doc = window.parent.document;
    const box = doc.querySelector('[class*="__PREFIX___chat_history"]');
    if (!box) return;
    const count = box.querySelectorAll('[data-testid="stChatMessage"]').length;
    const prev = parseInt(box.dataset.chatMsgCount || '-1', 10);
    if (prev < 0 || count !== prev) box.scrollTop = box.scrollHeight;
    box.dataset.chatMsgCount = String(count);
  } catch (e) {}
}, 100);
</script>
"""

# Даты индексации в подписях источников. Streamlit не даёт стилизовать часть
# markdown-подписи радио, поэтому оборачиваем вторую строку (дату) в span.
_SOURCE_DATES_JS = """
<style>html,body{margin:0;padding:0;height:0;overflow:hidden}</style>
<script>
setTimeout(() => {
  try {
    const doc = window.parent.document;
    const group = doc.querySelector('[role="radiogroup"][aria-label="Источники"]');
    if (!group) return;
    const dateRe = /^\\s*(\\d{2}\\.\\d{2}\\.\\d{4})\\s*$/;
    for (const label of group.querySelectorAll('label')) {
      if (label.querySelector('.src-date')) continue;
      const p = label.querySelector('[data-testid="stMarkdownContainer"] p');
      if (!p) continue;
      for (const node of Array.from(p.childNodes)) {
        if (node.nodeType !== 3) continue;
        const m = node.textContent.match(dateRe);
        if (!m) continue;
        const span = doc.createElement('span');
        span.className = 'src-date';
        span.textContent = m[1];
        p.replaceChild(span, node);
      }
    }
  } catch (e) {}
}, 150);
</script>
"""

# Streamlit даёт контейнеру с key класс .st-key-<key> (см. src.BnXM6qiK.js, wO()).
# Сайдбар отдан чату: делаем его чуть шире и аккуратно переносим текст в кнопках.
CHAT_SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"][aria-expanded="true"] {
  min-width: 380px;
}
section[data-testid="stSidebar"] div.stButton > button {
  white-space: normal;
  line-height: 1.2;
}
section[data-testid="stSidebar"] div[data-testid="stTextInput"] {
  min-width: 0 !important;
}
.chat-panel-title {
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.65;
  margin: 0 0 10px 0;
}
/* Бабблы чата: вопросы справа, ответы ИИ слева.
   Контейнеры имеют ключи вида <prefix>_chatmsg_<n>_user, класс .st-key-..._user. */
[class*="chatmsg"][class*="_user"] [data-testid="stChatMessage"],
[class*="chatmsg"][class*="_assistant"] [data-testid="stChatMessage"] {
  background: transparent;
  padding: 0;
}
[class*="chatmsg"][class*="_user"] [data-testid="stChatMessage"] {
  flex-direction: row-reverse;
}
[class*="chatmsg"][class*="_user"] [data-testid="stChatMessageContent"],
[class*="chatmsg"][class*="_assistant"] [data-testid="stChatMessageContent"] {
  flex: 0 1 auto !important;
  width: fit-content !important;
  max-width: 88%;
  min-width: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
  border-radius: 14px;
  padding: 8px 12px;
}
/* Streamlit задаёт content margin: auto. В flex это делит свободное место
   поровну слева и справа, поэтому короткий баббл «отъезжает» от аватара.
   Явно обнуляем лишние auto-поля: баббл всегда прижат к своей иконке. */
[class*="chatmsg"][class*="_user"] [data-testid="stChatMessageContent"] {
  background: rgba(31, 119, 180, 0.14);
  margin: 0 0 0 auto !important;
}
[class*="chatmsg"][class*="_assistant"] [data-testid="stChatMessageContent"] {
  background: rgba(128, 128, 128, 0.12);
  margin: 0 auto 0 0 !important;
}
/* Archetype id и at-коды в ответах ИИ: только моноширинный шрифт, без фона. */
[class*="chatmsg"][class*="_assistant"] [data-testid="stChatMessageContent"] code {
  background: transparent;
  border: 0;
  border-radius: 0;
  padding: 0;
  font-family: ui-monospace, SFMono-Regular, "Cascadia Mono", Consolas, "Liberation Mono", monospace;
  font-size: 0.92em;
  color: inherit;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}
/* Аватары: приглушённые оттенки вместо ярких красного/оранжевого.
   Иконка наследует цвет текста, поэтому читается и в светлой, и в тёмной теме. */
section[data-testid="stSidebar"] [data-testid="stChatMessageAvatarUser"],
section[data-testid="stSidebar"] [data-testid="stChatMessageAvatarAssistant"] {
  color: inherit !important;
  border: 1px solid transparent;
}
section[data-testid="stSidebar"] [data-testid="stChatMessageAvatarUser"] {
  background-color: rgba(31, 119, 180, 0.18) !important;
  border-color: rgba(31, 119, 180, 0.38);
}
section[data-testid="stSidebar"] [data-testid="stChatMessageAvatarAssistant"] {
  background-color: rgba(128, 128, 128, 0.16) !important;
  border-color: rgba(128, 128, 128, 0.38);
}
/* Раскладка чата: история скроллится, поле ввода всегда внизу сайдбара. */
/* Компактная шапка сайдбара: кнопка сворачивания остаётся, но над чатом
   больше нет 76 px пустоты — высвобождаем место под историю. */
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
  height: 2.5rem !important;
  min-height: 2.5rem !important;
  margin-bottom: 0.25rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  height: calc(100% - 2.75rem);
  min-height: 0;
  padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] [class*="chat_panel"] {
  height: calc(100vh - 2.75rem) !important;
  flex: 0 0 auto !important;
  min-height: 0;
}
section[data-testid="stSidebar"] [class*="chat_panel"] > div:has([class*="chat_history"]) {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  min-height: 0;
}
section[data-testid="stSidebar"] [class*="chat_panel"] > div:has([class*="chat_composer"]) {
  margin-top: auto;
}
section[data-testid="stSidebar"] [class*="chat_history"] {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding-right: 2px;
}
section[data-testid="stSidebar"] [class*="chat_composer"] {
  position: sticky;
  bottom: 0;
  flex: 0 0 auto;
  z-index: 50;
  background: var(--secondary-background-color);
  padding: 8px 0 10px 0;
  margin-top: auto;
  border-top: 1px solid rgba(128, 128, 128, 0.22);
}
/* Textarea растёт по вертикали с текстом (Chrome/Edge 123+), затем скроллится. */
section[data-testid="stSidebar"] [class*="chat_composer"] [data-testid="stTextArea"] {
  height: auto !important;
}
section[data-testid="stSidebar"] [class*="chat_composer"] textarea {
  field-sizing: content;
  min-height: 3rem;
  max-height: 12rem;
  height: auto !important;
  overflow-y: auto;
  resize: none;
}
.src-badge {
  display: inline-block;
  margin: 0 0 2px 0;
  padding: 0 8px;
  border: 1px solid;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 600;
  line-height: 1.6;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: inherit;
  opacity: 0.9;
}
.src-date {
  font-size: 0.72em;
  opacity: 0.55;
}
.src-badge-local {
  background: rgba(31, 119, 180, 0.18);
  border-color: rgba(31, 119, 180, 0.55);
}
.src-badge-ckm {
  background: rgba(230, 126, 34, 0.18);
  border-color: rgba(230, 126, 34, 0.60);
}
.match-hl {
  background: rgba(255, 196, 0, 0.35);
  color: inherit;
  border-radius: 2px;
  padding: 0 1px;
}
.result-context {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: inherit;
  font-size: 0.85rem;
  margin: 0;
}
</style>
"""


def highlight_matches_html(text: str, keywords) -> str:
    """Текст в блоке с сохранением переносов и подсветкой совпавших слов."""
    # Именно <div>, а не <pre>: Streamlit превращает <pre> в свой stMarkdownPre
    # с white-space: normal, и переносы строк схлопываются в сплошной текст.
    open_tag = (
        '<div class="result-context" '
        'style="white-space: pre-wrap; word-break: break-word;">'
    )
    body = html.escape(text or "")
    kws = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
    if not kws:
        return f"{open_tag}{body}</div>"
    parts = []
    for kw in dict.fromkeys(kws):
        stem = _stem_prefix(kw)
        parts.append(re.escape(stem) + (r"\w*" if stem != kw else r"\b"))
    pattern = re.compile(r"\b(?:" + "|".join(parts) + r")", flags=re.IGNORECASE)
    marked = pattern.sub(lambda m: f'<span class="match-hl">{m.group(0)}</span>', body)
    return f"{open_tag}{marked}</div>"


@dataclass(frozen=True)
class SearchTabSpec:
    prefix: str
    top_caption: str
    query_help: str

    filter_label: str
    filter_help: str
    filter_options: list[str]
    filter_key: str
    filter_matches: Callable[[list[str], dict], bool]

    vocab_key: str
    min_k: int

    result_count_label: str
    pick_all_label: str
    make_pick_key: Callable[[dict, int], str]
    # (doc, match_debug, dev_mode, short_name, source) — source: "local" | "ckm"
    render_result_name: Callable[[dict, dict, bool, str, str], None]
    render_result_details: Callable[[dict, dict, str], None]
    result_label: Callable[[dict], str]

    default_question_key: str
    build_context: Callable[..., str]
    select_first_warning: str
    stale_warning: str
    # Порог «только по смыслу» для документов на другом языке (None → из конфига).
    sim_floor_pure_cross: float | None = None
    # Ключ настройки бюджета токенов на ADL-контекст (для подписи в чате).
    context_tokens_key: str = ""
    # Какой индекс показывать в датах у источников: "archetype" | "field".
    index_kind: str = ""


def _index_date(index_dir: str, kind: str) -> str:
    """Дата сборки индекса: из build_info.json, иначе mtime (legacy)."""
    from adl_smart_search.index_build_info import format_index_date, index_built_at

    root = os.path.dirname(index_dir or ".")
    source = os.path.basename(index_dir or ".")
    ts = index_built_at(root, source, kind)
    return format_index_date(ts)


def _index_date_for_source(enabled_sources, source_name: str, kind: str) -> str:
    if not kind:
        return ""
    source = next((s for s in enabled_sources if s.name == source_name), None)
    return _index_date(source.index_dir, kind) if source else ""


def _source_label_with_date(label: str, date_s: str) -> str:
    """
    Подпись пункта «Источники» с датой индексации под ней (только дата, без текста).

    Даты нет только для смешанного режима: там непонятно, чья это дата.
    """
    if not date_s:
        return label
    return f"{label}  \n{date_s}"


def _run_pending_dim_check(ctx: TabContext) -> None:
    """Ленивая проверка совместимости индекса с моделью эмбеддингов (один раз на сессию).

    Модель грузим тут, а не на старте приложения: она всё равно нужна для encode
    при поиске, а сверка dim по имени модели уже сделана в app.py (index_model_matches).
    """
    pending = st.session_state.get("embed_dim_check_pending") or []
    if not pending:
        return
    try:
        embed_dim = ctx.embedder.embedding_dim()
        for index_dir, kind in pending:
            ensure_embedding_dim_compatible(index_dir, embed_dim, kind)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception:
        logger.warning("lazy embedding compat check skipped", exc_info=True)
    finally:
        st.session_state["embed_dim_check_pending"] = []


def _effective_cross_floor(ctx: TabContext, spec: SearchTabSpec) -> float:
    """Порог «только по смыслу» для другого языка: настройка вкладки или конфига."""
    if spec.sim_floor_pure_cross is not None:
        return float(spec.sim_floor_pure_cross)
    return float(ctx.sim_floor_pure_cross)


def render_source_badge(source: str) -> None:
    """Цветная метка источника (local/CKM) вместо префикса в копируемом id."""
    key = (source or "local").strip().lower()
    if key == "ckm":
        label = "CKM"
    elif key == "local":
        label = "local"
    else:
        label = key.upper()
    st.markdown(
        f'<span class="src-badge src-badge-{key}">{label}</span>',
        unsafe_allow_html=True,
    )


def _dev_penalty_reason(md: dict) -> str:
    if md.get("misuse_hit"):
        return "misuse"
    if md.get("weak_hit"):
        return "слабое совпадение"
    return "понижение"


def _dev_score_line(
    score_s, sim_s, bonus_s, penalty_s, md: dict, keyword_bonus: float, zscore: float | None = None
) -> str:
    """Расшифровка score: из чего он сложился, и z-оценка внутри источника."""
    score = float(score_s) if score_s is not None else 0.0
    sim = float(sim_s) if sim_s is not None else 0.0
    bonus = float(bonus_s) if bonus_s is not None else 0.0
    penalty = float(penalty_s) if penalty_s is not None else 1.0
    reason = _dev_penalty_reason(md)

    if not md.get("lexical_enabled", True):
        base = f"схожесть {sim:.2f}"
        if penalty < 0.999:
            base += f" × штраф {penalty:.2f} ({reason})"
        base += " · лексика отключена (другой язык)"
    else:
        base = f"схожесть {sim:.2f} + {float(keyword_bonus):.2f} × лексика {bonus:.2f}"
        if penalty < 0.999:
            base = f"({base}) × штраф {penalty:.2f} ({reason})"

    line = f"score {score:.3f} = {base}"
    if zscore is not None:
        line += f" · z {float(zscore):+.2f} (справочно)"
    return line


def _dev_matches_line(md: dict) -> str:
    """Где именно нашлись слова запроса: файл, якорь, тело, misuse."""
    bits = []
    if md.get("in_file"):
        bits.append("файл — " + ", ".join(md["in_file"][:8]))
    if md.get("in_anchor"):
        bits.append("якорь — " + ", ".join(md["in_anchor"][:8]))
    in_search = md.get("in_searchable") or md.get("in_text") or []
    if in_search:
        bits.append("текст — " + ", ".join(in_search[:8]))
    in_misuse = md.get("in_misuse") or []
    if in_misuse and (md.get("misuse_hit") or float(md.get("penalty_misuse") or 0) > 0):
        bits.append("misuse — " + ", ".join(in_misuse[:8]))
    return "; ".join(bits)


def render_dev_result_info(
    *,
    score,
    sim,
    bonus,
    penalty,
    md: dict,
    keyword_bonus: float,
    zscore: float | None = None,
    low_confidence: bool = False,
) -> None:
    """Компактная dev-сводка: как получился score, где совпали слова, предупреждение."""
    st.caption(_dev_score_line(score, sim, bonus, penalty, md, keyword_bonus, zscore))
    matches = _dev_matches_line(md)
    if matches:
        st.caption("Совпало: " + matches)
    elif md.get("lexical_enabled", True):
        st.caption("Точных совпадений нет — только по смыслу.")
    if low_confidence:
        st.caption("Возможно, нерелевантно.")


def dev_legend(keyword_bonus: float) -> str:
    """Одна строка-легенда для dev-режима."""
    return (
        f"dev: score = (схожесть + {float(keyword_bonus):.2f} × лексика) × штраф. "
        "Схожесть — смысловая близость, лексика — совпадения слов, "
        "штраф — множитель за misuse или слабое совпадение. "
        "Результаты источников сортируются по score; z показан справочно "
        "и на порядок не влияет."
    )


def _nearest_suggestions(similarities, docs, limit: int = 3) -> list[dict]:
    """Ближайшие по схожести документы — подсказки, когда ничего не нашлось."""
    try:
        sims = np.asarray(similarities, dtype=np.float32)
    except Exception:
        return []
    if sims.size == 0:
        return []
    k = max(0, int(limit))
    if not k:
        return []
    order = np.argsort(sims)[::-1][:k]
    return [{"index": int(i), "sim": float(sims[i])} for i in order]


# Кэш переводов на время сессии: один и тот же запрос переводится один раз,
# независимо от вкладки («Архетипы» / «Поля») и числа повторных поисков.
_TRANSLATION_CACHE_KEY = "translation_cache"


def _translate_query_cached(
    cache: dict,
    source: str,
    provider: Any,
    *,
    system: str | None,
    user_template: str | None,
) -> tuple[str, str]:
    """Перевод для CKM с кэшем: повторный запрос не идёт в LLM.

    В кэш попадают только успешные переводы — если LLM недоступен или вернул
    ошибку, следующий поиск попробует перевести снова.
    """
    source = (source or "").strip()
    if not source:
        return source, "not_needed"
    # Промпты входят в ключ: после их правки в dev-режиме запрос
    # переводится заново.
    key = (source, system or "", user_template or "")
    hit = cache.get(key)
    if hit is not None:
        return hit
    query_en, note = translate_query_to_en_mixed(
        source,
        provider,
        system=system,
        user_template=user_template,
    )
    if note == "translated":
        cache[key] = (query_en, note)
    return query_en, note


def _apply_pick_all(prefix: str) -> None:
    """on_change «Выбрать все»: ставит или снимает галочки у видимых результатов."""
    checked = bool(st.session_state.get(f"{prefix}_pick_all"))
    for key in st.session_state.get(f"{prefix}_pick_keys") or []:
        st.session_state[key] = checked


def _sync_pick_all(prefix: str) -> None:
    """on_change отдельного результата: «Выбрать все» активно, только если выбраны все."""
    keys = st.session_state.get(f"{prefix}_pick_keys") or []
    st.session_state[f"{prefix}_pick_all"] = bool(keys) and all(
        bool(st.session_state.get(key)) for key in keys
    )


def _clear_expand_term_widgets(prefix: str, terms) -> None:
    """Сбрасывает состояние чекбоксов для старого списка расширений."""
    for term in terms or []:
        st.session_state.pop(f"{prefix}_expand_term_{term}", None)


def render_search_tab(ctx: TabContext, spec: SearchTabSpec, docs, vectors) -> None:
    prefix = spec.prefix
    st.caption(spec.top_caption)
    # Текущие промпты из session_state (учитывает правки dev-панели).
    parts = prompts_from_session()

    if f"{prefix}_searched" not in st.session_state:
        st.session_state[f"{prefix}_searched"] = False
    st.session_state.setdefault(f"{prefix}_expand_terms", [])
    st.session_state.setdefault(f"{prefix}_expand_query", "")
    st.session_state.setdefault(f"{prefix}_expand_last_msg", "")
    st.session_state.setdefault(f"{prefix}_expand_last_kind", "")
    st.session_state.setdefault(f"{prefix}_expand_results", False)
    st.session_state.setdefault(f"{prefix}_query_en", "")
    # Текст запроса храним отдельно: состояние виджета в форме очищается,
    # когда вкладка не рендерится (переключение на «Поля» и обратно).
    st.session_state.setdefault(f"{prefix}_query_saved", "")

    enabled_sources = [s for s in (ctx.sources or []) if s.enabled]
    show_source_picker = len(enabled_sources) > 1
    source_mode = None

    def _mode_label(mode: str) -> str:
        if mode == "__all__":
            labels = [ctx.source_labels.get(s.name, s.name) for s in enabled_sources]
            # Две строки, чтобы радиокнопка смешанного режима стояла на одном
            # уровне с остальными источниками (у них вторая строка — дата).
            if len(labels) <= 1:
                return "".join(labels)
            return f"{labels[0]}  \n+ {' + '.join(labels[1:])}"
        label = ctx.source_labels.get(mode, mode)
        date_s = _index_date_for_source(enabled_sources, mode, spec.index_kind)
        return _source_label_with_date(label, date_s)

    with st.form(f"{prefix}_search_form", clear_on_submit=False, border=False):
        q1, q2 = st.columns([6, 1], vertical_alignment="bottom")
        with q1:
            query = st.text_input(
                "Запрос",
                value=st.session_state.get(f"{prefix}_query_saved", ""),
                help=spec.query_help,
                key=f"{prefix}_query",
                autocomplete="off",
            )
        with q2:
            run_search = st.form_submit_button("Искать", use_container_width=True)

        if show_source_picker:
            modes = [s.name for s in enabled_sources] + ["__all__"]
            radio_key = f"{prefix}_source_mode"
            saved_mode_key = f"{prefix}_source_mode_saved"
            saved_mode = st.session_state.get(saved_mode_key) or modes[0]
            if saved_mode not in modes:
                saved_mode = modes[0]
            # Чиним устаревшее значение и восстанавливаем выбор после смены вкладки.
            if st.session_state.get(radio_key) not in modes:
                st.session_state[radio_key] = saved_mode
            source_mode = st.radio(
                "Источники",
                options=modes,
                index=modes.index(saved_mode),
                horizontal=True,
                format_func=_mode_label,
                key=radio_key,
                help=(
                    "При выборе CKM русскоязычный запрос переводится с помощью ИИ "
                    "(если доступен). Англоязычный запрос не переводится."
                ),
            )
            st.session_state[saved_mode_key] = source_mode
            picked = {s.name for s in enabled_sources} if source_mode == "__all__" else {source_mode}
            # Подсказка нужна только для нелатинского запроса: английский не переводится.
            if (
                "ckm" in picked
                and (ctx.llm_aux_provider is None or ctx.llm_setup_msg)
                and not looks_latin(query)
            ):
                st.caption(
                    "Для CKM запрос переводится на английский с помощью ИИ — "
                    "ИИ недоступен, поиск производится только по смыслу, качество выдачи может быть ниже."
                )

        with st.expander(
            "Расширение запроса (ИИ)",
            expanded=bool(st.session_state.get(f"{prefix}_expand_results", False)),
        ):
            st.caption(
                "По кнопке ИИ предложит синонимы/переформулировки исходного запроса. "
                "Отметьте нужные варианты — они добавятся к запросу."
            )
            find_expand = st.form_submit_button("Предложить расширения", use_container_width=True)

            terms = list(st.session_state.get(f"{prefix}_expand_terms") or [])
            if (st.session_state.get(f"{prefix}_expand_query") or "") != (query or ""):
                _clear_expand_term_widgets(prefix, terms)
                terms = []
                st.session_state[f"{prefix}_expand_terms"] = []
                st.session_state[f"{prefix}_expand_selected"] = []
                st.session_state[f"{prefix}_expand_last_kind"] = ""
                st.session_state[f"{prefix}_expand_last_msg"] = ""
                st.session_state[f"{prefix}_expand_results"] = False

            # По умолчанию ничего не отмечено: пользователь выбирает варианты сам.
            st.session_state.setdefault(f"{prefix}_expand_selected", [])

            if find_expand:
                st.session_state[f"{prefix}_expand_results"] = False
                _clear_expand_term_widgets(prefix, terms)
                if not (query or "").strip():
                    st.session_state[f"{prefix}_expand_terms"] = []
                    st.session_state[f"{prefix}_expand_selected"] = []
                    st.session_state[f"{prefix}_expand_query"] = query or ""
                    st.session_state[f"{prefix}_expand_last_kind"] = "info"
                    st.session_state[f"{prefix}_expand_last_msg"] = (
                        "Введите запрос и нажмите «Предложить расширения»."
                    )
                elif ctx.llm_setup_msg or ctx.llm_aux_provider is None:
                    st.session_state[f"{prefix}_expand_terms"] = []
                    st.session_state[f"{prefix}_expand_selected"] = []
                    st.session_state[f"{prefix}_expand_query"] = query or ""
                    st.session_state[f"{prefix}_expand_last_kind"] = "warning"
                    st.session_state[f"{prefix}_expand_last_msg"] = (
                        ctx.llm_setup_msg or "LLM не настроен — расширение недоступно."
                    )
                else:
                    try:
                        terms_new = suggest_query_expansions(
                            query,
                            ctx.llm_aux_provider,
                            system=parts["query_expand"],
                            user_template=parts["query_expand_user"],
                            max_terms=_EXPAND_TERMS_LIMIT,
                        )
                        st.session_state[f"{prefix}_expand_terms"] = terms_new
                        st.session_state[f"{prefix}_expand_query"] = query or ""
                        if not terms_new:
                            st.session_state[f"{prefix}_expand_selected"] = []
                            st.session_state[f"{prefix}_expand_last_kind"] = "info"
                            st.session_state[f"{prefix}_expand_last_msg"] = "ИИ не предложил терминов."
                        else:
                            st.session_state[f"{prefix}_expand_selected"] = []
                            st.session_state[f"{prefix}_expand_last_kind"] = "success"
                            st.session_state[f"{prefix}_expand_last_msg"] = f"Предложено: {len(terms_new)}"
                            st.session_state[f"{prefix}_expand_results"] = True
                        terms = terms_new
                    except LLMError as e:
                        st.session_state[f"{prefix}_expand_terms"] = []
                        st.session_state[f"{prefix}_expand_selected"] = []
                        st.session_state[f"{prefix}_expand_last_kind"] = "warning"
                        st.session_state[f"{prefix}_expand_last_msg"] = e.user_message
                        logger.warning("query expansion failed (LLMError): %s", e.user_message)
                        terms = []
                    except Exception as e:
                        st.session_state[f"{prefix}_expand_terms"] = []
                        st.session_state[f"{prefix}_expand_selected"] = []
                        st.session_state[f"{prefix}_expand_last_kind"] = "warning"
                        st.session_state[f"{prefix}_expand_last_msg"] = f"Ошибка расширения: {e}"
                        logger.exception("query expansion failed: %s", e)
                        terms = []

            last_kind = st.session_state.get(f"{prefix}_expand_last_kind") or ""
            last_msg = (st.session_state.get(f"{prefix}_expand_last_msg") or "").strip()
            if last_kind and last_msg:
                if last_kind == "success":
                    st.success(last_msg)
                elif last_kind == "warning":
                    st.warning(last_msg)
                else:
                    st.info(last_msg)

            if terms:
                stored_selected = [
                    t for t in (st.session_state.get(f"{prefix}_expand_selected") or []) if t in terms
                ]
                st.caption("Отметьте варианты, которые нужно добавить к запросу:")
                selected_now: list[str] = []
                for term in terms[:_EXPAND_TERMS_LIMIT]:
                    checked = st.checkbox(
                        term,
                        value=term in stored_selected,
                        key=f"{prefix}_expand_term_{term}",
                    )
                    if checked:
                        selected_now.append(term)
                st.session_state[f"{prefix}_expand_selected"] = selected_now
                if selected_now:
                    st.caption(
                        "Расширенный запрос: "
                        + apply_manual_expansions(query, selected_now)
                    )

    # Уникальный маркер заставляет iframe перезагрузиться при смене вкладки/рендере:
    # иначе JS-обёртка даты не видит новые подписи на вкладке «Поля».
    st.session_state["_source_dates_seq"] = int(st.session_state.get("_source_dates_seq") or 0) + 1
    st.iframe(
        _SOURCE_DATES_JS + f"\n<!-- {prefix}-{st.session_state['_source_dates_seq']} -->",
        height=1,
    )

    query = st.session_state.get(f"{prefix}_query", "")
    if run_search or find_expand:
        st.session_state[f"{prefix}_query_saved"] = query
    short_token_hint(query, vocab=st.session_state.get(spec.vocab_key), dev_mode=ctx.dev_mode)

    saved_filter_key = f"{spec.filter_key}_saved"
    saved_filter = [
        x for x in (st.session_state.get(saved_filter_key) or []) if x in spec.filter_options
    ]
    filter_selection = st.multiselect(
        spec.filter_label,
        options=spec.filter_options,
        default=saved_filter,
        help=spec.filter_help,
        key=spec.filter_key,
    )
    st.session_state[saved_filter_key] = filter_selection

    if source_mode is None or source_mode == "__all__":
        selected_sources = {s.name for s in enabled_sources}
    else:
        selected_sources = {source_mode}

    if docs is None or vectors is None:
        st.warning("Индекс ещё не построен — поиск по нему недоступен.")
    elif run_search and query.strip():
        f_docs, f_vecs, back = docs, vectors, list(range(len(docs)))
        need_source_filter = len(enabled_sources) > 1 and bool(selected_sources)
        if filter_selection or need_source_filter:
            src_ok = set(selected_sources)

            def _keep(d: dict) -> bool:
                if filter_selection and not spec.filter_matches(filter_selection, d):
                    return False
                if need_source_filter:
                    return (d.get("source") or "local") in src_ok
                return True

            f_docs, f_vecs, back = filter_docs(docs, vectors, _keep)

        if len(f_docs) == 0 or f_vecs.size == 0:
            st.session_state[f"{prefix}_top_indices"] = []
            st.session_state[f"{prefix}_visible_count"] = 0
            st.session_state[f"{prefix}_match_debug"] = {}
            st.session_state[f"{prefix}_scores"] = []
            st.session_state[f"{prefix}_zscore_by_idx"] = {}
            st.session_state[f"{prefix}_searched"] = True
        else:
            expanded_query = None
            selected_expand_terms = list(st.session_state.get(f"{prefix}_expand_selected") or [])
            if selected_expand_terms:
                expanded_query = apply_manual_expansions(query, selected_expand_terms)
            st.session_state[f"{prefix}_expanded_query"] = expanded_query or ""

            # Перевод для CKM: используем его только для лексических совпадений
            # с английскими текстами. Схожесть считается исходным запросом.
            translation_source = (expanded_query or query or "").strip()
            query_en = translation_source
            translation_note = ""
            if "ckm" in selected_sources:
                if not looks_latin(translation_source):
                    if ctx.llm_aux_provider is None or ctx.llm_setup_msg:
                        translation_note = "no_llm"
                    else:
                        cache = st.session_state.setdefault(_TRANSLATION_CACHE_KEY, {})
                        try:
                            with st.spinner("Перевод запроса на английский (для CKM)..."):
                                query_en, translation_note = _translate_query_cached(
                                    cache,
                                    translation_source,
                                    ctx.llm_aux_provider,
                                    system=parts["translate"],
                                    user_template=parts["translate_user"],
                                )
                        except Exception:
                            logger.exception("query translation failed; using original query for CKM")
                            translation_note = "failed"
                else:
                    translation_note = "not_needed"

            with st.spinner("Подготовка модели и поиск..."):
                _run_pending_dim_check(ctx)
                scores, similarities, bonus, penalty, match_debug = compute_scores_multilingual(
                    query=query,
                    docs=f_docs,
                    vectors=f_vecs,
                    embedder=ctx.embedder,
                    query_en=query_en,
                    keyword_bonus=ctx.keyword_bonus,
                    misuse_penalty=ctx.misuse_penalty,
                    weak_match_penalty=ctx.weak_match_penalty,
                    enable_misuse_penalty=ctx.enable_misuse_penalty,
                    enable_weak_match_penalty=ctx.enable_weak_match_penalty,
                    token_vocab=st.session_state.get(spec.vocab_key),
                    expanded_query=expanded_query,
                )

            adjusted_scores: dict[int, float] = {}

            def _rank_group(idxs: list[int]):
                ordered, group_scores = reorder_picked_with_simi_preference(
                    [int(i) for i in idxs], f_docs, scores, eps=float(ctx.cfg.simi_pref_eps)
                )
                for k, v in (group_scores or {}).items():
                    adjusted_scores[int(k)] = float(v)
                return ordered, group_scores

            cross_floor = _effective_cross_floor(ctx, spec)
            pure_floors = pure_floor_by_doc_lang(
                f_docs,
                query=query,
                expanded_query=expanded_query,
                same_floor=float(ctx.sim_floor_pure),
                cross_floor=cross_floor,
            )
            top_local, sel_info = select_indices_by_relevance(
                scores,
                similarities,
                bonus,
                strictness=ctx.auto_strictness,
                max_k=int(ctx.max_results),
                min_k=spec.min_k,
                sources=[d.get("source") or "local" for d in f_docs],
                sim_floor=float(ctx.sim_floor),
                sim_floor_pure=float(ctx.sim_floor_pure),
                sim_floor_pure_by_idx=pure_floors,
                extend_k=_SHOW_MORE_LIMIT,
                rank_within_group=_rank_group,
            )

            if ctx.dev_mode:
                groups_info = sel_info.get("groups") or {}
                selection_parts = [
                    f"{name}: {len(info.get('picked') or [])}/{info.get('total')}"
                    for name, info in groups_info.items()
                ]
                selection_parts.append(
                    f"кандидатов: {len(sel_info.get('candidates') or [])}"
                )
                st.caption(
                    "Отбор: gap ≤ {0:.3f}, полоса {1:.3f}, порог схожести {2:.2f}/{3:.2f} "
                    "(другой язык {4:.2f}, пометка ниже {5:.2f})".format(
                        float(sel_info.get("gap_max") or 0.0),
                        float(sel_info.get("band") or 0.0),
                        float(sel_info.get("sim_floor") or 0.0),
                        float(sel_info.get("sim_floor_pure") or 0.0),
                        cross_floor,
                        float(ctx.sim_low_confidence),
                    )
                    + ((" | " + "; ".join(selection_parts)) if selection_parts else "")
                )
                used_q = (expanded_query or query or "").strip()
                if used_q and used_q != (query or "").strip():
                    with st.expander("Как расширился запрос (для поиска/бонуса)", expanded=False):
                        st.caption("Исходный запрос:")
                        st.code((query or "").strip(), language=None)
                        st.caption("Использовано для поиска (embedding + keyword):")
                        st.code(used_q, language=None)

            # «Показать ещё»: храним расширенный список (первичный отбор + запас),
            # а показываем только первые `visible_count` из него.
            candidate_local = list(sel_info.get("candidates") or top_local)
            top_indices = [back[i] for i in candidate_local]
            debug_back = {back[i]: v for i, v in match_debug.items()}
            score_back = {back[i]: float(adjusted_scores.get(i, scores[i])) for i in candidate_local}
            sim_back = {back[i]: float(similarities[i]) for i in candidate_local}
            bonus_back = {back[i]: float(bonus[i]) for i in candidate_local}
            penalty_back = {back[i]: float(penalty[i]) for i in candidate_local}
            zscores = sel_info.get("zscores") or {}
            z_back = {back[i]: float(zscores.get(i, 0.0)) for i in candidate_local}

            st.session_state[f"{prefix}_top_indices"] = top_indices
            st.session_state[f"{prefix}_visible_count"] = min(len(top_local), _SHOW_MORE_LIMIT)
            # Новый поиск: «Выбрать все» сбрасываем, галочки прошлой выдачи не трогаем.
            st.session_state[f"{prefix}_pick_all"] = False
            st.session_state[f"{prefix}_match_debug"] = debug_back
            st.session_state[f"{prefix}_score_by_idx"] = score_back
            st.session_state[f"{prefix}_similarity_by_idx"] = sim_back
            st.session_state[f"{prefix}_bonus_by_idx"] = bonus_back
            st.session_state[f"{prefix}_penalty_by_idx"] = penalty_back
            st.session_state[f"{prefix}_zscore_by_idx"] = z_back
            st.session_state[f"{prefix}_suggestions"] = [
                {"index": back[s["index"]], "sim": s["sim"]}
                for s in _nearest_suggestions(similarities, f_docs, limit=3)
                if s["index"] < len(back)
            ]
            st.session_state[f"{prefix}_translation_note"] = translation_note
            st.session_state[f"{prefix}_query_en_used"] = query_en
            st.session_state[f"{prefix}_query_en"] = query_en
            st.session_state[f"{prefix}_scores"] = scores.tolist()
            st.session_state[f"{prefix}_searched"] = True

        st.session_state["last_query"] = query

    all_indices = st.session_state.get(f"{prefix}_top_indices") or []
    try:
        visible_count = int(st.session_state.get(f"{prefix}_visible_count") or 0)
    except (TypeError, ValueError):
        visible_count = 0
    if visible_count <= 0:
        visible_count = len(all_indices)
    visible_count = max(0, min(visible_count, len(all_indices), _SHOW_MORE_LIMIT))
    top_indices = all_indices[:visible_count]
    match_debug = st.session_state.get(f"{prefix}_match_debug") or {}
    score_by_idx = st.session_state.get(f"{prefix}_score_by_idx") or {}
    sim_by_idx = st.session_state.get(f"{prefix}_similarity_by_idx") or {}
    bonus_by_idx = st.session_state.get(f"{prefix}_bonus_by_idx") or {}
    penalty_by_idx = st.session_state.get(f"{prefix}_penalty_by_idx") or {}
    zscore_by_idx = st.session_state.get(f"{prefix}_zscore_by_idx") or {}

    st.markdown(CHAT_SIDEBAR_CSS, unsafe_allow_html=True)

    with st.container():
        # Результаты поиска занимают всю ширину: чат живёт в сайдбаре.
        if st.session_state.get(f"{prefix}_searched") and query.strip():
            used_q = (st.session_state.get(f"{prefix}_expanded_query") or "").strip()
            q_used = (st.session_state.get(f"{prefix}_query_en_used") or "").strip()
            note = st.session_state.get(f"{prefix}_translation_note") or ""
            query_bits: list[str] = []
            if used_q and used_q != (query or "").strip():
                label = "Локальная база" if "local" in selected_sources else "Запрос с расширениями"
                query_bits.append(f"{label}: «{used_q}»")
            if note == "translated" and q_used:
                query_bits.append(f"Перевод для лексики CKM: «{q_used}»")
            elif note == "not_needed" and q_used:
                query_bits.append(f"CKM: «{q_used}» (перевод не нужен)")
            elif note == "no_llm":
                query_bits.append("CKM: перевод недоступен — поиск по смыслу")
            elif note == "failed":
                query_bits.append("CKM: перевод не удался — поиск по смыслу")
            if query_bits:
                st.caption(" · ".join(query_bits))

        if st.session_state.get(f"{prefix}_searched") and query.strip() and not all_indices:
            st.info(
                "Ничего не найдено: нет релевантных результатов "
                "(порог {0:.2f} с буквальным совпадением, {1:.2f} — только по смыслу, "
                "{2:.2f} — только по смыслу для другого языка).".format(
                    float(ctx.sim_floor),
                    float(ctx.sim_floor_pure),
                    _effective_cross_floor(ctx, spec),
                )
            )
            note = st.session_state.get(f"{prefix}_translation_note") or ""
            if note == "no_llm":
                st.warning(
                    "CKM искался по смыслу: перевод запроса недоступен (LLM не настроен), "
                    "буквальные совпадения для CKM отключены."
                )
            elif note == "failed":
                st.warning(
                    "Перевод запроса для CKM не удался — поиск идёт по смыслу, "
                    "буквальные совпадения для CKM отключены."
                )
            suggestions = st.session_state.get(f"{prefix}_suggestions") or []
            if suggestions:
                st.caption("Ближайшие по смыслу (ниже порога):")
                for s in suggestions:
                    idx = int(s.get("index", -1))
                    if idx < 0 or idx >= len(docs):
                        continue
                    st.caption(f"· {spec.result_label(docs[idx])} (схожесть {float(s.get('sim') or 0):.2f})")

        selected_idx: list[int] = []
        try:
            persisted_selected = {
                int(i) for i in (st.session_state.get(f"{prefix}_selected_persisted") or [])
            }
        except (TypeError, ValueError):
            persisted_selected = set()
        if top_indices:
            if len(all_indices) > len(top_indices):
                st.write(f"{spec.result_count_label} {len(top_indices)} из {len(all_indices)}")
            else:
                st.write(f"{spec.result_count_label} {len(top_indices)}")

            if ctx.dev_mode:
                st.caption(dev_legend(ctx.keyword_bonus))

            low_conf = float(ctx.sim_low_confidence)
            weak_set = {
                idx
                for idx in top_indices
                if is_low_confidence(sim_by_idx.get(idx), bonus_by_idx.get(idx), low_conf)
            }
            if weak_set:
                if ctx.dev_mode:
                    st.warning(
                        "Слабые совпадения: схожесть ниже {0:.2f} у {1} из {2} — "
                        "результаты могут быть нерелевантными.".format(
                            low_conf,
                            len(weak_set),
                            len(top_indices),
                        )
                    )
                else:
                    st.warning("Часть результатов слабо совпадает с запросом и может быть нерелевантной.")

            # "Выбрать все" для текущей выдачи
            pick_keys = [spec.make_pick_key(docs[idx], idx) for idx in top_indices]
            st.session_state[f"{prefix}_pick_keys"] = pick_keys

            all_selected = (
                all(
                    (bool(st.session_state.get(k)) if k in st.session_state else (idx in persisted_selected))
                    for k, idx in zip(pick_keys, top_indices)
                )
                if pick_keys
                else False
            )
            st.checkbox(
                spec.pick_all_label,
                value=all_selected,
                key=f"{prefix}_pick_all",
                on_change=_apply_pick_all,
                args=(prefix,),
            )

            for n, idx in enumerate(top_indices, start=1):
                doc = docs[idx]
                key = spec.make_pick_key(doc, idx)
                score_s = score_by_idx.get(idx)
                sim_s = sim_by_idx.get(idx)
                bonus_s = bonus_by_idx.get(idx)
                penalty_s = penalty_by_idx.get(idx)
                z_s = zscore_by_idx.get(idx)
                md = match_debug.get(idx) or {}
                fn_full = doc.get("file", "unknown")
                fn = os.path.basename(fn_full)
                if fn.lower().endswith(".adl"):
                    fn = fn[:-4]
                with st.container(border=True):
                    c_pick, c_name = st.columns([1, 13], vertical_alignment="center")
                    with c_pick:
                        picked = st.checkbox(
                            f"#{n}",
                            key=key,
                            value=(
                                bool(st.session_state.get(key))
                                if key in st.session_state
                                else (idx in persisted_selected)
                            ),
                            on_change=_sync_pick_all,
                            args=(prefix,),
                        )
                    with c_name:
                        spec.render_result_name(doc, md, ctx.dev_mode, fn, doc.get("source") or "local")
                    if ctx.dev_mode:
                        render_dev_result_info(
                            score=score_s,
                            sim=sim_s,
                            bonus=bonus_s,
                            penalty=penalty_s,
                            md=md,
                            keyword_bonus=ctx.keyword_bonus,
                            zscore=z_s,
                            low_confidence=idx in weak_set,
                        )
                    elif idx in weak_set:
                        st.caption("Возможно, нерелевантно.")
                    if picked:
                        selected_idx.append(idx)
                    spec.render_result_details(doc, md, fn)

            st.session_state[f"{prefix}_selected"] = selected_idx
            st.session_state[f"{prefix}_selected_persisted"] = selected_idx

            remaining = len(all_indices) - len(top_indices)
            if remaining > 0:
                step = min(_SHOW_MORE_STEP, remaining)

                def _show_more(prefix=prefix) -> None:
                    # «Выбрать все» относится к видимым результатам: после догрузки
                    # снимаем его, уже отмеченные результаты остаются выбранными.
                    st.session_state[f"{prefix}_visible_count"] = min(
                        int(st.session_state.get(f"{prefix}_visible_count") or 0) + _SHOW_MORE_STEP,
                        len(st.session_state.get(f"{prefix}_top_indices") or []),
                        _SHOW_MORE_LIMIT,
                    )
                    st.session_state[f"{prefix}_pick_all"] = False

                st.button(
                    f"Показать ещё {step}",
                    key=f"{prefix}_show_more",
                    use_container_width=True,
                    help=f"Ещё {step} менее релевантных результатов (всего не более {_SHOW_MORE_LIMIT})",
                    on_click=_show_more,
                )

    def _on_chat_provider_change():
        kind = st.session_state.get("chat_provider_choice")
        if kind not in CHAT_PROVIDER_OPTIONS:
            return
        # Выбор действует только в текущей сессии. При следующем запуске
        # снова используется МЭШ.ИИ.Хаб.
        st.session_state["api_kind"] = kind

    with st.sidebar:
        with st.container(key=f"{prefix}_chat_panel"):
            st.markdown('<div class="chat-panel-title">Чат с ИИ</div>', unsafe_allow_html=True)

            has_history = bool(get_chat_messages(prefix))
            current_kind = selected_chat_provider()
            provider_labels = {"gigachat": "ГигаЧат", "corp_async": "МЭШ.ИИ.Хаб"}
            if ctx.cfg.llm_mode != "ollama" and current_kind in provider_labels:
                if st.session_state.get("chat_provider_choice") not in CHAT_PROVIDER_OPTIONS:
                    st.session_state["chat_provider_choice"] = current_kind
                st.radio(
                    "Провайдер",
                    options=list(CHAT_PROVIDER_OPTIONS),
                    horizontal=True,
                    format_func=lambda kind: provider_labels[kind],
                    help=(
                        "МЭШ.ИИ.Хаб работает только при активном VPN-подключении "
                        "через КриптоПро NGate. ГигаЧат доступен без VPN."
                    ),
                    key="chat_provider_choice",
                    on_change=_on_chat_provider_change,
                )
            elif ctx.cfg.llm_mode == "ollama":
                st.caption("Активен локальный провайдер Ollama (dev-режим).")
            else:
                st.caption("Активен OpenAI-совместимый провайдер (выбран в dev-настройках).")

            if not top_indices and not has_history:
                st.caption("Выполните поиск и выберите нужные результаты, чтобы начать чат с ИИ")
            else:
                with st.container(key=f"{prefix}_chat_history"):
                    render_chat_messages(prefix)
                    # Сюда временно рисуем только что отправленный вопрос,
                    # пока ИИ готовит ответ.
                    pending_message_placeholder = st.empty()

                    sel = st.session_state.get(f"{prefix}_selected") or []
                    dialog_stale = has_history and rag_selection_stale(prefix, sel)
                    starting_new = not has_history or dialog_stale

                    if ctx.dev_mode:
                        # В dev-режиме контекст виден всегда: до первого ответа — тот,
                        # что уйдёт в модель, после — контекст из истории диалога.
                        if starting_new:
                            if sel:
                                ctx_preview = spec.build_context(
                                    query,
                                    [docs[i] for i in sel],
                                    embedder=ctx.embedder,
                                    query_en=st.session_state.get(f"{prefix}_query_en", ""),
                                    expanded_query=st.session_state.get(f"{prefix}_expanded_query", ""),
                                )
                                ctx_title = "Контекст для ИИ (новый запрос)"
                                ctx_note = "Будет отправлен со следующим сообщением."
                            else:
                                ctx_preview = ""
                                ctx_title = "Контекст для ИИ"
                                ctx_note = "Появится после выбора результатов поиска."
                        else:
                            ctx_preview = st.session_state.get(f"{prefix}_rag_context") or ""
                            ctx_title = "Контекст для ИИ (из истории диалога)"
                            pinned_query = st.session_state.get(f"{prefix}_rag_query") or ""
                            ctx_note = (
                                f"Запрос, для которого собран контекст: {pinned_query}"
                                if pinned_query
                                else "Контекст из первого сообщения текущего диалога."
                            )
                        if ctx_preview:
                            with st.expander(ctx_title, expanded=False):
                                size = f"{len(ctx_preview):,}".replace(",", " ")
                                st.caption(f"{ctx_note} · {size} символов")
                                st.text(ctx_preview)
                        else:
                            st.caption(f"{ctx_title}: {ctx_note}")
                        render_usage_caption(prefix)

                st.iframe(_CHAT_SCROLL_JS.replace("__PREFIX__", prefix), height=1)
                # Сюда же попадёт будущий ответ: сначала спиннер, после rerun — сообщения.
                thinking_placeholder = st.empty()

                with st.container(key=f"{prefix}_chat_composer"):
                    if starting_new and sel and spec.context_tokens_key:
                        budget = int(
                            st.session_state.get(spec.context_tokens_key) or 0
                        )
                        if budget > 0:
                            st.caption(
                                "Контекст для ИИ ограничен ~"
                                f"{budget:,} токенами.".replace(",", " ")
                            )
                    if not sel:
                        st.caption(
                            spec.select_first_warning
                            if starting_new
                            else "Отметьте хотя бы один результат, чтобы продолжить диалог."
                        )
                    if dialog_stale:
                        st.warning(spec.stale_warning)

                    placeholder = "Спросить у ИИ…" if starting_new else "Уточняющий вопрос…"
                    chat_key = f"{prefix}_ai_input"
                    if starting_new:
                        # Первое сообщение предзаполняем редактируемым промптом.
                        prefill_sig = (
                            f"{query}\x00{spec.default_question_key}\x00{int(dialog_stale)}"
                        )
                        if st.session_state.get(f"{prefix}_ai_prefill_sig") != prefill_sig:
                            st.session_state[chat_key] = format_default_question(
                                parts, spec.default_question_key, query
                            )
                            st.session_state[f"{prefix}_ai_prefill_sig"] = prefill_sig
                    else:
                        st.session_state.pop(f"{prefix}_ai_prefill_sig", None)
                    reset = False
                    # Enter отправляет сообщение; Shift+Enter — перенос строки.
                    user_q = st.chat_input(
                        placeholder,
                        key=chat_key,
                        max_chars=4000,
                        disabled=not sel,
                    )
                    submitted = user_q is not None
                    if has_history:
                        reset = st.button(
                            "Сбросить чат",
                            use_container_width=True,
                            key=f"{prefix}_ai_reset",
                        )

                if reset:
                    clear_chat_session(prefix)
                    st.rerun()

                if submitted:
                    sel = st.session_state.get(f"{prefix}_selected") or []
                    question = (user_q or "").strip()
                    if not sel:
                        with thinking_placeholder.container():
                            st.warning(spec.select_first_warning)
                    elif ctx.llm_setup_msg or ctx.llm_provider is None:
                        with thinking_placeholder.container():
                            st.warning(ctx.llm_setup_msg or "LLM не настроен: проверьте настройки LLM.")
                    elif not starting_new:
                        if not question:
                            with thinking_placeholder.container():
                                st.warning("Введите уточняющий вопрос.")
                        else:
                            send_followup(
                                prefix=prefix,
                                question=question,
                                stale_warning_text=spec.stale_warning,
                                llm_setup_msg=ctx.llm_setup_msg,
                                llm_provider=ctx.llm_provider,
                                prompts=parts,
                                status_placeholder=thinking_placeholder,
                                pending_message_placeholder=pending_message_placeholder,
                            )
                    else:
                        q = question or format_default_question(
                            parts, spec.default_question_key, query
                        )
                        selected_docs = [docs[i] for i in sel]
                        ctx_text = spec.build_context(
                            query,
                            selected_docs,
                            embedder=ctx.embedder,
                            query_en=st.session_state.get(f"{prefix}_query_en", ""),
                            expanded_query=st.session_state.get(f"{prefix}_expanded_query", ""),
                        )
                        st.session_state[f"{prefix}_rag_context"] = ctx_text
                        st.session_state[f"{prefix}_rag_query"] = query
                        st.session_state[f"{prefix}_rag_selection_key"] = make_rag_selection_key(sel)
                        # Сразу показываем вопрос, чтобы он не исчезал на время ответа.
                        append_chat_message(prefix, "user", q)
                        render_pending_chat_message(
                            pending_message_placeholder, prefix, q
                        )
                        try:
                            prompt = apply_placeholders(
                                prompt_text(parts, "answer_user_prompt"),
                                context=ctx_text,
                                question=q,
                            )
                            with thinking_placeholder.container():
                                with st.spinner("ИИ думает..."):
                                    sys_p = compose_system_prompt(
                                        parts,
                                        query=query,
                                    )
                                    dialog = make_llm_dialog(system_prompt=sys_p, user_prompt=prompt)
                                    ans = chat_with_continuation(
                                        ctx.llm_provider,
                                        dialog,
                                        continuation_prompt=str(
                                            parts.get("continuation_prompt") or ""
                                        ),
                                    )
                            store_last_usage(prefix, ctx.llm_provider)
                            st.session_state["last_error"] = ""
                            st.session_state["last_error_details"] = ""
                            st.session_state[f"{prefix}_dialog_messages"] = dialog + [
                                {"role": "assistant", "content": ans}
                            ]
                            append_chat_message(prefix, "assistant", ans)
                            st.session_state.pop(chat_key, None)
                            st.session_state.pop(f"{prefix}_ai_prefill_sig", None)
                            st.rerun()
                        except Exception as e:
                            report_llm_error(e)

                if st.session_state.get("last_error"):
                    with thinking_placeholder.container():
                        st.error(st.session_state["last_error"])
                        if ctx.dev_mode and st.session_state.get("last_error_details"):
                            with st.expander("Подробнее (для диагностики)", expanded=False):
                                st.text(st.session_state["last_error_details"][:4000])