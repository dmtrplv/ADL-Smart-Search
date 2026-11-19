"""Общий движок раздела поиска (используется «Поиск по архетипам» и «Поиск по полям»).

Раньше оба раздела почти полностью дублировали друг друга (~400–500 строк каждый).
Вся общая логика — форма поиска, расширение запроса ИИ, фильтр, индексация,
подсчёт и авто-выдача, рендер результатов с выбором, вопрос к ИИ, диалог
уточнений — вынесена сюда и переиспользуется через SearchTabSpec.
"""

import hashlib
import html
import json
import logging
import os
import re
import unicodedata
import uuid
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
    translate_context_to_ru,
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
from adl_smart_search.scoring import _SHORT_STOPWORDS, is_lexical_match, phrase_matches
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

# Очень сильное лексическое совпадение (нормированный BM25) может пройти
# с чуть меньшей семантической близостью. Это возвращает точные короткие
# запросы («Рост» → height, «Вес» → body_weight), не открывая дорогу мусору.
_STRONG_LEX_MIN = 0.9
_STRONG_LEX_SIM_FLOOR = 0.22

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

# История запросов: выпадающий список прямо под полем ввода.
# Нативный <datalist> в Chromium ставит попап не по левому краю, поэтому
# рисуем свой маленький список, привязанный к границам input.
_QUERY_HISTORY_JS = """
<style>html,body{margin:0;padding:0;height:0;overflow:hidden}</style>
<script>
setTimeout(() => {
  try {
    const doc = window.parent.document;
    const win = doc.defaultView || window.parent;
    // Поле ищем заново при каждом событии: Streamlit может пересоздать input
    // после перерисовки, и ссылка на старый элемент становится мёртвой.
    const field = () => doc.querySelector('.st-key-__PREFIX___query input')
      || doc.querySelector('input[aria-label="Запрос"]');
    if (!field()) return;
    const values = __VALUES__;
    const oldDatalist = doc.getElementById('adl-query-history-__PREFIX__');
    if (oldDatalist) oldDatalist.remove();

    win.__adlHistoryState = win.__adlHistoryState || {};
    const state = win.__adlHistoryState['__PREFIX__']
      || (win.__adlHistoryState['__PREFIX__'] = {});
    // Номер инъекции защищает от старых iframe-инъекций, которые догружаются
    // позже новых. Токен сбрасывает счётчик при перезапуске сессии.
    if (state.token !== '__TOKEN__') {
      state.token = '__TOKEN__';
      state.seq = 0;
    }
    const mySeq = __SEQ__;
    if (state.seq && mySeq < state.seq) return;
    state.seq = mySeq;
    state.field = field;
    state.values = values;

    // Ищем первый непрозрачный фон вверх по дереву — это фон поля в текущей теме.
    const isTransparent = c => !c || c === 'transparent'
      || /^rgba\\(0,\\s*0,\\s*0,\\s*0\\)$/.test(c);
    const withAlpha = (c, a) => {
      const m = /^rgba?\\(([^)]+)\\)$/.exec(c || '');
      if (!m) return 'rgba(128,128,128,' + a + ')';
      const p = m[1].split(',').map(s => s.trim());
      return 'rgba(' + p[0] + ',' + p[1] + ',' + p[2] + ',' + a + ')';
    };
    state.colors = inp => {
      let bg = '';
      let node = inp;
      while (node) {
        const c = win.getComputedStyle(node).backgroundColor;
        if (!isTransparent(c)) { bg = c; break; }
        node = node.parentElement;
      }
      if (!bg) bg = win.getComputedStyle(doc.body).backgroundColor;
      return { bg, color: win.getComputedStyle(inp).color || '' };
    };

    if (state.menu && !state.menu.isConnected) state.menu = null;
    if (!state.menu) {
      const menuId = 'adl-query-history-menu-__PREFIX__';
      const existing = doc.getElementById(menuId);
      if (existing) existing.remove();
      const menu = doc.createElement('div');
      menu.id = menuId;
      // Цвета берём не из темы Streamlit, а из вычисленных стилей самого поля:
      // так список совпадает с темой и не зависит от имён CSS-переменных.
      menu.style.cssText = [
        'position:fixed', 'display:none', 'z-index:1000000',
        'box-sizing:border-box', 'max-height:280px', 'overflow-y:auto',
        'padding:4px', 'border-radius:8px',
        'border:1px solid rgba(128,128,128,.35)',
        'box-shadow:0 8px 24px rgba(0,0,0,.18)',
        "font-family:inherit", 'font-size:.875rem'
      ].join(';');
      doc.body.appendChild(menu);
      state.menu = menu;
    }

    state.hide = () => { state.menu.style.display = 'none'; };
    // Позицию считаем отдельно: она нужна и при открытии, и при прокрутке.
    state.place = inp => {
      const rect = inp.getBoundingClientRect();
      state.menu.style.left = rect.left + 'px';
      state.menu.style.top = (rect.bottom + 4) + 'px';
      state.menu.style.width = rect.width + 'px';
    };
    state.render = () => {
      const inp = state.field();
      const menu = state.menu;
      if (!inp || doc.activeElement !== inp) { state.hide(); return; }
      // Клик по подписи «Запрос» фокусирует поле через label[for]; чтобы список
      // не открывался сразу после клика мимо, коротко подавляем показ.
      if (state.suppressUntil && Date.now() < state.suppressUntil) {
        state.hide();
        return;
      }
      const q = (inp.value || '').trim().toLowerCase();
      const items = (state.values || [])
        .filter(v => !q || String(v).toLowerCase().includes(q))
        .slice(0, 10);
      if (!items.length) { state.hide(); return; }

      const colors = state.colors(inp);
      if (colors.bg) menu.style.background = colors.bg;
      if (colors.color) menu.style.color = colors.color;
      state.hoverBg = withAlpha(colors.color, 0.12);
      state.place(inp);
      menu.innerHTML = '';
      for (const value of items) {
        const row = doc.createElement('div');
        row.textContent = value;
        row.style.cssText = [
          'padding:7px 9px', 'border-radius:5px', 'cursor:pointer',
          'white-space:nowrap', 'overflow:hidden', 'text-overflow:ellipsis'
        ].join(';');
        row.addEventListener('mouseenter', () => {
          row.style.background = state.hoverBg || 'rgba(128,128,128,.18)';
        });
        row.addEventListener('mouseleave', () => {
          row.style.background = 'transparent';
        });
        row.addEventListener('mousedown', e => e.preventDefault());
        row.addEventListener('click', () => {
          const target = state.field() || inp;
          const setter = Object.getOwnPropertyDescriptor(
            doc.defaultView.HTMLInputElement.prototype, 'value'
          ).set;
          setter.call(target, value);
          target.dispatchEvent(new Event('input', { bubbles: true }));
          target.dispatchEvent(new Event('change', { bubbles: true }));
          state.hide();
          target.focus();
        });
        menu.appendChild(row);
      }
      menu.style.display = 'block';
    };

    const isField = node => {
      const inp = state.field();
      return !!inp && (node === inp || inp.contains(node));
    };

    // Слушатели живут на документе, а не на самом поле: после перерисовки
    // Streamlit может заменить input, и обработчики на нём терялись — список
    // переставал открываться до переключения вкладок. Прошлый набор снимаем.
    win.__adlHistoryListeners = win.__adlHistoryListeners || {};
    for (const [target, type, fn, capture] of
      win.__adlHistoryListeners['__PREFIX__'] || []) {
      target.removeEventListener(type, fn, capture);
    }

    // Список позиционируется фиксированно, поэтому при прокрутке страницы его
    // нужно пересчитывать от поля, иначе он «отрывается» и висит на месте.
    let placeQueued = false;
    const keepPlaced = () => {
      if (placeQueued) return;
      placeQueued = true;
      try {
        win.requestAnimationFrame(() => {
          placeQueued = false;
          const states = win.__adlHistoryState || {};
          for (const key of Object.keys(states)) {
            const s = states[key];
            if (!s || !s.menu || s.menu.style.display !== 'block') continue;
            const inp = s.field && s.field();
            if (!inp) { s.hide(); continue; }
            const rect = inp.getBoundingClientRect();
            if (rect.bottom < 0 || rect.top > win.innerHeight) { s.hide(); continue; }
            s.place(inp);
          }
        });
      } catch (err) { placeQueued = false; }
    };

    const listeners = [
      [doc, 'focusin', e => {
        if (!isField(e.target)) return;
        state.render();
        try { win.requestAnimationFrame(() => state.render()); } catch (err) {}
      }, true],
      [doc, 'click', e => {
        if (isField(e.target)) state.render();
      }, true],
      [doc, 'input', e => {
        if (!isField(e.target)) return;
        state.suppressUntil = 0;
        state.render();
      }, true],
      [doc, 'focusout', e => {
        if (!isField(e.target)) return;
        setTimeout(() => {
          if (!isField(doc.activeElement)) state.hide();
        }, 120);
      }, true],
      [doc, 'keydown', e => {
        if (e.key === 'Escape' && isField(e.target)) state.hide();
      }, true],
      // Клик мимо поля и списка закрывает выпадашку.
      [doc, 'pointerdown', e => {
        const states = win.__adlHistoryState || {};
        for (const key of Object.keys(states)) {
          const s = states[key];
          if (!s || !s.menu || !s.field) continue;
          const inp = s.field();
          if (!inp) { s.hide(); continue; }
          const target = e.target;
          if (target === inp || inp.contains(target)) {
            s.suppressUntil = 0;
            continue;
          }
          if (s.menu.contains(target)) continue;
          s.hide();
          // Клик мог попасть в подпись «Запрос»: она связана с полем через
          // label[for] и сфокусирует input, что иначе открыло бы список снова.
          s.suppressUntil = Date.now() + 300;
        }
      }, true],
      [win, 'scroll', keepPlaced, true],
      [win, 'resize', keepPlaced],
    ];
    for (const [target, type, fn, capture] of listeners) {
      target.addEventListener(type, fn, capture);
    }
    win.__adlHistoryListeners['__PREFIX__'] = listeners;
    state.render();
  } catch (e) {}
}, 100);
</script>
"""

# Даты индексации в подписях источников оформляются markdown-кодом (``дата``):
# так они сразу рисуются мелким шрифтом, без JS-обёртки после первой отрисовки.
# Стили — в CHAT_SIDEBAR_CSS (селектор code внутри радиогруппы «Источники»).

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
  word-break: normal;
  overflow-wrap: normal;
  hyphens: none;
}
section[data-testid="stSidebar"] div.stButton > button * {
  word-break: normal;
  overflow-wrap: normal;
  hyphens: none;
}
section[data-testid="stSidebar"] div[data-testid="stTextInput"] {
  min-width: 0 !important;
}
.chat-panel-title {
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  opacity: 0.72;
  margin: 0 0 6px 0;
  line-height: 1.2;
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
  height: 2.2rem !important;
  min-height: 2.2rem !important;
  margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  height: calc(100% - 2.2rem);
  min-height: 0;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div:first-child {
  margin-top: 0 !important;
  padding-top: 0 !important;
}
section[data-testid="stSidebar"] [class*="chat_panel"] {
  height: calc(100vh - 2.2rem) !important;
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
[role="radiogroup"][aria-label="Источники"] label code {  font-family: inherit;
  font-size: 0.72em;
  opacity: 0.55;
  background: none;
  padding: 0;
  border: none;
  color: inherit;
  white-space: nowrap;
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
  background: rgba(255, 196, 0, 0.22);
  color: inherit;
  border-radius: 2px;
  padding: 0 1px;
}
.match-hl-phrase {
  background: rgba(255, 196, 0, 0.55);
  color: inherit;
  border-radius: 3px;
  padding: 0 2px;
  box-shadow: inset 0 -1px 0 rgba(160, 110, 0, 0.55);
}
.result-context {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: inherit;
  font-size: 0.85rem;
  margin: 0;
}
.field-focus {
  display: block;
  margin: 2px 0;
  padding: 2px 6px;
  border-left: 3px solid rgba(255, 196, 0, 0.85);
  border-radius: 4px;
  background: rgba(255, 196, 0, 0.14);
}
</style>
"""


def _word_highlight_pattern(keywords):
    """Регекс-кандидат и слова запроса для проверки однокоренности."""
    kws = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
    if not kws:
        return None
    parts = []
    for kw in dict.fromkeys(kws):
        stem = _stem_prefix(kw)
        parts.append(re.escape(stem) + (r"\w*" if stem != kw else r"\b"))
    pattern = re.compile(r"\b(?:" + "|".join(parts) + r")", flags=re.IGNORECASE)
    return pattern, list(dict.fromkeys(kws))


def _mark_highlighted_segment(segment: str, matcher, phrases) -> str:
    """Подсветка слов и фраз внутри уже экранированного фрагмента."""
    if not segment:
        return segment

    def mark_words(part: str) -> str:
        if matcher is None or not part:
            return part
        pattern, kws = matcher
        out: list[str] = []
        pos = 0
        for m in pattern.finditer(part):
            word = m.group(0)
            # regex даёт кандидатов по префиксу; подсвечиваем только однокоренные,
            # иначе «боль» подсвечивала бы «больше», а «макроцитарная» — «макроцефалию».
            if not any(is_lexical_match(kw, word) for kw in kws):
                continue
            out.append(part[pos : m.start()])
            out.append(f'<span class="match-hl">{word}</span>')
            pos = m.end()
        if not out:
            return part
        out.append(part[pos:])
        return "".join(out)

    spans = phrase_matches(segment, list(phrases or []))
    if not spans:
        return mark_words(segment)

    out: list[str] = []
    pos = 0
    for start, end in spans:
        out.append(mark_words(segment[pos:start]))
        out.append(f'<span class="match-hl-phrase">{segment[start:end]}</span>')
        pos = end
    out.append(mark_words(segment[pos:]))
    return "".join(out)


def _plain_context_html(text: str) -> str:
    return (
        '<div class="result-context" '
        'style="white-space: pre-wrap; word-break: break-word;">'
        f"{html.escape(text or '')}</div>"
    )


def render_context_with_translation(
    text: str,
    keywords,
    phrases,
    *,
    focus_span=None,
    cache_key: str,
    ctx,
    prompts: dict,
) -> None:
    """Показать CKM-контекст; при наличии LLM — кнопка перевода на русский.

    Перевод не влияет на поиск и индекс: это только отображение. Результат
    кэшируется в сессии по хэшу текста, поэтому rerender не вызывает LLM снова.
    """
    original = highlight_matches_html(
        text, keywords, phrases, focus_span=focus_span
    )
    provider = getattr(ctx, "llm_aux_provider", None)
    setup_msg = str(getattr(ctx, "llm_setup_msg", "") or "").strip()
    system_prompt = str((prompts or {}).get("ui_translate") or "").strip()
    user_template = str((prompts or {}).get("ui_translate_user") or "").strip()

    if provider is None or setup_msg:
        st.html(original)
        st.caption("Перевод недоступен: LLM не настроен.")
        return
    if not system_prompt or not user_template:
        st.html(original)
        st.caption(
            "Перевод не настроен: добавьте ui_translate и ui_translate_user "
            "в config/prompts.json."
        )
        return

    cache = st.session_state.setdefault("ui_translation_cache", {})
    if not isinstance(cache, dict):
        cache = {}
        st.session_state["ui_translation_cache"] = cache
    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    key = f"{cache_key}:{digest}"
    translated = str(cache.get(key) or "")

    if not translated:
        if st.button("Перевести на русский", key=f"tr_btn_{key}"):
            try:
                with st.spinner("Перевод контекста..."):
                    translated = translate_context_to_ru(
                        text,
                        provider,
                        system=system_prompt,
                        user_template=user_template,
                    )
                if translated:
                    cache[key] = translated
                else:
                    st.warning("ИИ не вернул перевод — показан оригинал.")
            except LLMError as e:
                st.warning(e.user_message)
            except Exception:
                logger.exception("context translation failed")
                st.warning("Не удалось перевести контекст — показан оригинал.")

    if not translated:
        st.html(original)
        return

    tab_tr, tab_orig = st.tabs(["Перевод", "Оригинал"])
    with tab_tr:
        st.html(_plain_context_html(translated))
    with tab_orig:
        st.html(original)


def highlight_matches_html(text: str, keywords, phrases=None, focus_span=None) -> str:
    """Текст с переносами, подсветкой фраз/слов и необязательной рамкой.

    ``focus_span`` — (start, end) в исходном тексте; этот фрагмент выделяется
    рамкой ``.field-focus`` (нужно для показа поля внутри структуры архетипа).
    """
    # Именно <div>, а не <pre>: Streamlit превращает <pre> в свой stMarkdownPre
    # с white-space: normal, и переносы строк схлопываются в сплошной текст.
    open_tag = (
        '<div class="result-context" '
        'style="white-space: pre-wrap; word-break: break-word;">'
    )
    raw = text or ""
    body = html.escape(raw)
    matcher = _word_highlight_pattern(keywords)
    phrases = list(phrases or [])

    focus: tuple[int, int] | None = None
    if focus_span is not None:
        try:
            start = max(0, int(focus_span[0]))
            end = min(len(raw), int(focus_span[1]))
        except (TypeError, ValueError, IndexError):
            start = end = 0
        if start < end:
            esc_start = len(html.escape(raw[:start]))
            focus = (esc_start, esc_start + len(html.escape(raw[start:end])))

    if focus is None:
        return f"{open_tag}{_mark_highlighted_segment(body, matcher, phrases)}</div>"

    fs, fe = focus
    before = _mark_highlighted_segment(body[:fs], matcher, phrases)
    middle = _mark_highlighted_segment(body[fs:fe], matcher, phrases)
    after = _mark_highlighted_segment(body[fe:], matcher, phrases)
    return (
        f"{open_tag}{before}"
        f'<div class="field-focus">{middle}</div>'
        f"{after}</div>"
    )


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
    # (doc, match_debug, dev_mode, short_name, sources) — sources: ["local", "ckm"]
    render_result_name: Callable[[dict, dict, bool, str, list[str]], None]
    # (doc, match_debug, short_name, source, ctx, prompts, group_size)
    render_result_details: Callable[[dict, dict, str, str, TabContext, dict, int], None]
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
    # Ключ схлопывания дублей local+CKM (None → не схлопывать).
    group_key: Callable[[dict], str] | None = None


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
    return f"{label}  \n`{date_s}`"


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


def render_source_badges(sources) -> None:
    """Цветные метки источников для одного результата (local, CKM)."""
    keys: list[str] = []
    for source in sources or ["local"]:
        key = str(source or "local").strip().lower()
        if key and key not in keys:
            keys.append(key)
    spans = []
    for key in keys:
        label = {"ckm": "CKM", "local": "local"}.get(key, key.upper())
        spans.append(f'<span class="src-badge src-badge-{key}">{label}</span>')
    st.markdown(" ".join(spans), unsafe_allow_html=True)


def render_source_badge(source: str) -> None:
    """Цветная метка источника (local/CKM) вместо префикса в копируемом id."""
    render_source_badges([source])


def _short_file_name(doc: dict) -> str:
    fn = os.path.basename(str(doc.get("file", "unknown")))
    if fn.lower().endswith(".adl"):
        fn = fn[:-4]
    return fn


def _group_result_indices(
    indices: list[int],
    docs: list[dict],
    group_key,
) -> list[list[int]]:
    """Схлопывает результаты с одинаковым ключом из разных источников.

    ``indices`` уже упорядочены по релевантности, поэтому группа встаёт на
    место первого (лучшего) представителя. Документы одного и того же источника
    не сливаются: одинаковое имя в local дважды — это два разных результата.
    """
    if group_key is None:
        return [[int(i)] for i in indices]

    groups: list[list[int]] = []
    by_key: dict[str, list[int]] = {}
    for idx in indices:
        idx = int(idx)
        doc = docs[idx]
        key = str(group_key(doc) or "")
        if not key:
            groups.append([idx])
            continue
        source = str(doc.get("source") or "local")
        for gi in by_key.get(key, []):
            group_sources = {str(docs[i].get("source") or "local") for i in groups[gi]}
            if source not in group_sources:
                groups[gi].append(idx)
                break
        else:
            groups.append([idx])
            by_key.setdefault(key, []).append(len(groups) - 1)
    return groups


def _group_pick_key(prefix: str, docs: list[dict], indices: list[int]) -> str:
    """Стабильный ключ чекбокса для группы результатов."""
    ident = "|".join(
        sorted(
            f"{docs[i].get('source') or 'local'}:{docs[i].get('file') or i}"
            for i in indices
        )
    )
    return f"{prefix}_pick_group_{ident}"


def _dev_penalty_reason(md: dict) -> str:
    if md.get("misuse_hit"):
        return "misuse"
    if md.get("weak_hit"):
        return "слабое совпадение"
    return "понижение"


def _dev_score_line(
    score_s,
    sim_s,
    bonus_s,
    penalty_s,
    md: dict,
    keyword_bonus: float,
    zscore: float | None = None,
    phrase_bonus: float = 0.0,
) -> str:
    """Расшифровка score: из чего он сложился, и z-оценка внутри источника."""
    score = float(score_s) if score_s is not None else 0.0
    sim = float(sim_s) if sim_s is not None else 0.0
    bonus = float(bonus_s) if bonus_s is not None else 0.0
    phrase = float(md.get("phrase") or 0.0)
    penalty = float(penalty_s) if penalty_s is not None else 1.0
    reason = _dev_penalty_reason(md)

    if not md.get("lexical_enabled", True):
        base = f"схожесть {sim:.2f}"
        if penalty < 0.999:
            base += f" × штраф {penalty:.2f} ({reason})"
        base += " · лексика отключена (другой язык)"
    else:
        base = f"схожесть {sim:.2f} + {float(keyword_bonus):.2f} × лексика {bonus:.2f}"
        if phrase > 0:
            base += f" + {float(phrase_bonus):.2f} × фраза {phrase:.2f}"
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
    phrase_bonus: float = 0.0,
    zscore: float | None = None,
    low_confidence: bool = False,
) -> None:
    """Компактная dev-сводка: как получился score, где совпали слова, предупреждение."""
    st.caption(
        _dev_score_line(
            score, sim, bonus, penalty, md, keyword_bonus, zscore, phrase_bonus
        )
    )
    matches = _dev_matches_line(md)
    if matches:
        st.caption("Совпало: " + matches)
    elif md.get("lexical_enabled", True):
        st.caption("Точных совпадений нет — только по смыслу.")
    if low_confidence:
        st.caption("Возможно, нерелевантно.")


def dev_legend(keyword_bonus: float, phrase_bonus: float = 0.0) -> str:
    """Одна строка-легенда для dev-режима."""
    return (
        f"dev: score = (схожесть + {float(keyword_bonus):.2f} × лексика"
        f" + {float(phrase_bonus):.2f} × словосочетание) × штраф. "
        "Схожесть — смысловая близость, лексика — совпадения слов, "
        "словосочетание — найденная фраза, "
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


def _history_with_query(history, query: str) -> list[str]:
    """Добавляет запрос в начало истории: без дублей, максимум 10."""
    q = (query or "").strip()
    if not q:
        return list(history or [])[:10]
    out = [h for h in (history or []) if str(h).casefold() != q.casefold()]
    out.insert(0, q)
    return out[:10]


def _remember_query(prefix: str, query: str) -> None:
    key = f"{prefix}_query_history"
    st.session_state[key] = _history_with_query(
        st.session_state.get(key) or [], query
    )


def _expand_base_vocab() -> set[str]:
    """Словарь токенов обоих индексов: архетипы + поля."""
    vocab: set[str] = set()
    for key in ("arch_token_vocab", "field_token_vocab"):
        value = st.session_state.get(key)
        if isinstance(value, set):
            vocab.update(value)
    return vocab


def _expand_term_in_base(term: str, vocab: set[str]) -> bool:
    """Есть ли все значимые слова термина в словаре базы."""
    normalized = unicodedata.normalize("NFKC", str(term or ""))
    tokens = [
        t
        for t in re.findall(r"\w+", normalized.lower())
        if len(t) >= 2 and t not in _SHORT_STOPWORDS
    ]
    if not tokens:
        return False
    for token in tokens:
        if len(token) >= 6:
            prefix = _stem_prefix(token)
            if not any(v.startswith(prefix) for v in vocab):
                return False
        elif token not in vocab:
            return False
    return True


def _filter_expand_terms_in_base(terms, vocab: set[str]) -> list[str]:
    if not vocab:
        return list(terms or [])
    return [t for t in (terms or []) if _expand_term_in_base(t, vocab)]


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
    st.session_state.setdefault(f"{prefix}_expand_open", False)
    st.session_state.setdefault(f"{prefix}_query_history", [])
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
            query_key = f"{prefix}_query"
            # Восстанавливаем текст после переключения вкладки: форма очищает
            # состояние виджета, но сохранённый запрос остаётся в _query_saved.
            if query_key not in st.session_state:
                st.session_state[query_key] = st.session_state.get(
                    f"{prefix}_query_saved", ""
                )
            query = st.text_input(
                "Запрос",
                help=spec.query_help,
                key=query_key,
                autocomplete="off",
            )
            query = query or ""
        with q2:
            run_search = st.form_submit_button("Искать", use_container_width=True)

        # Подписи расширенного запроса и перевода CKM заполняются после поиска,
        # но визуально стоят сразу под полем запроса.
        expanded_query_slot = st.empty()
        translation_slot = st.empty()

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
            expanded=bool(
                st.session_state.get(f"{prefix}_expand_open", False)
                or st.session_state.get(f"{prefix}_expand_results", False)
            ),
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

            terms = _filter_expand_terms_in_base(terms, _expand_base_vocab())

            # По умолчанию ничего не отмечено: пользователь выбирает варианты сам.
            st.session_state.setdefault(f"{prefix}_expand_selected", [])

            if find_expand:
                st.session_state[f"{prefix}_expand_open"] = True
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
                        raw_terms = suggest_query_expansions(
                            query,
                            ctx.llm_aux_provider,
                            system=parts["query_expand"],
                            user_template=parts["query_expand_user"],
                            max_terms=_EXPAND_TERMS_LIMIT,
                        )
                        terms_new = _filter_expand_terms_in_base(
                            list(raw_terms), _expand_base_vocab()
                        )
                        st.session_state[f"{prefix}_expand_terms"] = terms_new
                        st.session_state[f"{prefix}_expand_query"] = query or ""
                        if not raw_terms:
                            st.session_state[f"{prefix}_expand_selected"] = []
                            st.session_state[f"{prefix}_expand_last_kind"] = "info"
                            st.session_state[f"{prefix}_expand_last_msg"] = "ИИ не предложил терминов."
                        elif not terms_new:
                            st.session_state[f"{prefix}_expand_selected"] = []
                            st.session_state[f"{prefix}_expand_last_kind"] = "info"
                            st.session_state[f"{prefix}_expand_last_msg"] = (
                                "Подходящих в базе вариантов не найдено."
                            )
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

                # Перерисовать сразу с открытым expander: иначе он может
                # остаться в старом (свёрнутом) состоянии до следующего клика.
                st.session_state[f"{prefix}_query_saved"] = query or ""
                st.rerun()

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

    query = st.session_state.get(f"{prefix}_query") or ""
    if run_search or find_expand:
        st.session_state[f"{prefix}_query_saved"] = query
    if run_search and query.strip():
        _remember_query(prefix, query)

    history = list(st.session_state.get(f"{prefix}_query_history") or [])[:10]
    history_json = json.dumps(history, ensure_ascii=False).replace("</", "<\\/")
    if "_query_history_token" not in st.session_state:
        st.session_state["_query_history_token"] = uuid.uuid4().hex
    st.session_state["_query_history_seq"] = int(
        st.session_state.get("_query_history_seq") or 0
    ) + 1
    history_seq = st.session_state["_query_history_seq"]
    st.iframe(
        _QUERY_HISTORY_JS.replace("__PREFIX__", prefix)
        .replace("__TOKEN__", st.session_state["_query_history_token"])
        .replace("__SEQ__", str(history_seq))
        .replace("__VALUES__", history_json)
        + f"\n<!-- {prefix}-{history_seq} -->",
        height=1,
    )

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
            st.session_state[f"{prefix}_result_groups"] = []
            st.session_state[f"{prefix}_visible_count"] = 0
            st.session_state[f"{prefix}_match_debug"] = {}
            st.session_state[f"{prefix}_scores"] = []
            st.session_state[f"{prefix}_zscore_by_idx"] = {}
            # Не показываем подсказки/перевод от прошлого поиска: они относятся
            # к другому набору документов и выглядят как «найдено, но не выдано».
            st.session_state[f"{prefix}_suggestions"] = []
            st.session_state[f"{prefix}_translation_note"] = ""
            st.session_state[f"{prefix}_query_en_used"] = ""
            st.session_state[f"{prefix}_query_en"] = ""
            st.session_state[f"{prefix}_expanded_query"] = ""
            st.session_state[f"{prefix}_empty_kind"] = (
                "filtered" if (filter_selection or need_source_filter) else "no_docs"
            )
            st.session_state[f"{prefix}_searched"] = True
        else:
            st.session_state[f"{prefix}_empty_kind"] = ""
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
                            # Спиннер показываем там, где потом появится перевод —
                            # в слоте под строкой запроса.
                            with translation_slot:
                                with st.spinner(
                                    "Перевод запроса на английский (для CKM)..."
                                ):
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
                    phrase_bonus=ctx.phrase_bonus,
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
                strong_lex_min=_STRONG_LEX_MIN,
                strong_lex_sim_floor=_STRONG_LEX_SIM_FLOOR,
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
                    "(другой язык {4:.2f}, пометка ниже {5:.2f}; "
                    "сильная лексика ≥{6:.2f} → {7:.2f})".format(
                        float(sel_info.get("gap_max") or 0.0),
                        float(sel_info.get("band") or 0.0),
                        float(sel_info.get("sim_floor") or 0.0),
                        float(sel_info.get("sim_floor_pure") or 0.0),
                        cross_floor,
                        float(ctx.sim_low_confidence),
                        _STRONG_LEX_MIN,
                        _STRONG_LEX_SIM_FLOOR,
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
            result_groups = _group_result_indices(top_indices, docs, spec.group_key)
            picked_groups = _group_result_indices(
                [back[i] for i in top_local], docs, spec.group_key
            )
            st.session_state[f"{prefix}_result_groups"] = result_groups
            # Сначала показываем только первичный топ, как и до группировки;
            # остальные кандидаты остаются за «Показать ещё».
            st.session_state[f"{prefix}_visible_count"] = min(
                len(picked_groups) if picked_groups else len(result_groups),
                _SHOW_MORE_LIMIT,
            )
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

    translation_note = st.session_state.get(f"{prefix}_translation_note") or ""
    translation_q = (st.session_state.get(f"{prefix}_query_en_used") or "").strip()
    if st.session_state.get(f"{prefix}_searched") and query.strip():
        used_q = (st.session_state.get(f"{prefix}_expanded_query") or "").strip()
        # Строку показываем только в dev-режиме: в обычном она лишняя.
        if used_q and used_q != (query or "").strip() and ctx.dev_mode:
            with expanded_query_slot.container():
                st.caption(f"Расширенный запрос: «{used_q}»")

        translation_msg = ""
        if translation_note == "translated" and translation_q:
            translation_msg = f"Перевод для CKM: «{translation_q}»"
        elif translation_note == "not_needed" and translation_q:
            translation_msg = f"CKM: «{translation_q}» (перевод не нужен)"
        elif translation_note == "no_llm":
            translation_msg = "CKM: перевод недоступен — поиск по смыслу"
        elif translation_note == "failed":
            translation_msg = "CKM: перевод не удался — поиск по смыслу"
        if translation_msg:
            with translation_slot.container():
                st.caption(translation_msg)

    all_indices = st.session_state.get(f"{prefix}_top_indices") or []
    all_groups = st.session_state.get(f"{prefix}_result_groups") or []
    if not all_groups and all_indices:
        # Старая сессия без сохранённых групп либо негруппируемая вкладка.
        all_groups = _group_result_indices(all_indices, docs or [], spec.group_key)
    try:
        visible_count = int(st.session_state.get(f"{prefix}_visible_count") or 0)
    except (TypeError, ValueError):
        visible_count = 0
    if visible_count <= 0:
        visible_count = len(all_groups)
    visible_count = max(0, min(visible_count, len(all_groups), _SHOW_MORE_LIMIT))
    visible_groups = all_groups[:visible_count]
    top_indices = [i for group in visible_groups for i in group]
    match_debug = st.session_state.get(f"{prefix}_match_debug") or {}
    score_by_idx = st.session_state.get(f"{prefix}_score_by_idx") or {}
    sim_by_idx = st.session_state.get(f"{prefix}_similarity_by_idx") or {}
    bonus_by_idx = st.session_state.get(f"{prefix}_bonus_by_idx") or {}
    penalty_by_idx = st.session_state.get(f"{prefix}_penalty_by_idx") or {}
    zscore_by_idx = st.session_state.get(f"{prefix}_zscore_by_idx") or {}

    st.markdown(CHAT_SIDEBAR_CSS, unsafe_allow_html=True)

    with st.container():
        # Результаты поиска занимают всю ширину: чат живёт в сайдбаре.
        if st.session_state.get(f"{prefix}_searched") and query.strip() and not all_indices:
            empty_kind = st.session_state.get(f"{prefix}_empty_kind") or ""
            if empty_kind == "filtered":
                st.info(
                    "В выбранном фильтре ничего не найдено. "
                    "Измените фильтр или запрос и запустите поиск снова."
                )
            else:
                st.info("Ничего не найдено: нет релевантных результатов.")
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
                        label = f"· {spec.result_label(docs[idx])}"
                        if ctx.dev_mode:
                            label += f" (схожесть {float(s.get('sim') or 0):.2f})"
                        st.caption(label)

        selected_idx: list[int] = []
        try:
            persisted_selected = {
                int(i) for i in (st.session_state.get(f"{prefix}_selected_persisted") or [])
            }
        except (TypeError, ValueError):
            persisted_selected = set()
        if top_indices:
            if len(all_groups) > len(visible_groups):
                st.write(
                    f"{spec.result_count_label} {len(visible_groups)} из {len(all_groups)}"
                )
            else:
                st.write(f"{spec.result_count_label} {len(visible_groups)}")

            if ctx.dev_mode:
                st.caption(dev_legend(ctx.keyword_bonus, ctx.phrase_bonus))

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

            # "Выбрать все" для текущей выдачи: один ключ на группу результатов.
            if spec.group_key is not None:
                pick_keys = [_group_pick_key(prefix, docs, group) for group in visible_groups]
            else:
                pick_keys = [
                    spec.make_pick_key(docs[group[0]], group[0]) for group in visible_groups
                ]
            st.session_state[f"{prefix}_pick_keys"] = pick_keys

            def _group_checked(key: str, group: list[int]) -> bool:
                if key in st.session_state:
                    return bool(st.session_state.get(key))
                return all(i in persisted_selected for i in group)

            all_selected = (
                all(
                    _group_checked(key, group)
                    for key, group in zip(pick_keys, visible_groups)
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

            for n, group in enumerate(visible_groups, start=1):
                primary_idx = group[0]
                doc = docs[primary_idx]
                key = pick_keys[n - 1]
                sources = [docs[i].get("source") or "local" for i in group]
                md = match_debug.get(primary_idx) or {}
                fn = _short_file_name(doc)
                with st.container(border=True):
                    c_pick, c_name = st.columns([1, 13], vertical_alignment="center")
                    with c_pick:
                        picked = st.checkbox(
                            f"#{n}",
                            key=key,
                            value=_group_checked(key, group),
                            on_change=_sync_pick_all,
                            args=(prefix,),
                        )
                    with c_name:
                        spec.render_result_name(doc, md, ctx.dev_mode, fn, sources)
                    if ctx.dev_mode:
                        for member_idx in group:
                            if len(group) > 1:
                                member_src = docs[member_idx].get("source") or "local"
                                st.caption(f"Источник: {member_src}")
                            render_dev_result_info(
                                score=score_by_idx.get(member_idx),
                                sim=sim_by_idx.get(member_idx),
                                bonus=bonus_by_idx.get(member_idx),
                                penalty=penalty_by_idx.get(member_idx),
                                md=match_debug.get(member_idx) or {},
                                keyword_bonus=ctx.keyword_bonus,
                                phrase_bonus=ctx.phrase_bonus,
                                zscore=zscore_by_idx.get(member_idx),
                                low_confidence=member_idx in weak_set,
                            )
                    elif primary_idx in weak_set:
                        st.caption("Возможно, нерелевантно.")
                    if picked:
                        selected_idx.extend(group)
                    for member_idx in group:
                        member_doc = docs[member_idx]
                        spec.render_result_details(
                            member_doc,
                            match_debug.get(member_idx) or {},
                            _short_file_name(member_doc),
                            member_doc.get("source") or "local",
                            ctx,
                            parts,
                            len(group),
                        )

            st.session_state[f"{prefix}_selected"] = selected_idx
            st.session_state[f"{prefix}_selected_persisted"] = selected_idx

            remaining = len(all_groups) - len(visible_groups)
            if remaining > 0:
                step = min(_SHOW_MORE_STEP, remaining)

                def _show_more(prefix=prefix) -> None:
                    # «Выбрать все» относится к видимым результатам: после догрузки
                    # снимаем его, уже отмеченные результаты остаются выбранными.
                    st.session_state[f"{prefix}_visible_count"] = min(
                        int(st.session_state.get(f"{prefix}_visible_count") or 0) + _SHOW_MORE_STEP,
                        len(st.session_state.get(f"{prefix}_result_groups") or []),
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
        # Выбор действует только в текущей сессии и НЕ меняет значение
        # по умолчанию из config/llm.json (chat_provider).
        st.session_state["chat_provider_saved"] = kind
        st.session_state["api_kind"] = kind

    with st.sidebar:
        with st.container(key=f"{prefix}_chat_panel"):
            st.markdown('<div class="chat-panel-title">Чат с ИИ</div>', unsafe_allow_html=True)

            has_history = bool(get_chat_messages(prefix))
            current_kind = selected_chat_provider(ctx.cfg)
            provider_labels = {"gigachat": "ГигаЧат", "corp_async": "МЭШ.ИИ.Хаб"}
            if ctx.cfg.llm_mode != "ollama" and current_kind in provider_labels:
                if st.session_state.get("chat_provider_choice") not in CHAT_PROVIDER_OPTIONS:
                    # Виджет мог быть вычищен, когда радио не отрисовывалось:
                    # восстанавливаем последний выбор пользователя.
                    st.session_state["chat_provider_choice"] = (
                        st.session_state.get("chat_provider_saved") or current_kind
                    )
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