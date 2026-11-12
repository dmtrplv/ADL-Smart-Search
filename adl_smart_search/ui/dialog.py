"""Чат с ИИ, общий для вкладок «Архетипы» и «Поля».

История для показа пользователю хранится отдельно от API-диалога:
``{prefix}_chat_messages`` — пары «вопрос → ответ» без служебного контекста,
``{prefix}_dialog_messages`` — полный диалог (system + контекст + ответы), который
передаётся в LLM при уточнениях.
"""

import logging
import re
from contextlib import nullcontext

import streamlit as st

from adl_smart_search.orchestrator import handle_followup
from adl_smart_search.prompts import PromptConfigError
from adl_smart_search.rag import make_rag_selection_key

logger = logging.getLogger(__name__)

_INITIAL_QUESTION_MARKER = "ВОПРОС:"

_AT_CODE_RE = re.compile(r"\bat\d{4}\b|\bat\d+\.\d+\b", re.IGNORECASE)
_ARCHETYPE_ID_RE = re.compile(
    r"\bopenEHR-EHR-[A-Z_]+\.[A-Za-z0-9_]+(?:[-.][A-Za-z0-9_]+)*",
    re.IGNORECASE,
)
_CODE_SPAN_RE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)


def highlight_refs_markdown(text: str) -> str:
    """Оборачивает archetype id и at-коды в backticks для подсветки в Markdown.

    Уже существующие inline/fenced code-блоки не трогаем.
    """
    if not text:
        return text

    def _wrap(segment: str) -> str:
        segment = _ARCHETYPE_ID_RE.sub(lambda m: f"`{m.group(0)}`", segment)
        return _AT_CODE_RE.sub(lambda m: f"`{m.group(0)}`", segment)

    out: list[str] = []
    pos = 0
    for match in _CODE_SPAN_RE.finditer(text):
        out.append(_wrap(text[pos : match.start()]))
        out.append(match.group(0))
        pos = match.end()
    out.append(_wrap(text[pos:]))
    return "".join(out)


def chat_messages_key(prefix: str) -> str:
    return f"{prefix}_chat_messages"


def rag_selection_stale(prefix: str, current_indices: list) -> bool:
    stored = st.session_state.get(f"{prefix}_rag_selection_key", "")
    if not stored:
        return False
    return stored != make_rag_selection_key(current_indices)


def clear_rag_session(prefix: str) -> None:
    st.session_state[f"{prefix}_rag_context"] = ""
    st.session_state[f"{prefix}_rag_query"] = ""
    st.session_state[f"{prefix}_rag_selection_key"] = ""


def _split_initial_prompt(content: str) -> str:
    """Достаёт из «КОНТЕКСТ…ВОПРОС…» только сам вопрос (для отображения)."""
    if _INITIAL_QUESTION_MARKER in content:
        return content.split(_INITIAL_QUESTION_MARKER, 1)[1].strip()
    return content.strip()


def _derive_from_dialog(dialog: list[dict]) -> list[dict]:
    """Совместимость со старыми сессиями: собирает историю из полного диалога."""
    out: list[dict] = []
    first_user = True
    for message in dialog or []:
        role = message.get("role", "")
        content = (message.get("content") or "").strip()
        if role == "system" or not content:
            continue
        if role == "user" and first_user:
            first_user = False
            content = _split_initial_prompt(content)
        if role in ("user", "assistant"):
            out.append({"role": role, "content": content})
    return out


def get_chat_messages(prefix: str) -> list[dict]:
    messages = st.session_state.get(chat_messages_key(prefix))
    if messages:
        return messages
    derived = _derive_from_dialog(st.session_state.get(f"{prefix}_dialog_messages") or [])
    if derived:
        st.session_state[chat_messages_key(prefix)] = derived
        return derived
    if messages is None:
        st.session_state[chat_messages_key(prefix)] = []
    return []


def set_chat_messages(prefix: str, messages: list[dict]) -> None:
    st.session_state[chat_messages_key(prefix)] = list(messages)


def append_chat_message(prefix: str, role: str, content: str) -> list[dict]:
    messages = list(get_chat_messages(prefix))
    messages.append({"role": role, "content": content})
    st.session_state[chat_messages_key(prefix)] = messages
    return messages


def clear_chat_session(prefix: str) -> None:
    st.session_state[chat_messages_key(prefix)] = []
    st.session_state[f"{prefix}_dialog_messages"] = []
    st.session_state[f"{prefix}_last_usage"] = {}
    st.session_state.pop(f"{prefix}_ai_input", None)
    st.session_state.pop(f"{prefix}_ai_prefill_sig", None)
    clear_rag_session(prefix)


def store_last_usage(prefix: str, provider) -> None:
    """Сохраняет статистику токенов ответа для показа в dev-режиме."""
    usage = getattr(provider, "usage_total", None) or getattr(provider, "last_usage", None) or {}
    st.session_state[f"{prefix}_last_usage"] = dict(usage) if isinstance(usage, dict) else {}


def render_usage_caption(prefix: str) -> None:
    usage = st.session_state.get(f"{prefix}_last_usage") or {}
    if not isinstance(usage, dict) or not usage:
        return
    bits = []
    if usage.get("input_tokens") is not None:
        bits.append(f"вход {usage['input_tokens']}")
    if usage.get("output_tokens") is not None:
        bits.append(f"выход {usage['output_tokens']}")
    if usage.get("cached_tokens"):
        bits.append(f"из кэша {usage['cached_tokens']}")
    if bits:
        st.caption("Токены LLM: " + ", ".join(bits))


def _set_llm_error(e: Exception) -> None:
    # Streamlit hot-reload может перезагрузить модуль и "сломать" isinstance(e, LLMError).
    user_msg = getattr(e, "user_message", None)
    details = getattr(e, "details", "") or ""
    if user_msg:
        st.session_state["last_error"] = str(user_msg)
        st.session_state["last_error_details"] = str(details)
        logger.warning("LLM request failed: %s | %s", user_msg, (details or "")[:400])
    else:
        st.session_state["last_error"] = f"Неожиданная ошибка LLM: {e}"
        st.session_state["last_error_details"] = ""
        logger.exception("LLM request failed (unexpected): %s", e)


def render_chat_messages(prefix: str) -> None:
    """Рисует историю как чат: вопрос справа, ответ ИИ слева."""
    for position, message in enumerate(get_chat_messages(prefix)):
        role = message.get("role", "")
        content = message.get("content", "")
        if role not in ("user", "assistant") or not content:
            continue
        try:
            with st.container(key=f"{prefix}_chatmsg_{position}_{role}"):
                with st.chat_message(role):
                    st.markdown(
                        highlight_refs_markdown(content) if role == "assistant" else content
                    )
        except Exception:
            logger.warning("render_chat_messages: failed for role=%s", role)
            st.markdown(f"**{role}**")
            st.markdown(highlight_refs_markdown(content) if role == "assistant" else content)


def render_pending_chat_message(placeholder, prefix: str, content: str) -> None:
    """Показать сообщение пользователя сразу, до ответа ИИ."""
    if placeholder is None or not content:
        return
    with placeholder.container():
        with st.container(key=f"{prefix}_chatmsg_pending_user"):
            with st.chat_message("user"):
                st.markdown(content)


def send_followup(
    *,
    prefix: str,
    question: str,
    stale_warning_text: str,
    llm_setup_msg: str,
    llm_provider,
    prompts: dict | None = None,
    status_placeholder=None,
    pending_message_placeholder=None,
) -> None:
    """Отправляет уточнение в LLM и дописывает пару сообщений в историю."""
    status_ctx = status_placeholder.container() if status_placeholder is not None else nullcontext()
    selected = st.session_state.get(f"{prefix}_selected") or []
    if rag_selection_stale(prefix, selected):
        with status_ctx:
            st.toast(stale_warning_text, icon="⚠️")
        return
    if llm_setup_msg or llm_provider is None:
        with status_ctx:
            st.warning(llm_setup_msg or "LLM не настроен: проверьте настройки LLM.")
        return

    dialog_messages = list(st.session_state.get(f"{prefix}_dialog_messages") or [])

    # Сразу показываем вопрос в истории, чтобы он не «исчезал» на время ответа.
    append_chat_message(prefix, "user", question)
    render_pending_chat_message(pending_message_placeholder, prefix, question)

    try:
        with status_ctx:
            with st.spinner("ИИ думает..."):
                answer = handle_followup(
                    question=question,
                    rag_query=st.session_state.get(f"{prefix}_rag_query", ""),
                    pinned_context=st.session_state.get(f"{prefix}_rag_context", ""),
                    dialog_messages=dialog_messages,
                    prompts=prompts or {},
                    llm_provider=llm_provider,
                )
    except PromptConfigError as e:
        with status_ctx:
            st.error(e.user_message)
        return
    store_last_usage(prefix, llm_provider)

    append_chat_message(prefix, "assistant", answer)
    st.session_state[f"{prefix}_dialog_messages"] = dialog_messages + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    st.session_state["last_error"] = ""
    st.session_state["last_error_details"] = ""
    st.rerun()


def report_llm_error(e: Exception) -> None:
    _set_llm_error(e)
