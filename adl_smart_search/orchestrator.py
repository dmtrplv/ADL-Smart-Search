from __future__ import annotations

import logging
from typing import Any

from .llm import chat_with_continuation
from .prompts import (
    PromptConfigError,
    apply_placeholders,
    compose_system_prompt,
    prompt_text,
)

logger = logging.getLogger(__name__)


def make_llm_dialog(*, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    system_prompt = (system_prompt or "").strip()
    if not system_prompt:
        raise PromptConfigError(
            "Промпты не загружены: проверьте config/prompts.json."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt or ""},
    ]


def _history_turns(dialog_messages: list[dict]) -> list[dict[str, str]]:
    """Возвращает историю диалога без system-сообщений."""
    return [
        {"role": str(message.get("role", "")), "content": str(message.get("content", "") or "")}
        for message in (dialog_messages or [])
        if message.get("role") in ("user", "assistant")
    ]


def build_followup_dialog(
    *,
    prompts: dict[str, str],
    pinned_context: str,
    rag_query: str,
    dialog_messages: list[dict],
    user_message: str,
    max_history_messages: int = 6,
) -> list[dict[str, str]]:
    """Собирает запрос на уточнение в той же форме, что и первый.

    Актуальный system-промпт + сохранённая история + новый вопрос. Первая
    пара «контекст+вопрос → ответ» сохраняется всегда, чтобы модель видела,
    что уже было предложено. Ограничение применяется только к последующим
    репликам, чтобы не раздувать запрос.
    """
    sys_content = compose_system_prompt(prompts, query=rag_query)
    turns = _history_turns(dialog_messages)

    if not turns:
        # Совместимость со старыми сессиями: истории нет, но контекст есть.
        user_content = apply_placeholders(
            prompt_text(prompts, "answer_user_prompt"),
            context=pinned_context or "",
            question=(user_message or "").strip(),
        )
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

    head = turns[:2]
    tail = turns[2:]
    if max_history_messages > 0 and len(tail) > max_history_messages:
        tail = tail[-max_history_messages:]
    # Не начинаем обрезанную историю с ответа ассистента: после head это
    # дало бы два assistant-сообщения подряд.
    while tail and tail[0]["role"] != "user":
        tail = tail[1:]

    msgs: list[dict[str, str]] = [{"role": "system", "content": sys_content}]
    msgs.extend(head)
    msgs.extend(tail)
    msgs.append({"role": "user", "content": (user_message or "").strip()})
    return msgs


def handle_followup(
    *,
    question: str,
    rag_query: str,
    pinned_context: str,
    dialog_messages: list[dict],
    prompts: dict[str, str],
    llm_provider: Any,
) -> str:
    """Обрабатывает уточняющий вопрос: единый диалог → LLM."""
    try:
        llm_provider.usage_total = {}
    except Exception:
        pass

    dialog = build_followup_dialog(
        prompts=prompts,
        pinned_context=pinned_context,
        rag_query=rag_query,
        dialog_messages=dialog_messages,
        user_message=question,
    )
    return chat_with_continuation(
        llm_provider,
        dialog,
        continuation_prompt=str(prompts.get("continuation_prompt") or ""),
    )
