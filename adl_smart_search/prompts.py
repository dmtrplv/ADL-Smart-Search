"""Загрузка и сборка системных промптов из ``config/prompts.json``.

Тексты промптов не хранятся в коде: единственный источник истины —
``<config_dir>/prompts.json`` в приватном репозитории с конфигами.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

PROMPT_FILENAME = "prompts.json"

# Системный промпт редактируется отдельным большим полем, остальные части —
# отдельными блоками в dev-панели.
PROMPT_KEYS = (
    "system_prompt",
    "default_question_archetypes",
    "default_question_fields",
    "answer_user_prompt",
    "continuation_prompt",
    "query_expand",
    "query_expand_user",
    "translate",
    "translate_user",
)
PROMPT_PART_KEYS = (
    "default_question_archetypes",
    "default_question_fields",
    "answer_user_prompt",
    "continuation_prompt",
    "query_expand",
    "query_expand_user",
    "translate",
    "translate_user",
)

PROMPT_LABELS: dict[str, str] = {
    "default_question_archetypes": (
        "Промпт первого сообщения для архетипов (используйте {query})"
    ),
    "default_question_fields": (
        "Промпт первого сообщения для полей (используйте {query})"
    ),
    "answer_user_prompt": (
        "Сообщение пользователя для первого ответа "
        "(используйте {context} и {question})"
    ),
    "continuation_prompt": "Промпт автопродолжения оборванного ответа",
    "query_expand": "Расширение запроса (системный промпт)",
    "query_expand_user": (
        "Расширение запроса: сообщение пользователя "
        "(используйте {query}; {target_lang} — опционально)"
    ),
    "translate": "Перевод запроса для CKM (системный промпт)",
    "translate_user": (
        "Перевод запроса: сообщение пользователя (используйте {query})"
    ),
}


def apply_query_placeholder(text: str, query: str | None) -> str:
    """Подставляет поисковый запрос вместо ``{query}``.

    Если ``query is None``, текст возвращается без изменений. Остальные
    фигурные скобки не интерпретируются.
    """
    if query is None:
        return text
    return str(text or "").replace("{query}", (query or "").strip())


def apply_placeholders(text: str, **values: str) -> str:
    """Подставляет значения вида ``{name}``, не трогая остальные скобки.

    Один проход вместо ``str.format``: в промптах могут встречаться JSON-примеры
    и другие фигурные скобки, а подставленный текст не должен заменяться повторно.
    """
    def _sub(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        value = values[name]
        return str(value if value is not None else "")

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, str(text or ""))


def prompt_text(prompts: dict, key: str) -> str:
    """Возвращает обязательный текст промпта или бросает PromptConfigError."""
    text = str((prompts or {}).get(key) or "").strip()
    if not text:
        raise PromptConfigError(
            f"В файле промптов отсутствует или пуст ключ «{key}»: "
            "проверьте config/prompts.json."
        )
    return text


def format_default_question(prompts: dict, key: str, query: str) -> str:
    """Подставляет поисковый запрос в редактируемый вопрос по умолчанию."""
    template = str((prompts or {}).get(key) or "").strip()
    if not template:
        return ""
    return apply_query_placeholder(template, query).strip()


class PromptConfigError(RuntimeError):
    """Проблема с внешним файлом промптов."""

    @property
    def user_message(self) -> str:
        """Текст для пользователя: показывается при обращении к LLM."""
        return str(self)


def prompts_path(config_dir: str) -> str:
    """Путь к файлу промптов.

    ``ADL_PROMPTS_FILE`` позволяет указать файл явно (например, в тестах или
    при нестандартном расположении конфигов).
    """
    override = (os.getenv("ADL_PROMPTS_FILE") or "").strip()
    if override:
        return override
    return os.path.join(config_dir or ".", PROMPT_FILENAME)


def load_prompts(config_dir: str) -> dict[str, str]:
    """Читает и валидирует ``config/prompts.json``."""
    path = prompts_path(config_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise PromptConfigError(
            f"Не найден файл промптов: {path}. "
            "Администратор должен добавить prompts.json в репозиторий с конфигами."
        ) from e
    except Exception as e:
        raise PromptConfigError(f"Не удалось прочитать файл промптов {path}: {e}") from e
    if not isinstance(data, dict):
        raise PromptConfigError(f"Файл промптов {path} должен содержать JSON-объект.")
    missing = [key for key in PROMPT_KEYS if not isinstance(data.get(key), str) or not data[key].strip()]
    if missing:
        raise PromptConfigError(
            f"В файле промптов {path} отсутствуют или пусты обязательные ключи: "
            + ", ".join(missing)
        )
    return {key: data[key] for key in PROMPT_KEYS}


def save_prompts(config_dir: str, prompts: dict) -> bool:
    """Записывает полный набор промптов в ``config/prompts.json``."""
    data = {key: str((prompts or {}).get(key) or "") for key in PROMPT_KEYS}
    missing = [key for key in PROMPT_KEYS if not data[key].strip()]
    if missing:
        raise PromptConfigError(
            "Нельзя сохранить пустые обязательные промпты: " + ", ".join(missing)
        )
    path = prompts_path(config_dir)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
        return True
    except Exception:
        logger.exception("save_prompts: cannot write %s", path)
        return False


def compose_system_prompt(
    prompts: dict,
    *,
    query: str | None = None,
) -> str:
    """Собирает системный промпт и подставляет ``{query}``, если он задан."""
    text = prompt_text(prompts, "system_prompt")
    return apply_query_placeholder(text, query)
