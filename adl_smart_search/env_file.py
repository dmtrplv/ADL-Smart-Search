"""Чтение `.env` рядом с приложением (или exe) без внешних зависимостей.

Зачем: в пользовательском режиме UI-настроек LLM нет, все параметры берутся из
окружения. Чтобы не прописывать переменные Windows на каждом ПК, приложение
читает файл `.env`, который лежит рядом с exe (или в корне проекта при запуске
из Python).

Формат — по одной паре на строку:

    # комментарий
    LLM_MODE=api
    export LLM_API_KEY="ключ с пробелами и #"
    OLLAMA_BASE_URL='http://127.0.0.1:11434'

Правила:
- ключ: буквы/цифры/подчёркивание, начинается с буквы или `_`;
- значение можно взять в одинарные (как есть) или двойные кавычки
  (внутри двойных распознаются только `\\"` и `\\\\`, поэтому пути Windows
  можно писать как обычно);
- у значения без кавычек отрезается хвостовой комментарий ` #...`;
- уже заданные в системе переменные **не переопределяются** (система важнее файла);
- многострочные значения не поддерживаются.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_FILE_NAME = ".env"
ENV_FILE_OVERRIDE = "ADL_ENV_FILE"

_KEY_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_INLINE_COMMENT_RE = re.compile(r"\s#")
_DOUBLE_QUOTE_ESCAPES = {
    '"': '"',
    "\\": "\\",
}


def _parse_double_quoted(raw: str) -> str:
    out: list[str] = []
    i = 1
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            out.append(_DOUBLE_QUOTE_ESCAPES.get(nxt, "\\" + nxt))
            i += 2
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith('"'):
        return _parse_double_quoted(value)
    if value.startswith("'"):
        end = value.find("'", 1)
        return value[1:end] if end != -1 else value[1:]
    comment = _INLINE_COMMENT_RE.search(value)
    if comment:
        value = value[: comment.start()]
    return value.strip()


def parse_env_text(text: str) -> dict[str, str]:
    """Разбирает содержимое `.env` в словарь. Некорректные строки пропускает."""
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _KEY_RE.match(line)
        if not match:
            logger.warning("env: строка %d пропущена (ожидается KEY=VALUE)", line_no)
            continue
        key, raw_value = match.group(1), match.group(2)
        values[key] = _parse_value(raw_value)
    return values


def load_env_file(path: str | Path, *, override: bool = False) -> list[str]:
    """Загружает `.env` в `os.environ`.

    Возвращает список ключей, которые были применены. Существующие переменные
    окружения сохраняются, если `override=False` (по умолчанию).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Старый Блокнот мог сохранить файл в cp1251 — читаем максимально терпимо.
        text = p.read_text(encoding="cp1251", errors="replace")
    except OSError as e:
        logger.warning("env: не удалось прочитать %s: %s", p, e)
        return []

    values = parse_env_text(text)
    applied: list[str] = []
    for key, value in values.items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied.append(key)

    if applied:
        logger.info("env: загружено %d из %d переменных из %s", len(applied), len(values), p)
    else:
        logger.debug("env: %s прочитан, новых переменных нет", p)
    return applied


def load_app_envs(paths) -> Path | None:
    """Загружает несколько `.env` в порядке приоритета.

    Используется новой раскладкой: локальный `.env` → `<repo>/config/.env` →
    старый `.env` рядом с exe. Системные переменные всегда важнее файлов;
    внутри файлов побеждает первый найденный (локальный).
    """
    explicit = (os.environ.get(ENV_FILE_OVERRIDE) or "").strip()
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.is_file():
            load_env_file(explicit_path)
            return explicit_path.resolve()
        logger.warning("env: %s указывает на отсутствующий файл: %s", ENV_FILE_OVERRIDE, explicit_path)

    loaded: list[Path] = []
    for value in paths or ():
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            continue
        load_env_file(path)
        loaded.append(path.resolve())
    if loaded:
        os.environ.setdefault(ENV_FILE_OVERRIDE, str(loaded[0]))
        return loaded[0]
    return None


def load_app_env(app_root: str | Path | None = None) -> Path | None:
    """Находит и загружает `.env` для приложения.

    Порядок поиска:
    1. путь из `ADL_ENV_FILE`, если задан;
    2. `<app_root>/.env` (рядом с exe или в корне проекта);
    3. `./.env` в текущей рабочей папке.

    Возвращает путь к загруженному файлу или `None`.
    """
    candidates: list[Path] = []
    explicit = (os.environ.get(ENV_FILE_OVERRIDE) or "").strip()
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.is_file():
            candidates.append(explicit_path)
        else:
            logger.warning("env: %s указывает на отсутствующий файл: %s", ENV_FILE_OVERRIDE, explicit_path)
    if not candidates:
        if app_root:
            candidates.append(Path(app_root).expanduser() / ENV_FILE_NAME)
        cwd_file = Path.cwd() / ENV_FILE_NAME
        if cwd_file not in candidates:
            candidates.append(cwd_file)

    for path in candidates:
        if path.is_file():
            load_env_file(path)
            resolved = path.resolve()
            os.environ.setdefault(ENV_FILE_OVERRIDE, str(resolved))
            return resolved
    return None
