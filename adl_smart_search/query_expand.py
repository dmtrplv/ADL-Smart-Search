"""On-demand LLM query expansion (no static synonym dictionaries)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .prompts import PromptConfigError, apply_placeholders

logger = logging.getLogger(__name__)


def _required_system(system: str | None) -> str:
    """Системный промпт обязателен: текстов в коде больше нет (config/prompts.json)."""
    text = (system or "").strip()
    if not text:
        raise PromptConfigError(
            "Не задан системный промпт для запроса к LLM: проверьте config/prompts.json."
        )
    return text


def _required_user_template(template: str | None, key: str) -> str:
    text = (template or "").strip()
    if not text:
        raise PromptConfigError(
            f"Не задан промпт «{key}»: проверьте config/prompts.json."
        )
    return text


def build_query_expand_prompt(query: str, template: str) -> str:
    q = (query or "").strip()
    # {target_lang} не обязателен: базовый шаблон содержит только {query}.
    # Поддержка оставлена для администраторов, которым нужна языковая подсказка.
    target_lang = "английском" if looks_latin(q) else "русском"
    return apply_placeholders(
        _required_user_template(template, "query_expand_user"),
        query=q,
        target_lang=target_lang,
    )


def parse_query_expand_response(text: str, *, max_terms: int = 5) -> list[str]:
    """
    Разбирает ответ LLM в список терминов.
    Поддерживает JSON-массив; fallback — строки по запятым/переносам.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    # Срез markdown-ограждения ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()

    terms: list[str] = []
    try:
        # Попытка найти JSON-массив внутри текста
        start = raw.find("[")
        end = raw.rfind("]")
        candidate = raw[start : end + 1] if start >= 0 and end > start else raw
        data = json.loads(candidate)
        if isinstance(data, list):
            for x in data:
                if isinstance(x, str) and x.strip():
                    terms.append(x.strip())
                elif isinstance(x, (int, float)):
                    terms.append(str(x).strip())
        elif isinstance(data, dict):
            for key in ("terms", "synonyms", "expansions", "items"):
                v = data.get(key)
                if isinstance(v, list):
                    for x in v:
                        if isinstance(x, str) and x.strip():
                            terms.append(x.strip())
                    break
    except Exception:
        logger.debug("parse_query_expand_response: non-JSON answer, fallback to lines")
        terms = []

    if not terms:
        # Fallback: построчно / через запятую, убрать нумерацию и кавычки
        chunks: list[str] = []
        for line in re.split(r"[\n;]+", raw):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\-\*\d\.\)\]]+\s*", "", line)
            chunks.extend(re.split(r"\s*,\s*", line))
        for c in chunks:
            t = c.strip().strip("\"'`[]")
            if t and t.lower() not in {"json", "synonyms", "terms"}:
                terms.append(t)

    # Дедуп (casefold), ограничение длины
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        t = re.sub(r"\s+", " ", t).strip()
        if not t or len(t) > 80:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max(1, int(max_terms)):
            break
    return out


def apply_manual_expansions(query: str, terms: list[str] | None) -> str:
    """Склеивает исходный запрос с выбранными терминами: `query | t1, t2`."""
    q = (query or "").strip()
    if not q:
        return q
    add = [re.sub(r"\s+", " ", str(t)).strip() for t in (terms or []) if str(t).strip()]
    add = list(dict.fromkeys(add))
    if not add:
        return q
    # Не дублируем точные совпадения с исходным запросом
    q_cf = q.casefold()
    add = [t for t in add if t.casefold() != q_cf]
    if not add:
        return q
    if "|" in q:
        return q + ", " + ", ".join(add)
    return q + " | " + ", ".join(add)


def suggest_query_expansions(
    query: str,
    llm_provider: Any,
    *,
    system: str | None = None,
    user_template: str | None = None,
    max_terms: int = 5,
) -> list[str]:
    """Вызов LLM → список терминов для UI."""
    q = (query or "").strip()
    if not q:
        return []
    raw = llm_provider.generate(
        build_query_expand_prompt(q, user_template or ""),
        system=_required_system(system),
    )
    return parse_query_expand_response(raw, max_terms=max_terms)


def looks_latin(text: str) -> bool:
    """True, если в тексте нет кириллицы (там переводить в EN не нужно)."""
    t = (text or "").strip()
    return t != "" and not re.search(r"[а-яА-ЯёЁ]", t)


def split_query_by_script(text: str) -> tuple[str, str]:
    """Делит запрос на кириллическую и латинскую части.

    Нужно, чтобы переводить только русские слова, а латинские термины/at-коды
    («ИВЛ», «FiO2», «at0004») оставлять как есть.
    """
    t = (text or "").strip()
    if not t:
        return "", ""
    cyr: list[str] = []
    lat: list[str] = []
    for tok in t.split():
        if re.search(r"[а-яА-ЯёЁ]", tok):
            cyr.append(tok)
        else:
            lat.append(tok)
    return " ".join(cyr), " ".join(lat)


def build_translate_prompt(query: str, template: str) -> str:
    return apply_placeholders(
        _required_user_template(template, "translate_user"),
        query=query,
    )


def translate_query_to_en(
    query: str,
    llm_provider: Any,
    *,
    system: str | None = None,
    user_template: str | None = None,
) -> str:
    """
    Перевод поискового якоря с русского на английский (для CKM-источника).
    Кириллица отсутствует — возвращаем запрос как есть (passthrough).
    """
    q = (query or "").strip().strip('"').strip("'")
    if not q or looks_latin(q):
        return q
    if len(q) > 1200:
        q = q[:1200].rstrip()
    system_prompt = _required_system(system)
    prompt = build_translate_prompt(q, user_template or "")
    try:
        raw = llm_provider.generate(prompt, system=system_prompt)
        out = (raw or "").strip().strip('"').strip("'")
        # Обрезаем многострочный/неожиданный мусор и хвостовые знаки пунктуации
        out = re.split(r"\n+", out, maxsplit=1)[0].strip()
        out = out.rstrip(".,;: ").strip().strip('"').strip("'").strip()
        # Настоящий перевод — латиница; если LLM вернул кириллицу — считаем неудачей.
        if not out or not looks_latin(out):
            return q
        return out
    except Exception:
        logger.exception("translate_query_to_en: LLM call failed, fallback to original")
        return q


def translate_query_to_en_mixed(
    query: str,
    llm_provider: Any,
    *,
    system: str | None = None,
    user_template: str | None = None,
) -> tuple[str, str]:
    """Перевод запроса для CKM с сохранением латинской части.

    Возвращает ``(query_en, note)``:
      - ``note == "translated"`` — кириллическая часть переведена;
      - ``note == "no_llm"``     — перевод невозможен (нет LLM);
      - ``note == "failed"``     — LLM не дал корректный перевод;
      - ``note == "not_needed"`` — переводить нечего (запрос без кириллицы).
    """
    q = (query or "").strip().strip('"').strip("'")
    if not q:
        return q, "not_needed"

    cyr, lat = split_query_by_script(q)
    if not cyr:
        return q, "not_needed"
    if llm_provider is None:
        return q, "no_llm"

    translated = translate_query_to_en(
        cyr,
        llm_provider,
        system=system,
        user_template=user_template,
    )
    if not translated or translated == cyr or not looks_latin(translated):
        return q, "failed"

    # Оставляем в латинской части только осмысленные токены (без висящих | и запятых).
    lat_clean = " ".join(tok for tok in lat.split() if re.search(r"[A-Za-z0-9]", tok))
    parts = [p for p in (translated.strip(), lat_clean.strip()) if p]
    out = " ".join(parts).strip()
    out = re.sub(r"\s*[|,]\s*$", "", out).strip()
    return out, "translated"
