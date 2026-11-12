import json
from pathlib import Path

import pytest

from adl_smart_search.prompts import (
    PROMPT_KEYS,
    PROMPT_LABELS,
    PROMPT_PART_KEYS,
    PromptConfigError,
    apply_placeholders,
    apply_query_placeholder,
    compose_system_prompt,
    format_default_question,
    load_prompts,
    prompt_text,
    prompts_path,
    save_prompts,
)

FULL = {
    "system_prompt": "СИСТЕМА",
    "default_question_archetypes": "Опиши архетипы: {query}",
    "default_question_fields": "Объясни поля: {query}",
    "answer_user_prompt": "КОНТЕКСТ:\n{context}\nВОПРОС:\n{question}",
    "continuation_prompt": "ПРОДОЛЖИ",
    "query_expand": "РАСШИРЕНИЕ",
    "query_expand_user": "Запрос: {query}",
    "translate": "ПЕРЕВОД",
    "translate_user": "Переведи: {query}",
}


@pytest.fixture(autouse=True)
def _isolate_prompts_env(monkeypatch):
    # Тесты работают с явными tmp-путями, а не с общим файлом из conftest.
    monkeypatch.delenv("ADL_PROMPTS_FILE", raising=False)


def _write_prompts(config_dir: Path, data: dict) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "prompts.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_prompt_keys_and_labels_are_consistent():
    assert set(PROMPT_KEYS) == {"system_prompt", *PROMPT_PART_KEYS}
    assert set(PROMPT_LABELS) == set(PROMPT_PART_KEYS)


def test_load_prompts_reads_all_keys(tmp_path):
    _write_prompts(tmp_path, FULL)

    assert load_prompts(str(tmp_path)) == FULL


def test_load_prompts_missing_file_raises(tmp_path):
    with pytest.raises(PromptConfigError) as exc:
        load_prompts(str(tmp_path))

    assert "Не найден файл промптов" in str(exc.value)
    assert "prompts.json" in str(exc.value)


def test_load_prompts_missing_key_raises(tmp_path):
    data = dict(FULL)
    data.pop("translate")
    _write_prompts(tmp_path, data)

    with pytest.raises(PromptConfigError) as exc:
        load_prompts(str(tmp_path))

    assert "translate" in str(exc.value)


def test_load_prompts_empty_value_raises(tmp_path):
    data = dict(FULL, query_expand="   ")
    _write_prompts(tmp_path, data)

    with pytest.raises(PromptConfigError) as exc:
        load_prompts(str(tmp_path))

    assert "query_expand" in str(exc.value)


def test_load_prompts_not_object_raises(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(PromptConfigError):
        load_prompts(str(tmp_path))


def test_load_prompts_invalid_json_raises(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(PromptConfigError):
        load_prompts(str(tmp_path))


def test_save_prompts_roundtrip(tmp_path):
    assert save_prompts(str(tmp_path), FULL) is True

    assert prompts_path(str(tmp_path)) == str(tmp_path / "prompts.json")
    assert load_prompts(str(tmp_path)) == FULL


def test_save_prompts_rejects_empty(tmp_path):
    data = dict(FULL, system_prompt="")

    with pytest.raises(PromptConfigError):
        save_prompts(str(tmp_path), data)

    assert not (tmp_path / "prompts.json").exists()


def test_compose_system_prompt_returns_system_only():
    assert compose_system_prompt(FULL) == "СИСТЕМА"


def test_compose_system_prompt_substitutes_query():
    prompts = dict(FULL, system_prompt="Помоги с запросом: {query}")

    assert compose_system_prompt(prompts, query="шкала боли") == "Помоги с запросом: шкала боли"


def test_compose_system_prompt_keeps_placeholder_without_query():
    prompts = dict(FULL, system_prompt="Помоги с запросом: {query}")

    assert compose_system_prompt(prompts) == "Помоги с запросом: {query}"


def test_apply_query_placeholder_does_not_touch_other_braces():
    text = "Запрос: {query}; формат: {not_a_placeholder}"

    assert apply_query_placeholder(text, "боль") == "Запрос: боль; формат: {not_a_placeholder}"


def test_apply_placeholders_does_not_touch_other_braces():
    text = "Контекст: {context}\nВопрос: {question}\nJSON: {\"a\": 1}"

    out = apply_placeholders(text, context="CTX", question="Q")

    assert out == "Контекст: CTX\nВопрос: Q\nJSON: {\"a\": 1}"


def test_prompt_text_requires_non_empty_value():
    with pytest.raises(PromptConfigError) as exc:
        prompt_text({"system_prompt": "  "}, "system_prompt")

    assert "system_prompt" in str(exc.value)
    assert "config/prompts.json" in str(exc.value)


def test_compose_system_prompt_requires_system_text():
    with pytest.raises(PromptConfigError) as exc:
        compose_system_prompt(dict(FULL, system_prompt=""))

    assert "config/prompts.json" in str(exc.value)
    assert exc.value.user_message == str(exc.value)


def test_format_default_question_substitutes_query():
    out = format_default_question(FULL, "default_question_fields", "шкала боли")

    assert out == "Объясни поля: шкала боли"


def test_format_default_question_without_placeholder():
    prompts = dict(FULL, default_question_archetypes="Опиши выбранное")

    assert format_default_question(prompts, "default_question_archetypes", "x") == "Опиши выбранное"


def test_format_default_question_empty_template_returns_empty():
    prompts = dict(FULL, default_question_fields="")

    assert format_default_question(prompts, "default_question_fields", "x") == ""
