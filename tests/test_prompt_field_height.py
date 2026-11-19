from adl_smart_search.ui.sidebar import (
    _PROMPT_FIELD_MAX_HEIGHT,
    _PROMPT_FIELD_MIN_HEIGHT,
    prompt_field_height,
)


def test_empty_prompt_uses_min_height():
    assert prompt_field_height("") == _PROMPT_FIELD_MIN_HEIGHT


def test_height_grows_with_line_count():
    short = prompt_field_height("одна строка")
    tall = prompt_field_height("\n".join(["строка"] * 10))

    assert short < tall


def test_long_line_counts_as_several_lines():
    one_line = prompt_field_height("короткая строка")
    long_line = prompt_field_height("слово " * 80)

    assert long_line > one_line


def test_height_is_capped():
    huge = prompt_field_height("\n".join(["x"] * 500))

    assert huge == _PROMPT_FIELD_MAX_HEIGHT
