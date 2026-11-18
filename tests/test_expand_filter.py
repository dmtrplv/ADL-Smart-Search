from adl_smart_search.ui.search_tab import (
    _expand_term_in_base,
    _filter_expand_terms_in_base,
)

VOCAB = {
    "масса",
    "тела",
    "рост",
    "опухоли",
    "вес",
    "давление",
    "артериальное",
    "имт",
    "bmi",
    "blood",
    "group",
    "body",
    "weight",
    "spo2",
}


def test_expansion_is_present_when_all_significant_words_exist():
    assert _expand_term_in_base("масса тела", VOCAB) is True
    assert _expand_term_in_base("рост опухоли", VOCAB) is True


def test_expansion_is_hidden_when_a_word_is_missing():
    assert _expand_term_in_base("индекс массы тела", VOCAB) is False
    assert _expand_term_in_base("артериальное давление", VOCAB) is True
    assert _expand_term_in_base("венозное давление", VOCAB) is False


def test_expansion_ignores_service_words():
    assert _expand_term_in_base("масса и тела", VOCAB) is True


def test_filter_keeps_only_terms_present_in_base():
    terms = ["масса тела", "индекс массы тела", "рост опухоли", "что-то несуществующее"]

    assert _filter_expand_terms_in_base(terms, VOCAB) == [
        "масса тела",
        "рост опухоли",
    ]


def test_filter_keeps_terms_of_any_language_if_present_in_base():
    terms = ["масса тела", "blood group", "BMI", "ИМТ"]

    assert _filter_expand_terms_in_base(terms, VOCAB) == [
        "масса тела",
        "blood group",
        "BMI",
        "ИМТ",
    ]


def test_subscript_abbreviation_is_normalized_for_base_check():
    assert _expand_term_in_base("SpO₂", VOCAB) is True
    assert _filter_expand_terms_in_base(["SpO₂"], VOCAB) == ["SpO₂"]
