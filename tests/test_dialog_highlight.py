from adl_smart_search.ui.dialog import highlight_refs_markdown


def test_highlights_archetype_id_and_at_code():
    text = "Файл openEHR-EHR-OBSERVATION.blood_pressure.v1, поле at0004."

    out = highlight_refs_markdown(text)

    assert "`openEHR-EHR-OBSERVATION.blood_pressure.v1`" in out
    assert "`at0004`" in out
    assert out.endswith(".")


def test_does_not_double_wrap_existing_inline_code():
    text = "Уже `openEHR-EHR-OBSERVATION.blood_pressure.v1` и `at0004`."

    assert highlight_refs_markdown(text) == text


def test_fenced_code_block_is_untouched():
    text = "```\nopenEHR-EHR-OBSERVATION.blood_pressure.v1 at0004\n```"

    assert highlight_refs_markdown(text) == text


def test_does_not_match_longer_tokens():
    text = "at00045 и notat0004"

    assert highlight_refs_markdown(text) == text


def test_highlights_uppercase_at_code():
    assert highlight_refs_markdown("AT0004") == "`AT0004`"


def test_highlights_specialized_dotted_at_codes():
    text = "Поля at0.1, at0.12 и at0.123."

    out = highlight_refs_markdown(text)

    assert "`at0.1`" in out
    assert "`at0.12`" in out
    assert "`at0.123`" in out
    assert out.endswith(".")


def test_does_not_match_incomplete_dotted_code():
    assert highlight_refs_markdown("at0. и at.") == "at0. и at."
