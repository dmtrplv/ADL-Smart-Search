from adl_smart_search.ui.search_tab import highlight_matches_html
from adl_smart_search.ui.tab_fields import _archetype_lookup, _field_block_span

ARCH_CONTEXT = (
    "МЕТАДАННЫЕ:\n"
    'ЦЕЛЬ: "Для записи массы тела."\n'
    "\n"
    "СТРУКТУРА:\n"
    "  - Масса тела (at0000) | OBSERVATION\n"
    "    data:\n"
    "      - history (at0002) | HISTORY\n"
    "          - Вес (at0004) | DV_QUANTITY\n"
    "            Значения: Число (kg): 0.0—1000.0\n"
    "            Описание: Актуальный вес человека.\n"
    "          - Комментарий (at0024) | DV_TEXT\n"
    "            Описание: Дополнительная информация.\n"
)


def test_field_block_span_includes_values_and_description():
    span = _field_block_span(ARCH_CONTEXT, "at0004", "Вес")

    assert span is not None
    fragment = ARCH_CONTEXT[span[0] : span[1]]
    assert "Вес (at0004)" in fragment
    assert "Значения:" in fragment
    assert "Актуальный вес" in fragment
    assert "Комментарий" not in fragment


def test_field_block_span_returns_none_for_unknown_at_code():
    assert _field_block_span(ARCH_CONTEXT, "at9999", "Нет такого") is None


def test_archetype_lookup_prefers_exact_source():
    local_doc = {"file": "x/foo.adl", "source": "local", "full_context": "local"}
    ckm_doc = {"file": "x/foo.adl", "source": "ckm", "full_context": "ckm"}

    by_key, by_file = _archetype_lookup([local_doc, ckm_doc])

    assert by_key[("local", "x/foo.adl")] is local_doc
    assert by_key[("ckm", "x/foo.adl")] is ckm_doc
    assert by_file["x/foo.adl"] is local_doc


def test_highlight_wraps_focused_field_and_keeps_phrase():
    span = _field_block_span(ARCH_CONTEXT, "at0004", "Вес")

    out = highlight_matches_html(
        ARCH_CONTEXT,
        ["вес"],
        ["актуальный вес"],
        focus_span=span,
    )

    assert '<div class="field-focus">' in out
    focused = out.split('<div class="field-focus">', 1)[1].split("</div>", 1)[0]
    assert "(at0004)" in focused
    assert "Вес" in focused
    assert "Комментарий" not in focused
    assert '<span class="match-hl-phrase">Актуальный вес</span>' in focused
