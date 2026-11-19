from adl_smart_search.ui.search_tab import highlight_matches_html


def test_highlight_uses_block_with_preserved_newlines():
    out = highlight_matches_html("МЕТАДАННЫЕ:\n  - Поле (at0001)", ["поле"])

    # <pre> Streamlit превращает в stMarkdownPre и теряет переносы строк.
    assert "<pre" not in out
    assert 'class="result-context"' in out
    assert "white-space: pre-wrap" in out
    assert "\n  - " in out
    assert '<span class="match-hl">Поле</span>' in out


def test_highlight_phrase_is_one_span_and_other_words_are_lighter():
    out = highlight_matches_html(
        "боль в груди. Отдельно: боль и груди.",
        ["боль", "груди"],
        ["боль в груди"],
    )

    assert '<span class="match-hl-phrase">боль в груди</span>' in out
    # Слова вне фразы всё равно подсвечиваются, но другим классом.
    assert out.count('class="match-hl"') >= 2


def test_highlight_skips_word_with_common_prefix_only():
    out = highlight_matches_html("Макроцефалия: увеличение головы", ["макроцитарная"])

    assert "match-hl" not in out


def test_highlight_skips_shorter_compound_root():
    out = highlight_matches_html("Железо сыворотки крови", ["железодефицитная"])

    assert "match-hl" not in out


def test_highlight_keeps_inflected_form():
    out = highlight_matches_html("Одышкой считается затруднение дыхания", ["одышка"])

    assert '<span class="match-hl">Одышкой</span>' in out
