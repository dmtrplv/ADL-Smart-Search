from adl_smart_search.ui.search_tab import highlight_matches_html


def test_highlight_uses_block_with_preserved_newlines():
    out = highlight_matches_html("МЕТАДАННЫЕ:\n  - Поле (at0001)", ["поле"])

    # <pre> Streamlit превращает в stMarkdownPre и теряет переносы строк.
    assert "<pre" not in out
    assert 'class="result-context"' in out
    assert "white-space: pre-wrap" in out
    assert "\n  - " in out
    assert '<span class="match-hl">Поле</span>' in out
