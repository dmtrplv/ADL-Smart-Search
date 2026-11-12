from adl_smart_search.ui.search_tab import (
    _dev_matches_line,
    _dev_score_line,
    dev_legend,
)


def test_dev_score_line_shows_the_formula():
    md = {"lexical_enabled": True, "misuse_hit": True}

    line = _dev_score_line(0.643, 0.62, 0.05, 0.75, md, 0.5)

    assert "score 0.643" in line
    assert "схожесть 0.62" in line
    assert "0.50 × лексика 0.05" in line
    assert "штраф 0.75 (misuse)" in line


def test_dev_score_line_omits_penalty_when_none():
    line = _dev_score_line(0.643, 0.62, 0.05, 1.0, {"lexical_enabled": True}, 0.5)

    assert "штраф" not in line


def test_dev_score_line_marks_disabled_lexical():
    line = _dev_score_line(0.62, 0.62, 0.0, 1.0, {"lexical_enabled": False}, 0.5)

    assert "лексика отключена" in line
    assert "лексика 0.00" not in line


def test_dev_matches_line_groups_zones():
    md = {
        "in_file": ["одышка"],
        "in_anchor": ["дыхание"],
        "in_searchable": ["пациент"],
        "in_misuse": ["курение"],
        "misuse_hit": True,
    }

    line = _dev_matches_line(md)

    assert "файл — одышка" in line
    assert "якорь — дыхание" in line
    assert "текст — пациент" in line
    assert "misuse — курение" in line


def test_dev_matches_line_empty_when_no_hits():
    assert _dev_matches_line({}) == ""


def test_dev_score_line_shows_zscore_as_reference_only():
    line = _dev_score_line(
        0.933, 0.43, 1.0, 1.0, {"lexical_enabled": True}, 0.5, zscore=0.12
    )
    assert "z +0.12" in line
    assert "справочно" in line


def test_dev_score_line_shows_negative_zscore():
    line = _dev_score_line(
        0.766, 0.50, 0.53, 1.0, {"lexical_enabled": True}, 0.5, zscore=-0.40
    )
    assert "z -0.40" in line


def test_dev_legend_mentions_ordering_by_score():
    legend = dev_legend(0.5)
    assert "0.50" in legend
    assert "по score" in legend
    assert "z показан справочно" in legend
