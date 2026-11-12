from adl_smart_search.adl_parser import get_block_by_brace, get_block_by_bracket


def test_get_block_by_bracket_ignores_angle_brackets_inside_quotes():
    s = 'xxx < "a < 5" > yyy'
    out = get_block_by_bracket(s, 0)
    assert '"a < 5" ' in out


def test_get_block_by_brace_nested():
    s = "a { one { two } three } b"
    out = get_block_by_brace(s, 0)
    assert "one { two } three" in out
