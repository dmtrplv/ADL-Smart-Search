from adl_smart_search.ui.search_tab import _history_with_query


def test_history_adds_new_query_to_front():
    assert _history_with_query(["старый"], "новый") == ["новый", "старый"]


def test_history_moves_repeated_query_to_front_without_duplicate():
    assert _history_with_query(["боль", "рост", "вес"], "  РОСТ  ") == [
        "РОСТ",
        "боль",
        "вес",
    ]


def test_history_limits_to_ten():
    history = [f"запрос {i}" for i in range(12)]

    out = _history_with_query(history, "новый")

    assert len(out) == 10
    assert out[0] == "новый"


def test_history_ignores_empty_query():
    assert _history_with_query(["боль"], "   ") == ["боль"]
