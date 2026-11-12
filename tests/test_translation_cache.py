import adl_smart_search.ui.search_tab as search_tab


def _fake_translator(calls, result=("dyspnea", "translated")):
    def fake(query, provider, *, system=None, user_template=None):
        calls.append((query, system, user_template))
        return result

    return fake


def test_repeated_query_is_translated_once(monkeypatch):
    calls = []
    monkeypatch.setattr(search_tab, "translate_query_to_en_mixed", _fake_translator(calls))
    cache = {}

    first = search_tab._translate_query_cached(
        cache, "одышка", object(), system="sys", user_template="usr"
    )
    second = search_tab._translate_query_cached(
        cache, "одышка", object(), system="sys", user_template="usr"
    )

    assert first == second == ("dyspnea", "translated")
    assert len(calls) == 1


def test_different_queries_are_translated_separately(monkeypatch):
    calls = []
    monkeypatch.setattr(search_tab, "translate_query_to_en_mixed", _fake_translator(calls))
    cache = {}

    search_tab._translate_query_cached(
        cache, "одышка", object(), system="sys", user_template="usr"
    )
    search_tab._translate_query_cached(
        cache, "кашель", object(), system="sys", user_template="usr"
    )

    assert len(calls) == 2
    assert len(cache) == 2


def test_failed_translation_is_not_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(
        search_tab,
        "translate_query_to_en_mixed",
        _fake_translator(calls, result=("одышка", "failed")),
    )
    cache = {}

    search_tab._translate_query_cached(
        cache, "одышка", object(), system="sys", user_template="usr"
    )
    search_tab._translate_query_cached(
        cache, "одышка", object(), system="sys", user_template="usr"
    )

    assert len(calls) == 2
    assert cache == {}


def test_edited_prompt_bypasses_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(search_tab, "translate_query_to_en_mixed", _fake_translator(calls))
    cache = {}

    search_tab._translate_query_cached(
        cache, "одышка", object(), system="old", user_template="usr"
    )
    search_tab._translate_query_cached(
        cache, "одышка", object(), system="new", user_template="usr"
    )

    assert len(calls) == 2
