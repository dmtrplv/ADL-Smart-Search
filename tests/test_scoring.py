"""Тесты новой модели ранжирования: BM25-лексика, демоушены, отбор результатов."""

import numpy as np

from adl_smart_search.rag import (
    compute_scores,
    compute_scores_multilingual,
    is_low_confidence,
    pure_floor_by_doc_lang,
)
from adl_smart_search.ui.settings import select_indices_by_relevance


class _MockEmbedder:
    """Все документы одинаково «похожи» на запрос (sim = 1.0)."""

    def encode(self, texts, normalize_embeddings=True):
        dim = 8
        out = np.zeros((len(texts), dim), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _TermEmbedder:
    """Совпадение по термину: вектор = one-hot по наличию термина в тексте."""

    TERMS = ["пульс", "вентиляция", "масса", "температура", "давление"]

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), len(self.TERMS)), dtype=np.float32)
        for i, t in enumerate(texts):
            low = (t or "").lower()
            for j, term in enumerate(self.TERMS):
                if term in low:
                    out[i, j] = 1.0
        return out


def _doc(file: str, text: str, source: str = "local", anchor: str = "", misuse: str = "") -> dict:
    d = {"file": file, "source": source, "anchor": anchor or text, "text": text}
    if misuse:
        d["misuse"] = misuse
    return d


# ── Лексическая компонента (BM25) ────────────────────────────────────────────


def test_lexical_bonus_is_normalized_to_one():
    docs = [
        _doc("a.adl", "шкала оценки боли"),
        _doc("b.adl", "шкала оценки боли шкала"),
    ]
    vectors = np.zeros((2, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    _scores, _sims, bonus, _penalty, _md = compute_scores(
        query="шкала",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    assert float(bonus.max()) == 1.0
    assert 0.0 <= float(bonus.min()) <= 1.0


def test_lexical_matches_morphology_prefix():
    docs = [_doc("a.adl", "установка вентиляции в помещении")]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _scores, _sims, bonus, _penalty, md = compute_scores(
        query="вентиляция",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    assert float(bonus[0]) > 0.0
    assert md[0]["matched_unique"] == ["вентиляция"]


def test_short_token_matches_exactly_only():
    """Короткое слово не должно совпадать с более длинным («вес» ≠ «весь»)."""
    docs = [_doc("a.adl", "весь организм в целом")]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _scores, _sims, bonus, _penalty, md = compute_scores(
        query="вес",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
        token_vocab={"вес"},
    )
    assert float(bonus[0]) == 0.0
    assert md == {}


def test_short_stopword_is_ignored_even_if_in_vocab():
    docs = [_doc("a.adl", "уровень боли по шкале")]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _scores, _sims, bonus, _penalty, md = compute_scores(
        query="боль по шкале",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
        token_vocab={"боль", "по", "шкале"},
    )
    assert "по" not in md[0]["matched_unique"]
    assert float(bonus[0]) > 0.0


def test_lexical_bonus_scales_with_query_coverage():
    """Документ, совпавший только с частью запроса, получает меньший бонус."""
    docs = [
        _doc("a.adl", "настройки ИВЛ"),
        _doc("b.adl", "настройки экрана"),
    ]
    vectors = np.zeros((2, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    _scores, _sims, bonus, _penalty, md = compute_scores(
        query="настройки ИВЛ",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
        token_vocab={"настройки", "ивл"},
    )
    assert md[0]["matched_frac"] == 1.0
    assert md[1]["matched_frac"] == 0.5
    assert float(bonus[0]) > float(bonus[1]) > 0.0


def test_score_formula_sim_plus_weight_times_lex():
    docs = [_doc("a.adl", "шкала оценки"), _doc("b.adl", "шкала оценки шкала")]
    vectors = np.zeros((2, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    scores, sims, bonus, penalty, _md = compute_scores(
        query="шкала",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    for i in range(2):
        assert abs(float(scores[i]) - (float(sims[i]) + 0.5 * float(bonus[i])) * float(penalty[i])) < 1e-5


def test_misuse_hit_lowers_score_multiplicatively():
    docs = [
        _doc("a.adl", "шкала оценки", misuse="Не применять в амбулаторной карте"),
        _doc("b.adl", "шкала оценки"),
    ]
    vectors = np.zeros((2, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    scores, _sims, _bonus, penalty, md = compute_scores(
        query="амбулаторной",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    assert md[0]["misuse_hit"]
    assert float(penalty[0]) == 0.5
    assert float(penalty[1]) == 1.0
    assert float(scores[0]) < float(scores[1])


def test_misuse_penalty_skipped_when_term_also_in_main_text():
    """Термин в теле документа важнее его упоминания в «не использовать»."""
    docs = [
        _doc(
            "blood_pressure.adl",
            "артериальное давление",
            misuse="Не использовать для артериального давления другим методом",
        )
    ]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _scores, _sims, _bonus, penalty, md = compute_scores(
        query="артериальное давление",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    assert float(penalty[0]) == 1.0
    assert not md[0]["misuse_hit"]
    assert md[0]["in_searchable"]


# ── Языковая маршрутизация ───────────────────────────────────────────────────


def test_ckm_group_without_translation_is_semantic_only():
    docs = [
        _doc("local.adl", "вентиляция лёгких", source="local"),
        _doc("ckm.adl", "ventilation of lungs", source="ckm"),
    ]
    vectors = np.zeros((2, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    scores, sims, bonus, _penalty, md = compute_scores_multilingual(
        query="вентиляция",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        query_en="вентиляция",  # перевода нет
        keyword_bonus=0.5,
    )
    # локальная группа: лексика работает
    assert float(bonus[0]) > 0.0
    # CKM без перевода: только семантика
    assert float(bonus[1]) == 0.0
    assert abs(float(scores[1]) - float(sims[1])) < 1e-6


def test_ckm_group_with_translation_uses_lexical():
    docs = [
        _doc("ckm.adl", "ventilation settings", source="ckm"),
        _doc("ckm2.adl", "other archetype", source="ckm"),
    ]
    vectors = np.zeros((2, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    _scores, _sims, bonus, _penalty, md = compute_scores_multilingual(
        query="вентиляция",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        query_en="ventilation",
        keyword_bonus=0.5,
    )
    assert float(bonus[0]) > 0.0
    assert md.get(0, {}).get("lexical_enabled") is True


class _ScriptEmbedder:
    """Даёт разные векторы для кириллического и латинского запроса."""

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for i, t in enumerate(texts):
            has_cyr = any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in t)
            out[i, 0 if has_cyr else 1] = 1.0
        return out


def test_multilingual_similarity_uses_same_query_for_all_sources():
    docs = [
        _doc("local.adl", "боль", source="local"),
        _doc("ckm.adl", "pain", source="ckm"),
    ]
    # local-вектор совпадает с русским запросом, ckm — с английским.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    _scores, sims, _bonus, _penalty, _md = compute_scores_multilingual(
        query="боль",
        docs=docs,
        vectors=vectors,
        embedder=_ScriptEmbedder(),
        query_en="pain",
    )

    # similarity считается исходным запросом для обоих источников,
    # перевод используется только для лексики.
    assert abs(float(sims[0]) - 1.0) < 1e-6
    assert abs(float(sims[1]) - 0.0) < 1e-6


def test_multilingual_lexical_is_normalized_globally():
    docs = [
        _doc("local.adl", "боль боль боль", source="local"),
        _doc("ckm.adl", "pain", source="ckm"),
    ]
    vectors = np.zeros((2, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    _scores, _sims, bonus, _penalty, _md = compute_scores_multilingual(
        query="боль",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        query_en="pain",
        keyword_bonus=0.5,
    )
    assert float(bonus.max()) == 1.0
    assert 0.0 <= float(bonus.min()) <= 1.0


# ── Отбор результатов ────────────────────────────────────────────────────────


def test_selection_respects_max_k():
    scores = np.array([1.0] * 15 + [0.5] * 25, dtype=np.float32)
    sims = np.array([0.9] * 15 + [0.6] * 25, dtype=np.float32)
    picked, _info = select_indices_by_relevance(scores, sims, max_k=10, min_k=3)
    assert len(picked) == 10
    assert picked[0] == 0


def test_selection_drops_docs_below_similarity_floor():
    scores = np.array([0.9, 0.85, 0.1], dtype=np.float32)
    sims = np.array([0.8, 0.75, 0.1], dtype=np.float32)
    picked, info = select_indices_by_relevance(scores, sims, max_k=10, min_k=3, sim_floor=0.3)
    assert 2 not in picked
    assert len(picked) == 2  # добора до min_k нет: нечего добавлять


def test_pure_semantic_docs_need_higher_similarity():
    """«Чисто смысловые» документы отсекаются по более строгому порогу."""
    scores = np.array([0.9, 0.85], dtype=np.float32)
    sims = np.array([0.45, 0.52], dtype=np.float32)
    lex = np.array([0.0, 0.0], dtype=np.float32)
    picked, _info = select_indices_by_relevance(
        scores, sims, lex, max_k=10, min_k=3, sim_floor=0.30, sim_floor_pure=0.55
    )
    assert picked == []

    # тот же документ, но с буквальным совпадением, — проходит мягкий порог
    lex2 = np.array([0.0, 1.0], dtype=np.float32)
    picked2, _info2 = select_indices_by_relevance(
        scores, sims, lex2, max_k=10, min_k=3, sim_floor=0.30, sim_floor_pure=0.55
    )
    assert 1 in picked2


def test_selection_returns_nothing_when_all_below_floor():
    scores = np.array([0.9, 0.85], dtype=np.float32)
    sims = np.array([0.2, 0.1], dtype=np.float32)
    picked, info = select_indices_by_relevance(scores, sims, max_k=10, min_k=3, sim_floor=0.3)
    assert picked == []
    assert info["groups"][""]["reason"] == "below_floor"


def test_selection_cuts_by_gap():
    scores = np.array([1.0, 0.99, 0.5], dtype=np.float32)
    sims = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    picked, _info = select_indices_by_relevance(scores, sims, strictness=0.6, max_k=10, min_k=3)
    # gap между 2-м и 3-м больше gap_max (0.07) → третий отбрасывается,
    # но min_k добирает лучших по score
    assert set(picked) == {0, 1, 2}


def test_selection_keeps_tight_cluster_and_drops_far_ones():
    scores = np.array([1.0, 0.98, 0.97, 0.20], dtype=np.float32)
    sims = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    picked, _info = select_indices_by_relevance(scores, sims, strictness=0.6, max_k=10, min_k=3)
    assert 3 not in picked
    assert picked[:3] == [0, 1, 2]


def test_selection_keeps_both_sources_on_score_tie():
    # local: 1 сильный документ + много средних; ckm: 1 документ с тем же средним score
    scores = np.array([1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    sims = np.array([0.9, 0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    sources = ["local", "local", "local", "local", "ckm"]
    picked, info = select_indices_by_relevance(
        scores, sims, sources=sources, max_k=2, min_k=1, sim_floor=0.3
    )
    assert 0 in picked
    # при равенстве score 0.5 tie-breaker z поднимает единственный ckm-документ
    assert 4 in picked
    assert len(picked) == 2
    assert set(info["groups"]) == {"local", "ckm"}


def test_pure_floor_by_doc_lang_marks_cross_lingual_docs():
    docs = [
        {"source": "local", "lang": "ru"},
        {"source": "ckm", "lang": "en"},
        {"source": "ckm", "lang": "ru"},
    ]
    floors = pure_floor_by_doc_lang(docs, query="пульс", same_floor=0.55, cross_floor=0.48)
    assert np.allclose(floors, [0.55, 0.48, 0.55])


def test_pure_floor_by_doc_lang_uses_one_query_for_all_sources():
    docs = [{"source": "ckm", "lang": "en"}, {"source": "ckm", "lang": "ru"}]
    floors = pure_floor_by_doc_lang(
        docs, query="пульс", same_floor=0.55, cross_floor=0.48
    )
    assert np.allclose(floors, [0.48, 0.55])


def test_pure_floor_by_doc_lang_infers_lang_without_metadata():
    docs = [
        {"source": "ckm", "anchor": "pulse rate"},
        {"source": "ckm", "anchor": "частота пульса"},
    ]
    floors = pure_floor_by_doc_lang(docs, query="пульс", same_floor=0.55, cross_floor=0.48)
    assert np.allclose(floors, [0.48, 0.55])


def test_selection_uses_per_doc_pure_floor():
    scores = np.array([1.0, 0.98], dtype=np.float32)
    sims = np.array([0.9, 0.50], dtype=np.float32)
    picked, _info = select_indices_by_relevance(
        scores,
        sims,
        max_k=2,
        min_k=1,
        sim_floor_pure=0.55,
        sim_floor_pure_by_idx=np.array([0.55, 0.48], dtype=np.float32),
    )
    assert picked == [0, 1]


def test_is_low_confidence_flags_only_weak_semantic_matches():
    assert is_low_confidence(0.39, 0.0, 0.45)
    assert is_low_confidence(0.39, None, 0.45)
    assert not is_low_confidence(0.50, 0.0, 0.45)
    # Буквальное совпадение — не «слабое», даже если эмбеддинг близок мало.
    assert not is_low_confidence(0.20, 0.4, 0.45)
    assert not is_low_confidence(None, 0.0, 0.45)
    assert not is_low_confidence(float("nan"), 0.0, 0.45)


def test_selection_accepts_low_cross_floor_but_not_below():
    scores = np.array([1.0, 0.5], dtype=np.float32)
    sims = np.array([0.39, 0.29], dtype=np.float32)
    floors = np.array([0.30, 0.30], dtype=np.float32)
    picked, _info = select_indices_by_relevance(
        scores, sims, max_k=2, min_k=2, sim_floor_pure_by_idx=floors
    )
    assert picked == [0]


def test_selection_extend_k_returns_extra_candidates():
    scores = np.array([1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93], dtype=np.float32)
    sims = np.full(8, 0.9, dtype=np.float32)
    picked, info = select_indices_by_relevance(scores, sims, max_k=3, min_k=1, extend_k=6)
    assert len(picked) == 3
    candidates = info["candidates"]
    assert len(candidates) == 6
    assert candidates[:3] == picked
    assert candidates[3:] == [3, 4, 5]


def test_selection_extend_k_respects_similarity_floor():
    scores = np.array([1.0, 0.98, 0.8, 0.7], dtype=np.float32)
    sims = np.array([0.9, 0.9, 0.1, 0.9], dtype=np.float32)
    picked, info = select_indices_by_relevance(scores, sims, max_k=2, min_k=1, extend_k=10)
    assert set(picked) == {0, 1}
    assert 2 not in info["candidates"]
    assert info["candidates"] == [0, 1, 3]


def test_selection_candidates_default_to_picked():
    scores = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    sims = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    picked, info = select_indices_by_relevance(scores, sims, max_k=2, min_k=1)
    assert info["candidates"] == picked


def test_selection_ranks_sources_by_score_not_zscore():
    # local: 0.933, CKM: 0.766. Раньше CKM побеждал за счёт большего z.
    scores = np.array([0.933, 0.900, 0.800, 0.766, 0.400, 0.350], dtype=np.float32)
    sims = np.full(6, 0.9, dtype=np.float32)
    sources = ["local", "local", "local", "ckm", "ckm", "ckm"]

    picked, info = select_indices_by_relevance(
        scores, sims, max_k=6, min_k=3, sources=sources, sim_floor=0.3
    )

    assert picked[0] == 0
    assert picked.index(0) < picked.index(3)
    # z по-прежнему считается, но только для диагностики.
    assert info["zscores"][3] > info["zscores"][0]


def test_selection_prefers_higher_score_across_sources():
    # Воспроизведение реального случая: local score 1.207, CKM 0.970,
    # но z у CKM выше. Порядок должен определяться score.
    scores = np.array([1.207, 1.100, 1.050, 0.970, 0.900, 0.850], dtype=np.float32)
    sims = np.full(6, 0.85, dtype=np.float32)
    sources = ["local", "local", "local", "ckm", "ckm", "ckm"]

    picked, _info = select_indices_by_relevance(
        scores, sims, max_k=6, min_k=3, sources=sources, sim_floor=0.3
    )

    assert picked[0] == 0
    assert picked.index(0) < picked.index(3)


def test_selection_is_deterministic():
    scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    sims = np.array([0.7, 0.7, 0.7, 0.7], dtype=np.float32)
    first, _ = select_indices_by_relevance(scores, sims, max_k=3, min_k=3)
    second, _ = select_indices_by_relevance(scores, sims, max_k=3, min_k=3)
    assert first == second == [0, 1, 2]


def test_one_word_query_does_not_explode_result_set():
    """Ключевая проблема: 2 слова не должны давать сотни результатов."""
    docs = [_doc(f"d{i}.adl", f"шкала {i}") for i in range(200)]
    vectors = np.zeros((200, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    embedder = _MockEmbedder()

    def _pick(query: str) -> list[int]:
        scores, sims, _b, _p, _md = compute_scores(
            query=query, docs=docs, vectors=vectors, embedder=embedder, keyword_bonus=0.5
        )
        picked, _info = select_indices_by_relevance(scores, sims, max_k=10, min_k=3)
        return picked

    one = _pick("шкала")
    two = _pick("шкала оценки")
    assert 3 <= len(one) <= 10
    assert 3 <= len(two) <= 10
