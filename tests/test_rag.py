import numpy as np
import pytest

from adl_smart_search.orchestrator import (
    build_followup_dialog,
    handle_followup,
)
from adl_smart_search.query_expand import (
    apply_manual_expansions,
    build_query_expand_prompt,
    build_translate_prompt,
    looks_latin,
    parse_query_expand_response,
    suggest_query_expansions,
    translate_query_to_en,
    translate_query_to_en_mixed,
)
from adl_smart_search.rag import (
    _extract_metadata_lines,
    _extract_relevant_snippets,
    _full_context_without_misuse,
    _search_haystack_for_doc,
    build_compact_context_for_archetypes,
    build_compact_context_for_fields,
    compute_scores,
    compute_scores_multilingual,
    context_tokens_to_max_chars,
    context_window_to_max_chars,
    expand_query,
    expand_query_phrases,
    make_rag_selection_key,
    reorder_picked_with_simi_preference,
    simi_family_key_from_file,
    version_logical_key_and_num,
)

PROMPTS = {
    "system_prompt": (
        "Ты ассистент по openEHR-архетипам. "
        "Если вопрос вне темы — не перечисляй архетипы и поля."
    ),
    "answer_user_prompt": "КОНТЕКСТ:\n{context}\nВОПРОС:\n{question}",
    "continuation_prompt": "Продолжи ответ с того места, где остановился.",
    "query_expand": (
        "Расширение запроса: верни 3–5 элементов на языке исходного запроса."
    ),
    "query_expand_user": "Запрос: {query}",
    "translate": "Перевод запроса.",
    "translate_user": "Перевод: {query}",
}


class _MockEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        dim = 8
        out = np.zeros((len(texts), dim), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _TermEmbedder:
    """Бинарный эмбеддер по вхождениям терминов: dot = число общих терминов."""
    TERMS = ["одышка", "тахипноэ", "кашель", "пульс", "температура", "dyspnea", "fever"]

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), len(self.TERMS)), dtype=np.float32)
        for i, t in enumerate(texts):
            low = (t or "").lower()
            for j, term in enumerate(self.TERMS):
                if term in low:
                    out[i, j] = 1.0
        return out


class _RecordingEmbedder(_TermEmbedder):
    def __init__(self):
        super().__init__()
        self.encoded = []

    def encode(self, texts, normalize_embeddings=True):
        self.encoded.append({"texts": list(texts)})
        return super().encode(texts, normalize_embeddings=True)


class _BrokenEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        raise RuntimeError("model unavailable")


def test_expand_query_adds_expansions():
    q = "ИВЛ"
    out = expand_query(q, expansions={"ивл": ["вентиляция", "ventilation"]})
    assert out.startswith("ИВЛ")
    assert "вентиляция" in out


def test_expand_query_phrases_matches_normalized_phrase():
    q = "Дата,рождения пациента"
    out = expand_query_phrases(q, phrase_expansions={"дата рождения": ["Дата/время рождения"]})
    assert "Дата/время рождения" in out


def test_apply_manual_expansions():
    out = apply_manual_expansions("ИВЛ", ["вентиляция", "ventilation", "ИВЛ"])
    assert out.startswith("ИВЛ | ")
    assert "вентиляция" in out
    assert "ventilation" in out


def test_parse_query_expand_response_json_array():
    terms = parse_query_expand_response('["вентиляция", "ventilation", "respirator"]')
    assert terms == ["вентиляция", "ventilation", "respirator"]


def test_parse_query_expand_response_fenced_and_dedup():
    raw = '```json\n["ИВЛ", "ивл", "механическая вентиляция"]\n```'
    terms = parse_query_expand_response(raw, max_terms=8)
    assert terms[0] == "ИВЛ"
    assert "механическая вентиляция" in terms
    assert len([t for t in terms if t.casefold() == "ивл"]) == 1


def test_parse_query_expand_response_fallback_lines():
    terms = parse_query_expand_response("1. вентиляция\n2. respirator")
    assert "вентиляция" in terms
    assert "respirator" in terms


def test_suggest_query_expansions_uses_provider():
    class _FakeLLM:
        def generate(self, prompt, system=None):
            assert "ИВЛ" in prompt
            assert system
            return '["вентиляция", "ventilation"]'

    terms = suggest_query_expansions(
        "ИВЛ",
        _FakeLLM(),
        system=PROMPTS["query_expand"],
        user_template=PROMPTS["query_expand_user"],
    )
    assert terms == ["вентиляция", "ventilation"]


def test_suggest_query_expansions_caps_at_five():
    class _FakeLLM:
        def generate(self, prompt, system=None):
            return '["a", "b", "c", "d", "e", "f", "g"]'

    terms = suggest_query_expansions(
        "ИВЛ",
        _FakeLLM(),
        system=PROMPTS["query_expand"],
        user_template=PROMPTS["query_expand_user"],
    )

    assert terms == ["a", "b", "c", "d", "e"]


def test_parse_query_expand_response_defaults_to_five():
    raw = '["a", "b", "c", "d", "e", "f", "g"]'

    assert parse_query_expand_response(raw) == ["a", "b", "c", "d", "e"]


def test_query_expand_user_prompt_contains_only_query():
    prompt = build_query_expand_prompt("одышка", PROMPTS["query_expand_user"])

    assert prompt == "Запрос: одышка"


def test_query_expand_system_prompt_asks_for_at_most_five():
    assert "3–5" in PROMPTS["query_expand"]


def test_translate_user_prompt_contains_only_query():
    prompt = build_translate_prompt("одышка", PROMPTS["translate_user"])

    assert prompt == "Перевод: одышка"


def test_suggest_query_expansions_uses_custom_system_prompt():
    class _FakeLLM:
        def __init__(self):
            self.system = None

        def generate(self, prompt, system=None):
            self.system = system
            return '["вентиляция"]'

    llm = _FakeLLM()
    terms = suggest_query_expansions(
        "ИВЛ",
        llm,
        system="МОЙ ПРОМПТ",
        user_template="Запрос: {query}",
    )

    assert llm.system == "МОЙ ПРОМПТ"
    assert terms == ["вентиляция"]


def test_translate_query_uses_custom_system_prompt():
    llm = _FakeTranslateLLM("mechanical ventilation")

    out = translate_query_to_en(
        "вентиляция",
        llm,
        system="МОЙ ПРОМПТ",
        user_template="Переведи: {query}",
    )

    assert out == "mechanical ventilation"
    assert llm.last_system == "МОЙ ПРОМПТ"


# ─── Перевод запроса RU→EN для CKM ──────────────────────────────────────────

def test_looks_latin():
    assert looks_latin("mechanical ventilation")
    assert not looks_latin("механическая вентиляция")
    assert looks_latin("FiO2")
    assert not looks_latin("  ")


class _FakeTranslateLLM:
    def __init__(self, answer: str = "mechanical ventilation"):
        self.answer = answer
        self.last_prompt = ""
        self.last_system = None

    def generate(self, prompt, system=None):
        self.last_prompt = prompt
        self.last_system = system
        return self.answer


def test_translate_query_to_en_passthrough_for_latin():
    out = translate_query_to_en("ventilation", _FakeTranslateLLM())
    assert out == "ventilation"


def test_translate_query_to_en_uses_provider():
    llm = _FakeTranslateLLM("mechanical ventilation settings")
    out = translate_query_to_en(
        "механическая вентиляция", llm, system=PROMPTS["translate"], user_template=PROMPTS["translate_user"]
    )
    assert "механическая вентиляция" in llm.last_prompt
    assert llm.last_system is not None
    assert out == "mechanical ventilation settings"


def test_translate_query_to_en_strips_quotes_and_extras():
    llm = _FakeTranslateLLM('"mechanical ventilation".')
    out = translate_query_to_en(
        "Механическая вентиляция", llm, system=PROMPTS["translate"], user_template=PROMPTS["translate_user"]
    )
    assert out == "mechanical ventilation"


def test_translate_query_to_en_falls_back_on_exception():
    class _Broken:
        def generate(self, prompt, system=None):
            raise RuntimeError("boom")

    out = translate_query_to_en(
        "механическая вентиляция", _Broken(), system=PROMPTS["translate"], user_template=PROMPTS["translate_user"]
    )
    assert out == "механическая вентиляция"


def test_expand_prompt_supports_optional_target_lang():
    template = "Запрос: {query}\nОтветь на {target_lang} языке."

    ru = build_query_expand_prompt("ИВЛ", template)
    assert "русском" in ru

    en = build_query_expand_prompt("ventilation", template)
    assert "английском" in en


def test_translate_expanded_query_keeps_all_terms():
    llm = _FakeTranslateLLM("IVL ventilation artificial ventilation")

    out, note = translate_query_to_en_mixed(
        "ИВЛ | вентиляция, искусственная вентиляция",
        llm,
        system=PROMPTS["translate"], user_template=PROMPTS["translate_user"],
    )

    assert note == "translated"
    for term in ("ИВЛ", "вентиляция", "искусственная"):
        assert term in llm.last_prompt
    assert out == "IVL ventilation artificial ventilation"


def test_translate_expanded_query_keeps_latin_abbreviation():
    llm = _FakeTranslateLLM("IVL ventilation")

    out, note = translate_query_to_en_mixed(
        "ИВЛ | вентиляция, FiO2", llm, system=PROMPTS["translate"], user_template=PROMPTS["translate_user"]
    )

    assert note == "translated"
    assert "FiO2" in out
    assert "|" not in out


def test_compute_scores_uses_expanded_query_for_keywords():
    docs = [
        {
            "file": "x/vent.adl",
            "anchor": "вентиляция",
            "full_context": "МЕТАДАННЫЕ:\nСТРУКТУРА:\n- вентиляция лёгких\n",
        }
    ]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _, _, bonus_plain, _, _ = compute_scores(
        query="ИВЛ",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=2.0,
    )
    _, _, bonus_exp, _, md = compute_scores(
        query="ИВЛ",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=2.0,
        expanded_query=apply_manual_expansions("ИВЛ", ["вентиляция"]),
    )
    # Without expansion "вентиляция" is not in query keywords; with expansion bonus should fire.
    assert float(bonus_plain[0]) == 0.0
    assert float(bonus_exp[0]) > 0.0
    assert md[0].get("in_searchable") or md[0].get("in_anchor")


def test_simi_family_key_from_file():
    assert simi_family_key_from_file("x/openEHR-EHR-OBSERVATION.foo.adl") == ("openehr-ehr-observation.foo", False)
    assert simi_family_key_from_file("x/openEHR-EHR-OBSERVATION.foo-simi.adl") == (
        "openehr-ehr-observation.foo",
        True,
    )
    assert simi_family_key_from_file("x/openEHR-EHR-OBSERVATION.foo-simi.v1.adl") == (
        "openehr-ehr-observation.foo",
        True,
    )
    assert simi_family_key_from_file("x/openEHR-EHR-OBSERVATION.foo.v2.adl") == (
        "openehr-ehr-observation.foo",
        False,
    )


def test_reorder_picked_simi_above_plain_in_mixed_family():
    docs = [
        {"file": "a/foo.adl"},
        {"file": "a/foo-simi.adl"},
        {"file": "b/other.adl"},
    ]
    scores = np.array([0.95, 0.50, 0.40], dtype=np.float32)
    picked = [0, 1, 2]
    ordered, rank = reorder_picked_with_simi_preference(picked, docs, scores, eps=1e-4)
    # simi (idx 1) должен быть выше родителя (idx 0), несмотря на более низкий исходный score
    assert ordered.index(1) < ordered.index(0)
    assert rank[1] > rank[0]


def test_reorder_picked_unchanged_without_simi_pair():
    docs = [{"file": "a/foo.adl"}, {"file": "b/bar.adl"}]
    scores = np.array([0.9, 0.5], dtype=np.float32)
    picked = [0, 1]
    ordered, rank = reorder_picked_with_simi_preference(picked, docs, scores, eps=1e-4)
    assert ordered == [0, 1]
    assert rank[0] == pytest.approx(0.9) and rank[1] == pytest.approx(0.5)


def test_version_logical_key_and_num():
    assert version_logical_key_and_num("x/openEHR-EHR-OBSERVATION.pulse.v1.adl") == (
        "openehr-ehr-observation.pulse",
        1,
    )
    assert version_logical_key_and_num("x/openEHR-EHR-OBSERVATION.pulse.v2.adl") == (
        "openehr-ehr-observation.pulse",
        2,
    )
    assert version_logical_key_and_num("x/openEHR-EHR-OBSERVATION.pulse.adl") == (
        "openehr-ehr-observation.pulse",
        0,
    )


def test_reorder_picked_newer_version_above_older():
    docs = [
        {"file": "a/openEHR-EHR-OBSERVATION.pulse.v1.adl"},
        {"file": "a/openEHR-EHR-OBSERVATION.pulse.v2.adl"},
        {"file": "b/other.adl"},
    ]
    scores = np.array([0.95, 0.50, 0.40], dtype=np.float32)
    picked = [0, 1, 2]
    ordered, rank = reorder_picked_with_simi_preference(picked, docs, scores, eps=1e-4)
    assert ordered.index(1) < ordered.index(0)
    assert rank[1] > rank[0]


def test_reorder_picked_v1_above_implicit_v0():
    docs = [
        {"file": "a/openEHR-EHR-OBSERVATION.pulse.adl"},
        {"file": "a/openEHR-EHR-OBSERVATION.pulse.v1.adl"},
    ]
    scores = np.array([0.9, 0.4], dtype=np.float32)
    picked = [0, 1]
    ordered, _ = reorder_picked_with_simi_preference(picked, docs, scores, eps=1e-4)
    assert ordered.index(1) < ordered.index(0)


def test_reorder_picked_simi_then_version_same_family():
    """Сначала simi над plain, затем среди simi — более новая .vN выше."""
    docs = [
        {"file": "a/foo.v2.adl"},
        {"file": "a/foo-simi.v1.adl"},
    ]
    scores = np.array([0.9, 0.85], dtype=np.float32)
    picked = [0, 1]
    ordered, _ = reorder_picked_with_simi_preference(picked, docs, scores, eps=1e-4)
    # simi выше plain
    assert ordered.index(1) < ordered.index(0)


def test_reorder_picked_only_one_version_no_change():
    docs = [{"file": "a/x.v1.adl"}, {"file": "b/y.v1.adl"}]
    scores = np.array([0.8, 0.7], dtype=np.float32)
    picked = [0, 1]
    ordered, _ = reorder_picked_with_simi_preference(picked, docs, scores, eps=1e-4)
    assert ordered == [0, 1]


def test_make_rag_selection_key_sorted():
    assert make_rag_selection_key([3, 1, 2]) == "1,2,3"


def test_followup_dialog_keeps_first_context_and_answer():
    dialog_messages = [
        {"role": "system", "content": "old system"},
        {"role": "user", "content": "КОНТЕКСТ: --- АРХЕТИП: foo ---\nat0001"},
        {"role": "assistant", "content": "первый ответ"},
        {"role": "user", "content": "уточнение 1"},
        {"role": "assistant", "content": "ответ 1"},
    ]

    dialog = build_followup_dialog(
        prompts=PROMPTS,
        pinned_context="ctx",
        rag_query="ивл",
        dialog_messages=dialog_messages,
        user_message="уточнение 2",
    )

    assert dialog[0]["role"] == "system"
    assert dialog[1]["content"].startswith("КОНТЕКСТ:")
    assert dialog[2] == {"role": "assistant", "content": "первый ответ"}
    assert dialog[-1] == {"role": "user", "content": "уточнение 2"}


def test_followup_dialog_substitutes_query_in_system():
    prompts = dict(PROMPTS, system_prompt="Помоги с запросом: {query}")

    dialog = build_followup_dialog(
        prompts=prompts,
        pinned_context="ctx",
        rag_query="шкала боли",
        dialog_messages=[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        user_message="вопрос",
    )

    system = dialog[0]["content"]
    assert "Помоги с запросом: шкала боли" in system
    assert "{query}" not in system


def test_followup_dialog_keeps_off_topic_rules_in_system_only():
    dialog = build_followup_dialog(
        prompts=PROMPTS,
        pinned_context="ctx",
        rag_query="q",
        dialog_messages=[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        user_message="вопрос",
    )

    system = dialog[0]["content"]
    assert "вне темы" in system
    assert "не перечисляй" in system.lower()

    # Правила живут только в system prompt — в вопрос ничего не дописывается.
    assert dialog[-1]["content"] == "вопрос"


def test_followup_dialog_limits_tail_but_keeps_first_pair():
    dialog_messages = [
        {"role": "user", "content": "context+q1"},
        {"role": "assistant", "content": "a1"},
    ]
    for i in range(2, 8):
        dialog_messages.append({"role": "user", "content": f"q{i}"})
        dialog_messages.append({"role": "assistant", "content": f"a{i}"})

    dialog = build_followup_dialog(
        prompts=PROMPTS,
        pinned_context="ctx",
        rag_query="q",
        dialog_messages=dialog_messages,
        user_message="q8",
        max_history_messages=2,
    )

    contents = [message["content"] for message in dialog]
    assert "context+q1" in contents
    assert "a1" in contents
    assert "q7" in contents
    assert "a7" in contents
    assert "q2" not in contents
    assert contents[-1] == "q8"


def test_followup_dialog_tail_does_not_start_with_assistant():
    dialog = build_followup_dialog(
        prompts=PROMPTS,
        pinned_context="ctx",
        rag_query="q",
        dialog_messages=[
            {"role": "user", "content": "context+q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
        user_message="q3",
        max_history_messages=1,
    )

    assert [message["role"] for message in dialog] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert dialog[1]["content"] == "context+q1"
    assert dialog[2]["content"] == "a1"
    assert dialog[-1]["content"] == "q3"


def test_followup_dialog_falls_back_to_answer_prompt_without_history():
    dialog = build_followup_dialog(
        prompts=PROMPTS,
        pinned_context="ctx",
        rag_query="q",
        dialog_messages=[],
        user_message="вопрос",
    )

    assert dialog[0]["role"] == "system"
    assert dialog[1]["content"] == "КОНТЕКСТ:\nctx\nВОПРОС:\nвопрос"


def test_handle_followup_sends_first_answer_and_history():
    class _Provider:
        def __init__(self):
            self.messages: list[dict] | None = None
            self.last_finish_reason = "stop"

        def chat(self, messages):
            self.messages = messages
            return "ответ"

    provider = _Provider()
    answer = handle_followup(
        question="почему не второй?",
        rag_query="ивл",
        pinned_context="ctx",
        dialog_messages=[
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "КОНТЕКСТ: ctx\nВОПРОС: q1"},
            {"role": "assistant", "content": "первый ответ: варианты A, B, C"},
        ],
        prompts=PROMPTS,
        llm_provider=provider,
    )

    assert answer == "ответ"
    assert provider.messages is not None
    assert provider.messages[1]["content"] == "КОНТЕКСТ: ctx\nВОПРОС: q1"
    assert provider.messages[2]["content"] == "первый ответ: варианты A, B, C"
    assert provider.messages[-1] == {"role": "user", "content": "почему не второй?"}


def test_context_window_to_max_chars_is_calibrated_for_russian():
    chars = context_window_to_max_chars(32000)

    # ~3 символа на токен с запасом 15%.
    assert chars == int(32000 * 3.0 * 0.85)
    assert chars < 32000 * 5.0
    assert context_window_to_max_chars(0) >= 4000


def test_context_tokens_to_max_chars_uses_exact_budget():
    assert context_tokens_to_max_chars(10000) == 30000
    assert context_tokens_to_max_chars(0) == 30000
    assert context_tokens_to_max_chars(1000) == 4000


def test_build_compact_context_caps_documents():
    docs = [
        {
            "file": f"x/openEHR-EHR-OBSERVATION.a{i}.adl",
            "full_context": f"СТРУКТУРА:\n- Поле (at000{i}): значение {i}\n",
        }
        for i in range(8)
    ]

    ctx = build_compact_context_for_archetypes(
        query="поле",
        selected_docs=docs,
        max_documents=5,
        max_chars=20000,
    )

    assert ctx.count("--- АРХЕТИП:") == 5
    assert "a0" in ctx
    assert "a5" not in ctx


def test_full_context_without_misuse_strips_restrictions():
    full_ctx = (
        "МЕТАДАННЫЕ:\n"
        "ЦЕЛЬ: ИВЛ\n"
        "ИСПОЛЬЗОВАНИЕ: Реанимация\n"
        "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ: Не применять в амбулаторной карте\n"
        "СТРУКТУРА:\n"
        "- Режим (at0012)\n"
    )
    stripped = _full_context_without_misuse(full_ctx)
    assert "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ:" not in stripped
    assert "амбулаторной" not in stripped
    assert "ЦЕЛЬ: ИВЛ" in stripped
    assert "ИСПОЛЬЗОВАНИЕ: Реанимация" in stripped
    assert "Режим (at0012)" in stripped


def test_search_haystack_excludes_misuse_from_field_doc():
    doc = {
        "file": "x/foo.adl",
        "anchor": "режим at0012 | текст поля",
        "text": "режим вентиляции",
        "misuse": "Не использовать в амбулаторной карте",
    }
    hay = _search_haystack_for_doc(doc)
    assert "амбулаторной" not in hay
    assert "режим вентиляции" in hay


def test_compute_scores_no_bonus_from_misuse_only():
    docs = [
        {
            "file": "x/obs.adl",
            "anchor": "наблюдение | цель",
            "full_context": (
                "МЕТАДАННЫЕ:\n"
                "ЦЕЛЬ: Другое\n"
                "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ: Не применять в амбулаторной карте\n"
                "СТРУКТУРА:\n"
                "- Поле (at0001)\n"
            ),
            "misuse": "Не применять в амбулаторной карте",
        }
    ]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _, _, bonus, penalty, md = compute_scores(
        query="амбулаторной",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    assert float(bonus[0]) == 0.0
    assert float(penalty[0]) < 1.0  # приоритет понижен (мультипликативно)
    assert md[0]["misuse_hit"]
    assert md[0].get("in_misuse")


def test_compute_scores_bonus_from_use_not_misuse():
    docs = [
        {
            "file": "x/obs.adl",
            "anchor": "наблюдение",
            "full_context": (
                "МЕТАДАННЫЕ:\n"
                "ИСПОЛЬЗОВАНИЕ: В реанимации\n"
                "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ: Не для амбулатории\n"
                "СТРУКТУРА:\n"
                "- Поле (at0001)\n"
            ),
        }
    ]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _, _, bonus, penalty, md = compute_scores(
        query="реанимации",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    assert float(bonus[0]) > 0.0
    assert float(penalty[0]) == 1.0
    assert md[0].get("in_searchable")


def test_compute_scores_field_misuse_does_not_boost():
    docs = [
        {
            "file": "x/foo.adl",
            "anchor": "поле at0001 | значение",
            "text": "числовой показатель",
            "misuse": "Не для амбулаторной помощи",
        }
    ]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _, _, bonus, penalty, md = compute_scores(
        query="амбулаторной",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    assert float(bonus[0]) == 0.0
    assert float(penalty[0]) < 1.0
    assert md[0]["misuse_hit"]


def test_compute_scores_no_coverage_penalty():
    """Штрафа покрытия запроса больше нет: обе разметки шкалы одинаковы."""
    docs = [
        {
            "file": "x/scale.adl",
            "anchor": "шкала",
            "full_context": "МЕТАДАННЫЕ:\nСТРУКТУРА:\n- шкала оценки\n",
        },
        {
            "file": "x/hads-scale.adl",
            "anchor": "шкала hads",
            "full_context": "МЕТАДАННЫЕ:\nСТРУКТУРА:\n- шкала hads\n",
        },
    ]
    vectors = np.zeros((2, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    scores, _, bonus, penalty, _md = compute_scores(
        query="шкала hads",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
    )
    assert float(penalty[0]) == 1.0
    assert float(penalty[1]) == 1.0
    assert float(bonus[1]) > float(bonus[0])
    assert float(scores[1]) > float(scores[0])


def test_compute_scores_misuse_penalty_disabled():
    docs = [
        {
            "file": "x/foo.adl",
            "anchor": "поле",
            "misuse": "Не для амбулаторной помощи",
        }
    ]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _, _, _, penalty_on, _ = compute_scores(
        query="амбулаторной",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        enable_misuse_penalty=True,
    )
    _, _, _, penalty_off, _ = compute_scores(
        query="амбулаторной",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        enable_misuse_penalty=False,
    )
    assert float(penalty_on[0]) < 1.0
    assert float(penalty_off[0]) == 1.0


def test_compute_scores_weak_match_penalty_file_only():
    docs = [
        {
            "file": "x/openEHR-EHR-OBSERVATION.hads.adl",
            "anchor": "поле at0001",
            "text": "числовой показатель",
        }
    ]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _, _, bonus, penalty, md = compute_scores(
        query="hads",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        keyword_bonus=0.5,
        enable_weak_match_penalty=True,
    )
    assert float(bonus[0]) > 0.0
    assert float(penalty[0]) < 1.0
    assert md[0]["weak_hit"]
    assert not md[0].get("in_searchable")


def test_extract_metadata_lines_includes_misuse():
    full_ctx = (
        "МЕТАДАННЫЕ:\n"
        "ЦЕЛЬ: Для учёта ИВЛ\n"
        "ИСПОЛЬЗОВАНИЕ: В реанимации\n"
        "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ: Не использовать для самодиагностики\n"
        "ОБОГАЩЕННАЯ СТРУКТУРА:\n"
        "- Поле (at0001)\n"
    )
    purpose, use, misuse = _extract_metadata_lines(full_ctx)
    assert purpose == "Для учёта ИВЛ"
    assert use == "В реанимации"
    assert misuse == "Не использовать для самодиагностики"


def test_build_compact_context_for_archetypes_includes_misuse():
    docs = [
        {
            "file": "x/openEHR-EHR-OBSERVATION.ivl.v1.adl",
            "full_context": (
                "МЕТАДАННЫЕ:\n"
                "ЦЕЛЬ: ИВЛ\n"
                "ИСПОЛЬЗОВАНИЕ: Реанимация\n"
                "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ: Не применять в амбулаторной карте\n"
                "ОБОГАЩЕННАЯ СТРУКТУРА:\n"
                "- Режим (at0012): CPAP\n"
            ),
        }
    ]
    ctx = build_compact_context_for_archetypes(query="ивл", selected_docs=docs, max_chars=5000)
    assert "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ:" in ctx
    assert "Не применять в амбулаторной карте" in ctx


def test_build_compact_context_for_fields_misuse_from_archetype_fallback():
    fields = [
        {
            "file": "x/foo.adl",
            "field": "Режим",
            "at_code": "at0012",
            "rm_type": "DV_TEXT",
            "path": "/content",
            "text": "Режим вентиляции",
        }
    ]
    archetypes = {
        "x/foo.adl": {
            "full_context": (
                "МЕТАДАННЫЕ:\n"
                "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ: Только для стационара\n"
                "ОБОГАЩЕННАЯ СТРУКТУРА:\n"
                "- Режим (at0012)\n"
            )
        }
    }
    ctx = build_compact_context_for_fields(
        query="режим",
        selected_fields=fields,
        archetype_docs_by_file=archetypes,
        max_chars=5000,
    )
    assert "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ:" in ctx
    assert "Только для стационара" in ctx


def test_handle_followup_calls_llm_for_related_term():
    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            return "ответ по контексту"

    llm = _FakeLLM()
    ctx = "--- АРХЕТИП: openEHR-EHR-OBSERVATION.blood ---\n- Показатель крови (at0004): ..."
    ans = handle_followup(
        question="А что с эритроцитами?",
        rag_query="кровь",
        pinned_context=ctx,
        dialog_messages=[],
        prompts=PROMPTS,
        llm_provider=llm,
    )

    assert ans == "ответ по контексту"
    assert llm.calls == 1


def test_handle_followup_sends_off_topic_to_llm():
    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            return "вопрос вне темы, но модель ответила"

    llm = _FakeLLM()
    ctx = "--- АРХЕТИП: openEHR-EHR-OBSERVATION.ivl ---\n- Режим (at0012): CPAP"
    ans = handle_followup(
        question="Схема лечения сахарного диабета",
        rag_query="ивл",
        pinned_context=ctx,
        dialog_messages=[],
        prompts=PROMPTS,
        llm_provider=llm,
    )

    assert ans == "вопрос вне темы, но модель ответила"
    assert llm.calls == 1


# ─── Мультиязычный скоринг по источникам ────────────────────────────────────

def _ml_doc(source, file, anchor):
    return {
        "file": file,
        "source": source,
        "anchor": anchor,
        "full_context": f"СТРУКТУРА:\n- {anchor}\n",
    }


def test_compute_scores_multilingual_uses_query_en_for_ckm():
    docs = [
        _ml_doc("local", "local/vent.adl", "вентиляция лёгких"),
        _ml_doc("ckm", "ckm/vent_open.adl", "mechanical ventilation"),
        _ml_doc("ckm", "ckm/pulse.adl", "pulse rate"),
    ]
    vectors = np.zeros((3, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    _, _, bonus, _, md = compute_scores_multilingual(
        query="вентиляция",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        query_en="ventilation",
        keyword_bonus=2.0,
        expanded_query=None,
    )
    # локальный документ находит RU-слово
    assert bonus[0] > 0, "local doc должен получить бонус по русскому слову"
    # CKM-документ находит EN-слово из перевода запроса
    assert bonus[1] > 0, "CKM doc должен получить бонус по английскому переводу"
    # CKM-документ без совпадения — без бонуса
    assert bonus[2] == 0


def test_compute_scores_multilingual_without_query_en_falls_back_to_query():
    docs = [
        _ml_doc("ckm", "ckm/vent.adl", "mechanical ventilation"),
    ]
    vectors = np.zeros((1, 8), dtype=np.float32)
    vectors[0, 0] = 1.0
    _, _, bonus, _, _ = compute_scores_multilingual(
        query="вентиляция",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        query_en=None,
        keyword_bonus=2.0,
    )
    # без перевода CKM скорится русским запросом (fallback, как раньше)
    assert bonus[0] == 0


def test_compute_scores_multilingual_preserves_order():
    docs = [
        _ml_doc("local", "local/a.adl", "температура тела"),
        _ml_doc("ckm", "ckm/b.adl", "body temperature"),
        _ml_doc("local", "local/c.adl", "пульс"),
    ]
    vectors = np.zeros((3, 8), dtype=np.float32)
    vectors[:, 0] = 1.0
    scores, sim, _, _, _ = compute_scores_multilingual(
        query="температура",
        docs=docs,
        vectors=vectors,
        embedder=_MockEmbedder(),
        query_en="temperature",
    )
    assert len(scores) == 3
    assert sim.shape == (3,)


def test_extract_relevant_snippets_vector_picks_top_block_by_similarity():
    full = (
        "МЕТАДАННЫЕ:\nЦЕЛЬ: ИВЛ\nСТРУКТУРА:\n"
        "- ЧСС (at0001) | DV_COUNT\n  Значения: пульс\n"
        "- Одышка (at0002) | DV_BOOLEAN\n  Описание: одышка при нагрузке\n"
        "- Кашель (at0003) | DV_BOOLEAN\n  Описание: сухой кашель\n"
    )
    out = _extract_relevant_snippets(
        full, keywords=[], max_lines=80, embedder=_TermEmbedder(), query_text="одышка", vector_top_k=1
    )
    assert "Одышка" in out
    assert "одышка при нагрузке" in out
    assert "ЧСС" not in out
    assert "Кашель" not in out


def test_extract_relevant_snippets_vector_keeps_selected_blocks_in_doc_order():
    full = (
        "СТРУКТУРА:\n"
        "- Кашель (at0001) | DV_BOOLEAN\n  Описание: сухой кашель\n"
        "- Одышка (at0002) | DV_BOOLEAN\n  Описание: одышка при нагрузке\n"
        "- Тахипноэ (at0003) | DV_BOOLEAN\n  Описание: тахипноэ в покое\n"
    )
    out = _extract_relevant_snippets(
        full, keywords=[], max_lines=80, embedder=_TermEmbedder(), query_text="тахипноэ", vector_top_k=2
    )
    # по similarity «Тахипноэ» — первый, но в контексте блоки должны идти в порядке документа
    assert out.index("Одышка") < out.index("Тахипноэ")


def test_extract_relevant_snippets_lexical_fallback_when_embedder_fails():
    full = (
        "СТРУКТУРА:\n"
        "- ИВЛ (at0001) | DV_TEXT\n  Описание: вентиляция лёгких\n"
        "- Пульс (at0002) | DV_COUNT\n  Значения: пульс\n"
    )
    out = _extract_relevant_snippets(full, keywords=["ивл"], max_lines=80, embedder=_BrokenEmbedder(), query_text="ивл")
    assert "ИВЛ" in out
    assert "вентиляция лёгких" in out


def test_build_compact_context_for_archetypes_vector_routes_query_by_source():
    local_doc = {
        "file": "a/ivl.adl",
        "source": "local",
        "full_context": (
            "СТРУКТУРА:\n"
            "- ЧСС (at0001) | DV_COUNT\n  Значения: пульс\n"
            "- Одышка (at0002) | DV_BOOLEAN\n  Описание: одышка при нагрузке\n"
        ),
    }
    ckm_doc = {
        "file": "ckm/vent.adl",
        "source": "ckm",
        "full_context": (
            "СТРУКТУРА:\n"
            "- Dyspnea (at0002) | DV_BOOLEAN\n  Description: dyspnea on exertion\n"
            "- Fever (at0003) | DV_BOOLEAN\n"
        ),
    }
    emb = _RecordingEmbedder()
    ctx = build_compact_context_for_archetypes(
        query="одышка",
        selected_docs=[local_doc, ckm_doc],
        embedder=emb,
        query_en="dyspnea",
        expanded_query="одышка | тахипноэ",
        max_chars=5000,
    )
    assert len(emb.encoded) == 2
    assert emb.encoded[0]["texts"][0] == "одышка | тахипноэ"
    assert emb.encoded[1]["texts"][0] == "dyspnea"
    assert "--- АРХЕТИП: ivl" in ctx
    assert "--- АРХЕТИП: vent" in ctx


def test_build_compact_context_for_archetypes_lexical_without_embedder():
    docs = [
        {
            "file": "x/openEHR-EHR-OBSERVATION.ivl.v1.adl",
            "full_context": (
                "СТРУКТУРА:\n"
                "- ИВЛ (at0012): CPAP\n"
                "- Пульс (at0013): количество\n"
            ),
        }
    ]
    ctx = build_compact_context_for_archetypes(query="ивл", selected_docs=docs, max_chars=5000)
    assert "ИВЛ" in ctx

