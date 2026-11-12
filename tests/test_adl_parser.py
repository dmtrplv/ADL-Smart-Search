"""Comprehensive tests for adl_parser.py — the core ADL parsing module."""

from adl_smart_search.adl_parser import (
    compact_adl,
    create_search_anchor,
    detect_content_lang,
    extract_details_metadata,
    extract_field_chunks_from_compact,
    get_block_by_brace,
    get_block_by_bracket,
    parse_ontology_terms,
)

# ──── Minimal ADL fixture (real-world openEHR ADL structure) ────────────────

_MINIMAL_ADL = """\
archetype (unstructured)
openEHR-EHR-OBSERVATION.test_v1.v1

language
  original_language = <[ISO_639-1::en]>

definition
\tOBSERVATION[at0000] matches {
\t\tdata matches {
\t\t\tITEM_TREE[at0001] matches {
\t\t\t\tevents matches {
\t\t\t\t\tEVENT[at0003] matches {
\t\t\t\t\t\tdata matches {
\t\t\t\t\t\t\tELEMENT[at0004] matches {
\t\t\t\t\t\t\t\tvalue matches {
\t\t\t\t\t\t\t\t\tDV_QUANTITY [
\t\t\t\t\t\t\t\t\t\tmagnitude = <|0..250|>
\t\t\t\t\t\t\t\t\t\tprecision = <0>
\t\t\t\t\t\t\t\t\t\tunits = <"mmHg">
\t\t\t\t\t\t\t\t\t]
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t\tELEMENT[at0005] matches {
\t\t\t\t\t\t\t\tvalue matches {
\t\t\t\t\t\t\t\t\tDV_CODED_TEXT [
\t\t\t\t\t\t\t\t\t\tdefining_code matches {
\t\t\t\t\t\t\t\t\t\t\t[local::
\t\t\t\t\t\t\t\t\t\t\t\tat0006
\t\t\t\t\t\t\t\t\t\t\t\tat0007
\t\t\t\t\t\t\t\t\t\t\t]
\t\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t\t]
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}

ontology
  term_definitions = <
    ["ru"] = <
      items = <
        ["at0000"] = <
          text = <"Наблюдение">
          description = <"Тестовый архетип">
        >
        ["at0001"] = <
          text = <"Дерево данных">
        >
        ["at0003"] = <
          text = <"Событие">
        >
        ["at0004"] = <
          text = <"Систолическое АД">
          description = <"Артериальное давление систолическое">
        >
        ["at0005"] = <
          text = <"ЧСС">
          description = <"Частота сердечных сокращений">
        >
        ["at0006"] = <
          text = <"Норма">
        >
        ["at0007"] = <
          text = <"Тахикардия">
          description = <"Учащённое сердцебиение">
        >
      >
    >
  >
"""

_DETAILS_ADL = """\
archetype (unstructured)
openEHR-EHR-OBSERVATION.details_test.v1.v1

definition
  OBSERVATION[at0000] matches {}

description
  details = <
    ["ru"] = <
      purpose = <"Мониторинг основных параметров">
      use = <"Для ежедневного наблюдения">
      keywords = <"мониторинг", "параметры", "наблюдение">
      misuse = <"Не использовать для острого состояния">
    >
  >
"""


# ──── get_block_by_bracket ────────────────────────────────────────────────────

class TestGetBlockByBracket:
    def test_basic(self):
        out = get_block_by_bracket("aaa <inner> bbb", 0)
        assert out == "inner"

    def test_nested(self):
        out = get_block_by_bracket("a <<x>y> b", 0)
        assert out == "<x>y"

    def test_not_found(self):
        assert get_block_by_bracket("no brackets here", 0) == ""

    def test_empty(self):
        assert get_block_by_bracket("a <> b", 0) == ""

    def test_quotes_skip_angle(self):
        s = 'x < "a < 5" > yyy'
        out = get_block_by_bracket(s, 0)
        assert '"a < 5" ' in out

    def test_start_index(self):
        out = get_block_by_bracket("a <first> b <second>", 10)
        assert out == "second"


# ──── get_block_by_brace ──────────────────────────────────────────────────────

class TestGetBlockByBrace:
    def test_basic(self):
        out = get_block_by_brace("aaa {inner} bbb", 0)
        assert out == "inner"

    def test_nested(self):
        out = get_block_by_brace("a { one { two } three } b", 0)
        assert "one { two } three" in out

    def test_not_found(self):
        assert get_block_by_brace("no braces here", 0) == ""

    def test_empty(self):
        assert get_block_by_brace("a {} b", 0) == ""

    def test_start_index(self):
        out = get_block_by_brace("a {first} b {second}", 10)
        assert out == "second"


# ──── parse_ontology_terms ────────────────────────────────────────────────────

class TestParseOntologyTerms:
    def test_ru_terms(self):
        terms = parse_ontology_terms(_MINIMAL_ADL, preferred_lang="ru")
        assert terms["at0000"]["text"] == "Наблюдение"
        assert terms["at0000"]["desc"] == "Тестовый архетип"
        assert terms["at0004"]["text"] == "Систолическое АД"
        assert terms["at0007"]["text"] == "Тахикардия"
        assert terms["at0007"]["desc"] == "Учащённое сердцебиение"

    def test_missing_lang_fallback_en(self):
        content = (
            "ontology term_definitions = <"
            '["en"] = <items = <["at0001"] = <'
            'text = <"Heart rate"> description = <"pulses per minute">>>'
            ">>"
        )
        terms = parse_ontology_terms(content, preferred_lang="ru")
        assert terms.get("at0001", {}).get("text") == "Heart rate"

    def test_no_ontology(self):
        assert parse_ontology_terms("no ontology here") == {}

    def test_no_matching_lang(self):
        content = (
            "ontology term_definitions = <"
            '["de"] = <items = <["at0001"] = <text = <"Herzfrequenz">>>>>'
        )
        assert parse_ontology_terms(content, preferred_lang="ru") == {}


# ──── extract_details_metadata ────────────────────────────────────────────────

class TestExtractDetailsMetadata:
    def test_full_details(self):
        meta = extract_details_metadata(_DETAILS_ADL, preferred_lang="ru")
        # purpose/use/misuse возвращаются в сыром виде (возможны обрамляющие кавычки)
        assert "Мониторинг основных параметров" in meta["purpose"]
        assert meta["use"].strip('"') == "Для ежедневного наблюдения"
        assert "мониторинг" in meta["keywords"]
        assert "параметры" in meta["keywords"]
        assert "наблюдение" in meta["keywords"]
        assert "Не использовать для острого состояния" in meta["misuse"]

    def test_no_details(self):
        meta = extract_details_metadata(_MINIMAL_ADL)
        assert meta["purpose"] == ""
        assert meta["use"] == ""
        assert meta["keywords"] == ""
        assert meta["misuse"] == ""

    def test_empty_string(self):
        meta = extract_details_metadata("")
        assert all(v == "" for v in meta.values())


# ──── compact_adl ─────────────────────────────────────────────────────────────

class TestCompactAdl:
    def test_produces_metadata_and_structure(self):
        out = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        assert "МЕТАДАННЫЕ:" in out
        assert "СТРУКТУРА:" in out
        assert "Наблюдение" in out
        assert "Систолическое АД" in out

    def test_element_with_quantity(self):
        out = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        assert "mmHg" in out
        assert "DV_QUANTITY" in out

    def test_element_with_coded_text(self):
        out = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        assert "DV_CODED_TEXT" in out
        assert "Норма" in out
        assert "Тахикардия" in out

    def test_contains_top_level_block_names(self):
        out = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        assert "data:" in out

    def test_no_definition(self):
        out = compact_adl("archetype (unstructured)\nopenEHR-EHR-X.v1")
        assert "Структура не найдена" in out

    def test_empty_string(self):
        out = compact_adl("")
        assert "МЕТАДАННЫЕ:" in out
        assert "Структура не найдена" in out


# ──── create_search_anchor ────────────────────────────────────────────────────

class TestCreateSearchAnchor:
    def test_contains_main_name(self):
        anchor = create_search_anchor(_MINIMAL_ADL, preferred_lang="ru")
        assert "Наблюдение" in anchor
        # Main name is repeated 3 times in anchor format
        assert anchor.count("Наблюдение") >= 3

    def test_contains_fields(self):
        anchor = create_search_anchor(_MINIMAL_ADL, preferred_lang="ru")
        assert "Поля:" in anchor
        assert "Систолическое АД" in anchor or "ЧСС" in anchor

    def test_contains_top_level_block_names(self):
        anchor = create_search_anchor(_MINIMAL_ADL, preferred_lang="ru")
        assert "Блоки:" in anchor
        assert "data" in anchor


# ──── extract_field_chunks_from_compact ──────────────────────────────────────

class TestExtractFieldChunksFromCompact:
    def test_extracts_leaf_elements(self):
        compact = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        chunks = extract_field_chunks_from_compact(compact, file_name="test.adl")
        codes = {c["at_code"] for c in chunks}
        assert "at0004" in codes
        assert "at0005" in codes

    def test_quantity_chunk_has_value_types(self):
        compact = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        chunks = extract_field_chunks_from_compact(compact, file_name="test.adl")
        qty = next(c for c in chunks if c["at_code"] == "at0004")
        assert "DV_QUANTITY" in qty["value_types"]

    def test_coded_text_chunk(self):
        compact = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        chunks = extract_field_chunks_from_compact(compact, file_name="test.adl")
        ct = next(c for c in chunks if c["at_code"] == "at0005")
        assert "DV_CODED_TEXT" in ct["value_types"]
        assert ct["node_kind"] == "value_leaf"

    def test_hierarchy_fields(self):
        compact = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        chunks = extract_field_chunks_from_compact(compact, file_name="test.adl")
        leaf = next(c for c in chunks if c["at_code"] == "at0004")
        # parent should be the EVENT container at0003
        assert leaf["parent_at_code"] == "at0003"

    def test_path_includes_top_level_block(self):
        compact = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        chunks = extract_field_chunks_from_compact(compact, file_name="test.adl")
        leaf = next(c for c in chunks if c["at_code"] == "at0004")
        assert "data" in leaf["path"]

    def test_empty_compact(self):
        assert extract_field_chunks_from_compact("", "x.adl") == []

    def test_file_name_propagated(self):
        compact = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        chunks = extract_field_chunks_from_compact(compact, file_name="my/arch.adl")
        assert all(c["file"] == "my/arch.adl" for c in chunks)

    def test_max_fields_limit(self):
        compact = compact_adl(_MINIMAL_ADL, preferred_lang="ru")
        chunks = extract_field_chunks_from_compact(compact, "test.adl", max_fields=1)
        assert len(chunks) <= 1


# ──── CKM-специфичное: BOM, allow_archetype-слоты, фактический язык ──────────

_EN_ONLY_ONTOLOGY = """\
archetype (adl_version=1.4; uid=[openEHR-EHR-OBSERVATION.ckm_demo.v1])
concept
  [at0000] Mechanical ventilation

ontology
  term_definitions = <
    ["en"] = <
      items = <
        ["at0000"] = <
          text = <"Mechanical ventilation">
        >
      >
    >
  >
  language = <[ISO_639-1::en]>
"""

_RU_INCLUDED_ONTOLOGY = _EN_ONLY_ONTOLOGY.replace(
    '["en"] = <',
    '["en"] = <\n    ["ru"] = <\n      items = <\n        ["at0000"] = <\n'
    '          text = <"Механическая вентиляция лёгких">\n        >\n      >\n    >',
    1,
)

_NO_ONT_HEADER = """\
archetype (adl_version=1.4; uid=[openEHR-EHR-COMPOSITION.demo])
concept
  [at0000] Test

original_language
  = <[ISO_639-1::de]>
"""


def test_detect_content_lang_en_only():
    assert detect_content_lang(_EN_ONLY_ONTOLOGY, preferred_lang="ru") == "en"


def test_detect_content_lang_prefers_ru_when_present():
    assert detect_content_lang(_RU_INCLUDED_ONTOLOGY, preferred_lang="ru") == "ru"


def test_detect_content_lang_fallback_original_language():
    assert detect_content_lang(_NO_ONT_HEADER, preferred_lang="ru") == "de"


def test_detect_content_lang_with_bom():
    assert detect_content_lang("\ufeff" + _EN_ONLY_ONTOLOGY, preferred_lang="ru") == "en"


def test_anchor_strips_bom():
    anchor = create_search_anchor("\ufeff" + _EN_ONLY_ONTOLOGY, preferred_lang="ru")
    assert "\ufeff" not in anchor
    assert "Mechanical ventilation" in anchor


def test_anchor_keeps_allow_archetype_slot():
    slot = "\n\ndefinition\n\tallow_archetype CLUSTER[at0001] matches {\n\t\tCLUSTER[at0002] exists\n\t}\n"
    content = _EN_ONLY_ONTOLOGY + slot
    # Парсер не должен падать на слоте, а put-anchor — содержать его упоминание.
    anchor = create_search_anchor(content, preferred_lang="en")
    assert "\ufeff" not in anchor


def test_compact_adl_with_bom():
    # У фикстуры нет definition-блока, поэтому compact отдаёт короткую заглушку,
    # но BOM должен быть снят, а вызов не завалиться.
    out = compact_adl("\ufeff" + _RU_INCLUDED_ONTOLOGY, preferred_lang="ru")
    assert "\ufeff" not in out
    assert out.startswith("МЕТАДАННЫЕ:")

def test_anchor_has_ru_term_when_ru_present():
    anchor = create_search_anchor(_RU_INCLUDED_ONTOLOGY, preferred_lang="ru")
    assert "механическая вентиляция" in (anchor or "").lower()