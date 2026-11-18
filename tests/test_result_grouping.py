from adl_smart_search.ui.search_tab import _group_result_indices
from adl_smart_search.ui.tab_archetypes import (
    ARCH_SPEC,
    _archetype_version_key,
    _details_title,
)


def _doc(path: str, source: str) -> dict:
    return {"file": path, "source": source}


def test_same_archetype_version_is_grouped_across_sources():
    docs = [
        _doc("Local/Observations/openEHR-EHR-OBSERVATION.body_weight.v2.adl", "local"),
        _doc("entry/observation/openEHR-EHR-OBSERVATION.body_weight.v2.adl", "ckm"),
    ]

    groups = _group_result_indices([0, 1], docs, _archetype_version_key)

    assert groups == [[0, 1]]


def test_different_versions_are_not_grouped():
    docs = [
        _doc("Local/openEHR-EHR-OBSERVATION.body_weight.v2.adl", "local"),
        _doc("ckm/openEHR-EHR-OBSERVATION.body_weight.v1.adl", "ckm"),
    ]

    groups = _group_result_indices([0, 1], docs, _archetype_version_key)

    assert groups == [[0], [1]]


def test_same_source_duplicates_are_not_grouped():
    docs = [
        _doc("Local/openEHR-EHR-OBSERVATION.body_weight.v2.adl", "local"),
        _doc("!spec/openEHR-EHR-OBSERVATION.body_weight.v2.adl", "local"),
    ]

    groups = _group_result_indices([0, 1], docs, _archetype_version_key)

    assert groups == [[0], [1]]


def test_group_takes_position_of_best_member():
    docs = [
        _doc("ckm/openEHR-EHR-OBSERVATION.a.v1.adl", "ckm"),
        _doc("Local/openEHR-EHR-OBSERVATION.b.v1.adl", "local"),
        _doc("Local/openEHR-EHR-OBSERVATION.a.v1.adl", "local"),
    ]

    groups = _group_result_indices([0, 1, 2], docs, _archetype_version_key)

    # Первый член группы — самый релевантный; «a» остаётся на первом месте.
    assert groups == [[0, 2], [1]]


def test_arch_spec_uses_version_aware_grouping():
    assert ARCH_SPEC.group_key is _archetype_version_key


def test_single_archetype_details_title_has_no_source():
    assert _details_title("local", 1) == "Метаданные и структура архетипа"
    assert _details_title("ckm", 1) == "Метаданные и структура архетипа"


def test_grouped_archetype_details_title_has_source():
    assert _details_title("local", 2) == (
        "Метаданные и структура архетипа из локальной базы"
    )
    assert _details_title("ckm", 2) == "Метаданные и структура архетипа из CKM"
