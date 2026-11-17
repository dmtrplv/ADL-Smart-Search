from adl_smart_search.ui.tab_archetypes import (
    ARCH_SPEC,
    _archetype_class_from_file,
)


def test_archetype_class_from_plain_file_name():
    assert (
        _archetype_class_from_file("openEHR-EHR-OBSERVATION.body_weight.v2.adl")
        == "OBSERVATION"
    )


def test_archetype_class_from_nested_path():
    assert (
        _archetype_class_from_file(
            "International/Observations/openEHR-EHR-OBSERVATION.body_weight.v2.adl"
        )
        == "OBSERVATION"
    )


def test_archetype_class_from_windows_and_spec_paths():
    assert (
        _archetype_class_from_file(
            r"!spec\openEHR-EHR-ACTION.imaging_exam-simi.v0.adl"
        )
        == "ACTION"
    )


def test_archetype_class_is_empty_for_non_archetype_files():
    assert _archetype_class_from_file("notes.txt") == ""
    assert _archetype_class_from_file("") == ""


def test_arch_filter_matches_nested_path():
    doc = {
        "file": "Local/Observations/openEHR-EHR-OBSERVATION.body_weight.v2.adl"
    }

    assert ARCH_SPEC.filter_matches(["OBSERVATION"], doc) is True
    assert ARCH_SPEC.filter_matches(["ACTION"], doc) is False
