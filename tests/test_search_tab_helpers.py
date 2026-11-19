from types import SimpleNamespace

import numpy as np

from adl_smart_search.ui.search_tab import (
    _effective_cross_floor,
    _nearest_suggestions,
    _short_file_name,
)
from adl_smart_search.ui.utils import filter_docs


def test_effective_cross_floor_uses_spec_override():
    ctx = SimpleNamespace(sim_floor_pure_cross=0.3)
    spec = SimpleNamespace(sim_floor_pure_cross=0.42)

    assert _effective_cross_floor(ctx, spec) == 0.42


def test_effective_cross_floor_falls_back_to_context():
    ctx = SimpleNamespace(sim_floor_pure_cross=0.3)
    spec = SimpleNamespace(sim_floor_pure_cross=None)

    assert _effective_cross_floor(ctx, spec) == 0.3


def test_nearest_suggestions_orders_by_similarity_and_limits():
    similarities = [0.2, 0.9, 0.5, 0.7]

    out = _nearest_suggestions(similarities, docs=[{}, {}, {}, {}], limit=3)

    assert [item["index"] for item in out] == [1, 3, 2]
    assert [round(item["sim"], 2) for item in out] == [0.9, 0.7, 0.5]


def test_nearest_suggestions_handles_empty_and_zero_limit():
    assert _nearest_suggestions([], []) == []
    assert _nearest_suggestions([0.5], [{}], limit=0) == []


def test_short_file_name_strips_directory_and_adl_extension():
    assert _short_file_name({"file": r"C:\archetypes\blood_pressure.adl"}) == "blood_pressure"
    assert _short_file_name({"file": "folder/NAME.ADL"}) == "NAME"
    assert _short_file_name({"file": "notes.txt"}) == "notes.txt"
    assert _short_file_name({}) == "unknown"

def test_filter_docs_keeps_vectors_and_original_indices_aligned():
    docs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    vectors = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    kept, vecs, back = filter_docs(docs, vectors, lambda d: d["id"] != "b")

    assert kept == [{"id": "a"}, {"id": "c"}]
    assert vecs.tolist() == [[1.0], [3.0]]
    assert back == [0, 2]


def test_filter_docs_handles_empty_result_and_input():
    docs = [{"id": "a"}]
    vectors = np.array([[1.0]], dtype=np.float32)

    kept, vecs, back = filter_docs(docs, vectors, lambda d: False)
    assert kept == []
    assert vecs.size == 0
    assert back == []

    kept, vecs, back = filter_docs([], np.array([], dtype=np.float32), lambda d: True)
    assert kept == []
    assert vecs.size == 0
    assert back == []
