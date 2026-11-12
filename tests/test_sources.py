"""Tests for multi-source merge (sources.py)."""

import os

import numpy as np

from adl_smart_search.config import AppConfig
from adl_smart_search.index_files import source_index_dir
from adl_smart_search.sources import (
    LOCAL_SOURCE,
    MergedSources,
    SourceSpec,
    discover_sources,
    discover_sources_from_values,
    load_all,
)


def _fake_vec(n: int):
    out = np.zeros((n, 4), dtype=np.float32)
    out[:, 0] = 1.0
    return out


def _make_docs(n: int):
    return [{"file": f"local/doc{i}.adl", "anchor": f"термин {i}"} for i in range(n)]


def test_source_index_dir_layout():
    assert source_index_dir("data", "local") == os.path.join("data", "local")
    assert source_index_dir("data", "ckm") == os.path.join("data", "ckm")


def test_discover_sources_local_only(tmp_path):
    cfg = AppConfig(
        adl_dir=str(tmp_path / "local"),
        index_dir=str(tmp_path / "data"),
    )
    specs = discover_sources(cfg)
    assert [s.name for s in specs] == ["local"]
    assert specs[0].index_dir == os.path.join(str(tmp_path / "data"), "local")


def test_discover_sources_local_plus_ckm(tmp_path):
    ckm_dir = tmp_path / "ckm"
    ckm_dir.mkdir()
    cfg = AppConfig(
        adl_dir=str(tmp_path / "local"),
        index_dir=str(tmp_path / "data"),
        ckm_adl_dir=str(ckm_dir),
    )
    specs = discover_sources(cfg)
    assert [s.name for s in specs] == ["local", "ckm"]
    assert specs[1].enabled
    assert specs[1].index_dir == os.path.join(str(tmp_path / "data"), "ckm")


def test_discover_sources_ckm_ignored_when_dir_and_index_missing(tmp_path):
    cfg = AppConfig(
        adl_dir=str(tmp_path / "local"),
        index_dir=str(tmp_path / "data"),
        ckm_adl_dir=str(tmp_path / "missing_ckm"),
    )
    specs = discover_sources(cfg)
    assert [s.name for s in specs] == ["local"]


def test_discover_ckm_from_prebuilt_index_without_folder(tmp_path):
    ckm_index = tmp_path / "data" / "ckm"
    ckm_index.mkdir(parents=True)
    (ckm_index / "archetype_data.json").write_text("[]", encoding="utf-8")
    cfg = AppConfig(
        adl_dir=str(tmp_path / "local"),
        index_dir=str(tmp_path / "data"),
        ckm_adl_dir=str(tmp_path / "missing_ckm"),
    )
    specs = discover_sources(cfg)
    assert [s.name for s in specs] == ["local", "ckm"]
    assert specs[1].enabled


def test_discover_sources_from_values_filters_ckm(tmp_path):
    specs = discover_sources_from_values(
        adl_dir=str(tmp_path / "local"),
        index_dir=str(tmp_path / "data"),
        ckm_adl_dir="",
    )
    assert [s.name for s in specs] == ["local"]


def test_load_all_merges_and_tags_sources(tmp_path, monkeypatch):
    def fake_arch(index_dir):
        if os.path.basename(index_dir) == "ckm":
            return _make_docs(2), _fake_vec(2)
        return _make_docs(3), _fake_vec(3)

    def fake_field(index_dir):
        if os.path.basename(index_dir) == "ckm":
            return [{"file": f"ckm/d{i}.adl", "anchor": f"c{i}"} for i in range(2)], _fake_vec(2)
        return [{"file": f"local/f{i}.adl", "anchor": f"f{i}"} for i in range(4)], _fake_vec(4)

    monkeypatch.setattr("adl_smart_search.sources.load_archetype_index_safe", fake_arch)
    monkeypatch.setattr("adl_smart_search.sources.load_field_index_safe", fake_field)

    cfg = AppConfig(
        adl_dir=str(tmp_path / "local"),
        index_dir=str(tmp_path / "data"),
        ckm_adl_dir=str(tmp_path / "ckm"),
    )
    # передаём оба источника явно (каталог CKM может не существовать)
    specs = [
        SourceSpec(LOCAL_SOURCE, "Локальная база", str(tmp_path / "local"), str(tmp_path / "data" / "local")),
        SourceSpec("ckm", "CKM", str(tmp_path / "ckm"), str(tmp_path / "data" / "ckm")),
    ]
    merged = load_all(cfg, specs)
    assert isinstance(merged, MergedSources)
    assert len(merged.docs) == 5
    assert len(merged.field_docs) == 6
    assert merged.vectors.shape == (5, 4)
    assert merged.field_vectors.shape == (6, 4)
    srcs = {d["source"] for d in merged.docs}
    assert srcs == {"local", "ckm"}
    assert merged.source_labels == {"local": "Локальная база", "ckm": "CKM"}


def test_load_all_overrides_stale_source_tag(tmp_path, monkeypatch):
    """Старый баг: индекс CKM был собран с меткой source="local" — она обязана переопределяться."""

    def fake_arch(index_dir):
        if os.path.basename(index_dir) == "ckm":
            return [{"file": "ckm/a.adl", "anchor": "x", "source": "local"}], _fake_vec(1)
        return [{"file": "local/a.adl", "anchor": "y", "source": "local"}], _fake_vec(1)

    def fake_field(index_dir):
        if os.path.basename(index_dir) == "ckm":
            return [{"file": "ckm/a.adl", "anchor": "x", "source": "local"}], _fake_vec(1)
        return [{"file": "local/a.adl", "anchor": "y", "source": "local"}], _fake_vec(1)

    monkeypatch.setattr("adl_smart_search.sources.load_archetype_index_safe", fake_arch)
    monkeypatch.setattr("adl_smart_search.sources.load_field_index_safe", fake_field)

    cfg = AppConfig(
        adl_dir=str(tmp_path / "local"),
        index_dir=str(tmp_path / "data"),
        ckm_adl_dir=str(tmp_path / "ckm"),
    )
    specs = [
        SourceSpec(LOCAL_SOURCE, "Локальная база", str(tmp_path / "local"), str(tmp_path / "data" / "local")),
        SourceSpec("ckm", "CKM", str(tmp_path / "ckm"), str(tmp_path / "data" / "ckm")),
    ]
    merged = load_all(cfg, specs)
    assert [d["source"] for d in merged.docs] == ["local", "ckm"]
    assert [d["source"] for d in merged.field_docs] == ["local", "ckm"]


def test_load_all_empty_when_indexes_missing(tmp_path):
    cfg = AppConfig(
        adl_dir=str(tmp_path / "local"),
        index_dir=str(tmp_path / "data"),
    )
    merged = load_all(cfg)
    assert merged.docs == []
    assert merged.vectors.shape == (0,)
    assert merged.field_docs == []


def test_source_spec_enabled_false_for_empty_dir(tmp_path):
    spec = SourceSpec("x", "X", "", str(tmp_path / "missing"))
    assert not spec.enabled


def test_source_spec_enabled_for_prebuilt_index_without_folder(tmp_path):
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "archetype_data.json").write_text("[]", encoding="utf-8")
    spec = SourceSpec("x", "X", "", str(idx))
    assert spec.enabled


def test_source_spec_enabled_for_folder_without_index(tmp_path):
    src = tmp_path / "adl"
    src.mkdir()
    spec = SourceSpec("x", "X", str(src), str(tmp_path / "idx"))
    assert spec.enabled