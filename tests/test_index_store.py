"""Comprehensive tests for index_store.py — index persistence and incremental builds."""

import json

import numpy as np
import pytest

from adl_smart_search.index_store import (
    IndexPaths,
    _build_old_doc_lookup,
    _iter_adl_files_recursive,
    _load_existing,
    _load_existing_fields,
    _norm_rel_path,
    _resolve_old_doc,
    _sha1_bytes,
    load_archetype_index,
    load_field_index,
)

# ──── _sha1_bytes ─────────────────────────────────────────────────────────────

class TestSha1Bytes:
    def test_deterministic(self):
        assert _sha1_bytes(b"hello") == _sha1_bytes(b"hello")

    def test_different_inputs(self):
        assert _sha1_bytes(b"hello") != _sha1_bytes(b"world")

    def test_known_value(self):
        # SHA1 of "hello" is well-known
        h = _sha1_bytes(b"hello")
        assert len(h) == 40
        assert h == "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"


# ──── _norm_rel_path ──────────────────────────────────────────────────────────

class TestNormRelPath:
    def test_backslash_to_forward(self):
        assert _norm_rel_path("a\\b\\c.adl") == "a/b/c.adl"

    def test_strip_leading_dot_slash(self):
        assert _norm_rel_path("./foo/bar.adl") == "foo/bar.adl"

    def test_strip_leading_dot_backslash(self):
        assert _norm_rel_path(".\\foo\\bar.adl") == "foo/bar.adl"

    def test_already_normalized(self):
        assert _norm_rel_path("a/b/c.adl") == "a/b/c.adl"

    def test_empty(self):
        assert _norm_rel_path("") == ""


# ──── IndexPaths ───────────────────────────────────────────────────────────────

class TestIndexPaths:
    def test_archetype_paths(self):
        p = IndexPaths(dir_path="/tmp/idx")
        assert p.docs_json.endswith("archetype_data.json")
        assert p.vectors_npy.endswith("archetype_vectors.npy")

    def test_field_paths(self):
        p = IndexPaths(dir_path="/tmp/idx")
        assert p.field_docs_json.endswith("field_data.json")
        assert p.field_vectors_npy.endswith("field_vectors.npy")


# ──── _iter_adl_files_recursive ───────────────────────────────────────────────

class TestIterAdlFilesRecursive:
    def test_finds_adl_files(self, tmp_path):
        (tmp_path / "a.adl").write_text("x")
        (tmp_path / "b.ADL").write_text("y")
        (tmp_path / "c.txt").write_text("z")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.adl").write_text("w")
        files = _iter_adl_files_recursive(str(tmp_path))
        assert sorted(files) == ["a.adl", "b.ADL", "sub/d.adl"]

    def test_empty_dir(self, tmp_path):
        assert _iter_adl_files_recursive(str(tmp_path)) == []


# ──── _build_old_doc_lookup / _resolve_old_doc ────────────────────────────────

class TestBuildOldDocLookup:
    def test_lookup_by_file(self):
        docs = [{"file": "a/b.adl"}, {"file": "c.adl"}]
        by_file, by_base = _build_old_doc_lookup(docs)
        assert "a/b.adl" in by_file
        assert "c.adl" in by_file

    def test_lookup_by_basename_unique(self):
        docs = [{"file": "x/adl/arch.adl"}, {"file": "y/other.adl"}]
        _, by_base = _build_old_doc_lookup(docs)
        # basename "arch.adl" is unique → should be in by_base
        assert len(by_base.get("arch.adl", [])) == 1

    def test_lookup_by_basename_duplicate(self):
        docs = [{"file": "a/arch.adl"}, {"file": "b/arch.adl"}]
        _, by_base = _build_old_doc_lookup(docs)
        # basename "arch.adl" appears twice → fallback won't resolve
        cands = by_base.get("arch.adl", [])
        assert len(cands) == 2

    def test_skip_docs_without_file(self):
        docs = [{"file": "a.adl"}, {"something": "else"}]
        by_file, _ = _build_old_doc_lookup(docs)
        assert len(by_file) == 1


class TestResolveOldDoc:
    def test_direct_match(self):
        by_file = {"a/b.adl": {"file": "a/b.adl"}}
        by_base = {}
        assert _resolve_old_doc("a/b.adl", by_file, by_base) == {"file": "a/b.adl"}

    def test_fallback_unique_basename(self):
        by_file = {}
        by_base = {"arch.adl": [{"file": "x/arch.adl"}]}
        result = _resolve_old_doc("y/arch.adl", by_file, by_base)
        assert result == {"file": "x/arch.adl"}

    def test_fallback_duplicate_basename_returns_none(self):
        by_file = {}
        by_base = {"arch.adl": [{"file": "a"}, {"file": "b"}]}
        assert _resolve_old_doc("c/arch.adl", by_file, by_base) is None

    def test_not_found(self):
        assert _resolve_old_doc("missing.adl", {}, {}) is None


# ──── _load_existing / _load_existing_fields ──────────────────────────────────

class TestLoadExisting:
    def _write_arch_index(self, tmp_path, docs, vecs):
        (tmp_path / "archetype_data.json").write_text(json.dumps(docs), encoding="utf-8")
        np.save(str(tmp_path / "archetype_vectors.npy"), vecs)

    def test_v0_list_format(self, tmp_path):
        docs = [{"file": "a.adl"}, {"file": "b.adl"}]
        vecs = np.random.rand(2, 4).astype(np.float32)
        self._write_arch_index(tmp_path, docs, vecs)
        paths = IndexPaths(dir_path=str(tmp_path))
        loaded_docs, loaded_vecs = _load_existing(paths)
        assert len(loaded_docs) == 2
        assert loaded_vecs.shape == (2, 4)

    def test_v1_wrapper_format(self, tmp_path):
        docs = [{"file": "x.adl"}]
        vecs = np.random.rand(1, 3).astype(np.float32)
        self._write_arch_index(tmp_path, {"index_version": 1, "docs": docs}, vecs)
        paths = IndexPaths(dir_path=str(tmp_path))
        loaded_docs, _ = _load_existing(paths)
        assert loaded_docs[0]["file"] == "x.adl"

    def test_unsupported_version_raises(self, tmp_path):
        (tmp_path / "archetype_data.json").write_text(
            json.dumps({"index_version": 99, "docs": []}), encoding="utf-8"
        )
        (tmp_path / "archetype_vectors.npy").write_text("x")  # dummy
        paths = IndexPaths(dir_path=str(tmp_path))
        # Version check происходит в _load_existing до np.load
        with pytest.raises(ValueError):
            _load_existing(paths)

    def test_missing_files_returns_empty(self, tmp_path):
        paths = IndexPaths(dir_path=str(tmp_path))
        docs, vecs = _load_existing(paths)
        assert docs == []
        assert vecs.size == 0

    def test_vectors_normalized(self, tmp_path):
        docs = [{"file": "a.adl"}]
        # Non-unit vector
        vecs = np.array([[3.0, 4.0]], dtype=np.float32)
        self._write_arch_index(tmp_path, docs, vecs)
        paths = IndexPaths(dir_path=str(tmp_path))
        _, loaded_vecs = _load_existing(paths)
        norm = np.linalg.norm(loaded_vecs[0])
        assert abs(norm - 1.0) < 1e-5


class TestLoadExistingFields:
    def _write_field_index(self, tmp_path, docs, vecs):
        (tmp_path / "field_data.json").write_text(json.dumps(docs), encoding="utf-8")
        np.save(str(tmp_path / "field_vectors.npy"), vecs)

    def test_v1_format(self, tmp_path):
        docs = [{"file": "a.adl", "at_code": "at0001"}]
        vecs = np.random.rand(1, 4).astype(np.float32)
        self._write_field_index(tmp_path, {"index_version": 1, "docs": docs}, vecs)
        paths = IndexPaths(dir_path=str(tmp_path))
        loaded_docs, _ = _load_existing_fields(paths)
        assert loaded_docs[0]["at_code"] == "at0001"


# ──── load_archetype_index / load_field_index ─────────────────────────────────

class TestLoadArchetypeIndex:
    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_archetype_index(str(tmp_path))


class TestLoadFieldIndex:
    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_field_index(str(tmp_path))


# ──── index_files_exist ────────────────────────────────────────────────────────

class TestIndexFilesExist:
    def test_empty_dir_false(self, tmp_path):
        from adl_smart_search.index_files import index_files_exist

        assert not index_files_exist(str(tmp_path))

    def test_any_index_file_true(self, tmp_path):
        from adl_smart_search.index_files import index_files_exist

        (tmp_path / "archetype_data.json").write_text("x", encoding="utf-8")
        assert index_files_exist(str(tmp_path))

    def test_vectors_only_true(self, tmp_path):
        from adl_smart_search.index_files import index_files_exist

        (tmp_path / "field_vectors.npy").write_text("x", encoding="utf-8")
        assert index_files_exist(str(tmp_path))

    def test_missing_dir_false(self, tmp_path):
        from adl_smart_search.index_files import index_files_exist

        assert not index_files_exist(str(tmp_path / "nope"))
