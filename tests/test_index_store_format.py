import json

import numpy as np
import pytest

from adl_smart_search.index_store import (
    META_EMBED_DIM,
    META_EMBED_MODEL,
    PARSER_VERSION,
    build_or_load_field_index_incremental,
    build_or_load_index_incremental,
    ensure_embedding_dim_compatible,
    index_meta,
    index_model_matches,
    load_archetype_index,
    load_field_index,
)


def _write_vectors(path, n: int, dim: int = 4):
    vecs = np.random.rand(n, dim).astype(np.float32)
    np.save(path, vecs)


def test_load_archetype_index_supports_v0_list(tmp_path):
    docs = [{"file": "a.adl", "anchor": "x"}, {"file": "b.adl", "anchor": "y"}]
    pth = tmp_path / "archetype_data.json"
    pth.write_text(json.dumps(docs), encoding="utf-8")
    _write_vectors(tmp_path / "archetype_vectors.npy", n=len(docs))

    out_docs, out_vecs = load_archetype_index(index_dir=str(tmp_path))
    assert len(out_docs) == 2
    assert out_vecs.shape[0] == 2


def test_load_field_index_supports_v1_wrapper(tmp_path):
    docs = [{"file": "a.adl", "anchor": "x", "at_code": "at0001"}]
    wrapped = {"index_version": 1, "docs": docs}
    (tmp_path / "field_data.json").write_text(json.dumps(wrapped), encoding="utf-8")
    _write_vectors(tmp_path / "field_vectors.npy", n=len(docs))

    out_docs, out_vecs = load_field_index(index_dir=str(tmp_path))
    assert out_docs[0]["at_code"] == "at0001"
    assert out_vecs.shape[0] == 1


# ──── Метаданные эмбеддера в индексе (embed_model / embed_dim) ────────────────

class TestIndexMeta:
    def test_missing_file_returns_empty(self, tmp_path):
        meta = index_meta(str(tmp_path), "archetype")
        assert meta[META_EMBED_MODEL] == ""
        assert meta[META_EMBED_DIM] == 0

    def test_legacy_v0_list_returns_empty(self, tmp_path):
        (tmp_path / "archetype_data.json").write_text(
            json.dumps([{"file": "a.adl"}]), encoding="utf-8"
        )
        meta = index_meta(str(tmp_path), "archetype")
        assert meta[META_EMBED_MODEL] == ""
        assert meta[META_EMBED_DIM] == 0

    def test_reads_meta_for_both_kinds(self, tmp_path):
        for kind, fn in (("archetype", "archetype_data.json"), ("field", "field_data.json")):
            (tmp_path / fn).write_text(
                json.dumps(
                    {
                        "index_version": 1,
                        META_EMBED_MODEL: "my-model",
                        META_EMBED_DIM: 384,
                        "docs": [],
                    }
                ),
                encoding="utf-8",
            )
        assert index_meta(str(tmp_path), "archetype") == {META_EMBED_MODEL: "my-model", META_EMBED_DIM: 384}
        assert index_meta(str(tmp_path), "field") == {META_EMBED_MODEL: "my-model", META_EMBED_DIM: 384}


class TestEnsureEmbeddingDimCompatible:
    def test_legacy_meta_absent_ok(self, tmp_path):
        # без меты (legacy) — не проверяем
        ensure_embedding_dim_compatible(str(tmp_path), embed_dim=384, kind="archetype")

    def test_matching_dim_ok(self, tmp_path):
        (tmp_path / "archetype_data.json").write_text(
            json.dumps({META_EMBED_MODEL: "m", META_EMBED_DIM: 384, "docs": []}), encoding="utf-8"
        )
        ensure_embedding_dim_compatible(str(tmp_path), embed_dim=384, kind="archetype")

    def test_mismatch_raises_with_clear_message(self, tmp_path):
        (tmp_path / "archetype_data.json").write_text(
            json.dumps({META_EMBED_MODEL: "wrong-model", META_EMBED_DIM: 768, "docs": []}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="wrong-model"):
            ensure_embedding_dim_compatible(str(tmp_path), embed_dim=384, kind="archetype")


class TestIndexModelMatches:
    def test_no_file_returns_none(self, tmp_path):
        assert index_model_matches(str(tmp_path), "paraphrase-multilingual-MiniLM-L12-v2", "archetype") is None

    def test_legacy_v0_list_returns_none(self, tmp_path):
        (tmp_path / "archetype_data.json").write_text(
            json.dumps([{"file": "a.adl"}]), encoding="utf-8"
        )
        assert index_model_matches(str(tmp_path), "m", "archetype") is None

    def test_empty_model_name_returns_none(self, tmp_path):
        (tmp_path / "archetype_data.json").write_text(
            json.dumps({META_EMBED_MODEL: "m", META_EMBED_DIM: 384, "docs": []}), encoding="utf-8"
        )
        assert index_model_matches(str(tmp_path), "", "archetype") is None

    def test_same_model_returns_true(self, tmp_path):
        (tmp_path / "archetype_data.json").write_text(
            json.dumps(
                {META_EMBED_MODEL: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "docs": []}
            ),
            encoding="utf-8",
        )
        # короткое имя нормализуется в codebase-проект org/
        assert index_model_matches(str(tmp_path), "paraphrase-multilingual-MiniLM-L12-v2", "archetype") is True

    def test_different_model_returns_false(self, tmp_path):
        (tmp_path / "archetype_data.json").write_text(
            json.dumps({META_EMBED_MODEL: "other-model", META_EMBED_DIM: 768, "docs": []}),
            encoding="utf-8",
        )
        assert index_model_matches(str(tmp_path), "paraphrase-multilingual-MiniLM-L12-v2", "archetype") is False

    def test_kind_respected(self, tmp_path):
        (tmp_path / "field_data.json").write_text(
            json.dumps({META_EMBED_MODEL: "sentence-transformers/field-model", "docs": []}), encoding="utf-8"
        )
        assert index_model_matches(str(tmp_path), "field-model", "field") is True
        assert index_model_matches(str(tmp_path), "field-model", "archetype") is None


# ──── End-to-end: мета записывается при сборке индекса ────────────────────────

_MINIMAL_ADL = """\
archetype (unstructured)
openEHR-EHR-OBSERVATION.test_meta.v1.v1

language
  original_language = <[ISO_639-1::ru]>

description
  details = <
    ["ru"] = <
      purpose = <"Мониторинг">
      use = <"Для наблюдения">
      keywords = <"наблюдение", "параметры">
    >
  >

definition
  OBSERVATION[at0000] matches {
    data matches {
      ITEM_TREE[at0001] matches {
        events matches {
          EVENT[at0003] matches {
            data matches {
              ELEMENT[at0004] matches {
                value matches {}
              }
            }
          }
        }
      }
    }
  }

ontology
  term_definitions = <
    ["ru"] = <
      items = <
        ["at0000"] = <text = <"Наблюдение">>
        ["at0001"] = <text = <"Дерево данных">>
        ["at0003"] = <text = <"Событие">>
        ["at0004"] = <text = <"Систолическое АД">>
      >
    >
  >
"""


class _StubEmbedder:
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self._dim = dim

    def encode(self, texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        rng = np.random.default_rng(42)
        return rng.normal(size=(n, self._dim)).astype(np.float32)

    def embedding_dim(self) -> int:
        return self._dim


def test_build_writes_embed_meta(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.adl").write_text(_MINIMAL_ADL, encoding="utf-8")
    idx = tmp_path / "data"
    embedder = _StubEmbedder("sentence-transformers/test-model-v2", 384)

    docs, vectors = build_or_load_index_incremental(
        adl_dir=str(src), embedder=embedder, preferred_lang="ru", index_dir=str(idx)
    )
    field_docs, field_vectors = build_or_load_field_index_incremental(
        adl_dir=str(src), embedder=embedder, preferred_lang="ru", index_dir=str(idx)
    )
    assert len(docs) >= 1
    assert vectors.shape == (len(docs), 384)
    assert len(field_docs) >= 1
    assert field_vectors.shape == (len(field_docs), 384)

    arch_meta = index_meta(str(idx), "archetype")
    assert arch_meta[META_EMBED_MODEL] == "sentence-transformers/test-model-v2"
    assert arch_meta[META_EMBED_DIM] == 384

    field_meta = index_meta(str(idx), "field")
    assert field_meta[META_EMBED_MODEL] == "sentence-transformers/test-model-v2"
    assert field_meta[META_EMBED_DIM] == 384

    # инкрементальная пересборка сохраняет мету
    build_or_load_index_incremental(
        adl_dir=str(src), embedder=embedder, preferred_lang="ru", index_dir=str(idx)
    )
    assert index_meta(str(idx), "archetype") == arch_meta


def test_field_progress_total_is_stable_when_reusing_by_basename(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.adl").write_text(_MINIMAL_ADL, encoding="utf-8")
    idx = tmp_path / "data"
    embedder = _StubEmbedder("sentence-transformers/test-model-v2", 384)
    build_or_load_field_index_incremental(
        adl_dir=str(src), embedder=embedder, preferred_lang="ru", index_dir=str(idx)
    )

    (src / "sub").mkdir()
    (src / "sub" / "a.adl").write_text(_MINIMAL_ADL, encoding="utf-8")
    (src / "zzz.adl").write_text(_MINIMAL_ADL.replace("Наблюдение", "Наблюдение 2"), encoding="utf-8")

    events = []
    build_or_load_field_index_incremental(
        adl_dir=str(src),
        embedder=embedder,
        preferred_lang="ru",
        index_dir=str(idx),
        progress_cb=lambda done, total, phase: events.append((done, total, phase)),
    )

    assert events
    assert all(done <= total for done, total, _phase in events)
    assert {total for _done, total, phase in events if "Сканирование" in phase} == {3}


def test_old_parser_version_forces_rebuild(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.adl").write_text(_MINIMAL_ADL, encoding="utf-8")
    idx = tmp_path / "data"
    embedder = _StubEmbedder("sentence-transformers/test-model-v2", 384)

    build_or_load_index_incremental(
        adl_dir=str(src), embedder=embedder, preferred_lang="ru", index_dir=str(idx)
    )

    # Имитируем индекс, собранный старым парсером.
    docs_path = idx / "archetype_data.json"
    payload = json.loads(docs_path.read_text(encoding="utf-8"))
    for doc in payload["docs"]:
        doc.pop("parser_version", None)
    docs_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    docs, _ = build_or_load_index_incremental(
        adl_dir=str(src), embedder=embedder, preferred_lang="ru", index_dir=str(idx)
    )

    assert docs
    assert all(d.get("parser_version") == PARSER_VERSION for d in docs)

