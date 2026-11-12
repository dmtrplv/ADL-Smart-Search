import numpy as np

from adl_smart_search.adl_parser import has_language
from adl_smart_search.index_store import (
    build_or_load_field_index_incremental,
    build_or_load_index_incremental,
)

_BILINGUAL = """\
archetype (unstructured)
openEHR-EHR-OBSERVATION.test_lang.v1.v1

language
  original_language = <[ISO_639-1::en]>

description
  details = <
    ["en"] = <
      purpose = <"Monitoring">
      use = <"For observation">
      keywords = <"observation", "parameters">
    >
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
    ["en"] = <
      items = <
        ["at0000"] = <text = <"Observation">>
        ["at0001"] = <text = <"Data tree">>
        ["at0003"] = <text = <"Event">>
        ["at0004"] = <text = <"Systolic BP">>
      >
    >
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

_RUSSIAN_ONLY = """\
archetype (unstructured)
openEHR-EHR-OBSERVATION.test_ru.v1.v1

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
    model_name = "test-embedder"

    def encode(self, texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        rng = np.random.default_rng(42)
        return rng.normal(size=(n, 8)).astype(np.float32)

    def embedding_dim(self):
        return 8


def _write_sources(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "both.adl").write_text(_BILINGUAL, encoding="utf-8")
    (src / "ru_only.adl").write_text(_RUSSIAN_ONLY, encoding="utf-8")
    return src


def test_has_language_detects_english():
    assert has_language(_BILINGUAL, "en")
    assert has_language(_BILINGUAL, "ru")
    assert not has_language(_RUSSIAN_ONLY, "en")


def test_ckm_archetype_index_uses_english_only(tmp_path):
    src = _write_sources(tmp_path)

    docs, _ = build_or_load_index_incremental(
        adl_dir=str(src),
        embedder=_StubEmbedder(),
        preferred_lang="ru",
        index_dir=str(tmp_path / "ckm"),
        source="ckm",
    )

    assert [d["file"] for d in docs] == ["both.adl"]
    assert docs[0]["lang"] == "en"
    assert "observation" in docs[0]["anchor"]
    assert "наблюдение" not in docs[0]["anchor"]


def test_ckm_field_index_uses_english_only(tmp_path):
    src = _write_sources(tmp_path)

    fields, _ = build_or_load_field_index_incremental(
        adl_dir=str(src),
        embedder=_StubEmbedder(),
        preferred_lang="ru",
        index_dir=str(tmp_path / "ckm"),
        source="ckm",
    )

    assert fields
    assert {d["file"] for d in fields} == {"both.adl"}
    assert all(d["lang"] == "en" for d in fields)
    assert any("Systolic" in (d.get("field") or "") for d in fields)
    assert not any("Систолическое" in (d.get("field") or "") for d in fields)


def test_local_index_still_uses_russian(tmp_path):
    src = _write_sources(tmp_path)

    docs, _ = build_or_load_index_incremental(
        adl_dir=str(src),
        embedder=_StubEmbedder(),
        preferred_lang="ru",
        index_dir=str(tmp_path / "local"),
        source="local",
    )

    assert {d["file"] for d in docs} == {"both.adl", "ru_only.adl"}
    assert any(
        "наблюдение" in (d.get("anchor") or "").lower()
        for d in docs
        if d["file"] == "both.adl"
    )


def test_ckm_rebuild_replaces_old_russian_entries(tmp_path):
    src = _write_sources(tmp_path)
    idx = tmp_path / "mixed"
    embedder = _StubEmbedder()

    # Сначала как локальная база: в индексе остаются русские записи.
    build_or_load_index_incremental(
        adl_dir=str(src),
        embedder=embedder,
        preferred_lang="ru",
        index_dir=str(idx),
        source="local",
    )

    # Затем CKM поверх старого индекса: русские записи не переиспользуются,
    # bilingual берётся по-английски, ru-only пропускается.
    docs, _ = build_or_load_index_incremental(
        adl_dir=str(src),
        embedder=embedder,
        preferred_lang="ru",
        index_dir=str(idx),
        source="ckm",
    )

    assert [d["file"] for d in docs] == ["both.adl"]
    assert docs[0]["lang"] == "en"
