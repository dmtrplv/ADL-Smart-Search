import os

import numpy as np

from adl_smart_search.index_cache import _normalize_rows, index_signature
from adl_smart_search.sources import SourceSpec


def test_normalize_rows_makes_unit_vectors():
    mat = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    out = _normalize_rows(mat)

    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], [0.0, 0.0])


def test_index_signature_changes_with_index_files(tmp_path):
    index_dir = tmp_path / "local"
    index_dir.mkdir()
    (index_dir / "archetype_data.json").write_text("{}", encoding="utf-8")
    (index_dir / "archetype_vectors.npy").write_bytes(b"x")
    (index_dir / "field_data.json").write_text("{}", encoding="utf-8")
    (index_dir / "field_vectors.npy").write_bytes(b"y")
    spec = SourceSpec("local", "Локальная база", str(tmp_path), str(index_dir))

    first = index_signature([spec])
    os.utime(index_dir / "field_data.json", (1, 1))
    second = index_signature([spec])

    assert first != second
