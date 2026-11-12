import threading

import adl_smart_search.embedder as emb_mod


class _StubEmbedder:
    def __init__(self, model_name: str = "m"):
        self.model_name = emb_mod._normalize_st_model_name(model_name)
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def embedding_dim(self) -> int:
        return 0


def test_is_loaded_property(tmp_path, monkeypatch):
    e = emb_mod.Embedder("test-model", str(tmp_path))
    assert e.is_loaded is False


def test_normalize_st_model_name_short_and_full():
    short = "paraphrase-multilingual-MiniLM-L12-v2"
    assert emb_mod._normalize_st_model_name(short) == f"sentence-transformers/{short}"
    assert emb_mod._normalize_st_model_name("sentence-transformers/x") == "sentence-transformers/x"


def _reset_warmup_state():
    emb_mod._WARMUP_DONE = False


def test_warmup_spawns_thread_once_then_idempotent(monkeypatch):
    _reset_warmup_state()
    stub = _StubEmbedder()
    spawns: list[str] = []

    monkeypatch.setattr(emb_mod, "get_default_embedder", lambda **kw: stub)

    started = threading.Event()

    class _FakeThread:
        def __init__(self, target, name="", daemon=False):
            spawns.append(name)
            assert daemon is True
            self._target = target

        def start(self):
            # НЕ запускаем реальную загрузку модели: только помечаем, что поток стартовал.
            started.set()

    monkeypatch.setattr(emb_mod.threading, "Thread", _FakeThread)

    emb_mod.warmup_embedder("test-model")
    assert started.wait(2)
    assert len(spawns) == 1
    assert spawns[0] == "embedder-warmup"
    assert emb_mod._WARMUP_DONE is True

    # Повторный вызов — поток не создаётся повторно.
    emb_mod.warmup_embedder("test-model")
    assert len(spawns) == 1


def test_warmup_skips_thread_when_already_loaded(monkeypatch):
    _reset_warmup_state()
    stub = _StubEmbedder()
    stub._loaded = True
    monkeypatch.setattr(emb_mod, "get_default_embedder", lambda **kw: stub)
    spawns: list[str] = []

    class _FakeThread:
        def __init__(self, target, name="", daemon=False):
            spawns.append(name)

        def start(self):
            pass

    monkeypatch.setattr(emb_mod.threading, "Thread", _FakeThread)
    emb_mod.warmup_embedder("test-model")
    assert spawns == []
    assert emb_mod._WARMUP_DONE is True


def test_warmup_failure_no_crash(monkeypatch):
    _reset_warmup_state()
    stub = _StubEmbedder()

    def _boom():
        raise RuntimeError("no torch here")

    monkeypatch.setattr(stub, "embedding_dim", _boom)
    monkeypatch.setattr(emb_mod, "get_default_embedder", lambda **kw: stub)

    class _FakeThread:
        def __init__(self, target, name="", daemon=False):
            self._target = target

        def start(self):
            # Логика потока: ошибка должна быть поймана внутри _load() и не проброситься наружу.
            self._target()

    monkeypatch.setattr(emb_mod.threading, "Thread", _FakeThread)
    emb_mod.warmup_embedder("test-model")  # не должно упасть