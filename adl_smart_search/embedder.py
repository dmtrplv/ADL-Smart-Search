import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _normalize_st_model_name(model_name: str) -> str:
    """
    In offline mode, short names like 'paraphrase-multilingual-MiniLM-L12-v2'
    cannot be resolved without hitting HuggingFace Hub. Normalize common case
    to the canonical repo id under the sentence-transformers org.
    """
    name = (model_name or "").strip()
    if not name:
        return "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    if "/" not in name:
        return f"sentence-transformers/{name}"
    return name


class Embedder:
    def __init__(self, model_name: str, cache_dir: str):
        self.model_name = _normalize_st_model_name(model_name)
        self.cache_dir = cache_dir
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """True, если модель уже загружена в память (без запуска загрузки)."""
        return self._model is not None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            self._load_model_unlocked()
        return self._model

    def _load_model_unlocked(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Не найден пакет 'sentence-transformers'. Установите зависимости в текущий Python:\n"
                    "  python -m pip install -r requirements.txt\n"
                    "Если используете venv — сначала активируйте его (.venv\\Scripts\\activate)."
                ) from e

            offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"

            # Важно: после первого скачивания хотим грузить ТОЛЬКО из локального кэша,
            # чтобы индексация не зависела от сети и не "проверяла" Hub каждый раз.
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    local_files_only=True,
                )
            except Exception as e:
                # Persist full traceback for frozen/no-console debugging.
                try:
                    logs_dir = Path(os.environ.get("ADL_LOG_DIR", "")).expanduser()
                    if not str(logs_dir):
                        logs_dir = Path(self.cache_dir) / ".." / "logs"
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    (logs_dir / "embedder_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                except Exception:
                    logger.debug("embedder: cannot write embedder_traceback.txt")
                logger.exception("embedder: model load failed: %s", self.model_name)

                msg = (str(e) or "").lower()
                looks_like_missing_cache = (
                    isinstance(e, (FileNotFoundError, OSError))
                    and (
                        "couldn't connect to" in msg
                        or "could not locate" in msg
                        or "could not find" in msg
                        or "cached files" in msg
                        or "local_files_only" in msg
                    )
                )

                # If we are offline and it looks like cache is missing -> keep a friendly message.
                if offline and looks_like_missing_cache:
                    raise RuntimeError(
                        "Embedding-модель не найдена локально в кэше.\n"
                        f"Ожидаем, что она уже скачана в папку models: {self.cache_dir}\n"
                        "Подготовьте кэш заранее (см. README: предзагрузка sentence-transformers), "
                        "либо временно отключите офлайн-переменные HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE."
                    ) from e

                # Otherwise show the real error so we can fix packaging/DLL issues.
                raise RuntimeError(
                    "Ошибка загрузки embedding-модели.\n"
                    f"model={self.model_name}\n"
                    f"cache_dir={self.cache_dir}\n"
                    f"error_type={type(e).__name__}\n"
                    f"error={str(e)[:500]}"
                ) from e
        return self._model

    def encode(self, texts, show_progress_bar: bool = False, *, normalize_embeddings: bool = True):
        model = self._ensure_model()
        return model.encode(texts, show_progress_bar=show_progress_bar, normalize_embeddings=normalize_embeddings)

    def embedding_dim(self) -> int:
        """Размерность векторов используемой модели (для проверки совместимости индекса)."""
        model = self._ensure_model()
        dim_fn = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        if callable(dim_fn):
            return int(dim_fn())
        first = model.encode([""], normalize_embeddings=True)
        return int(len(first[0]))


def default_cache_dir() -> str:
    # launcher.py (frozen) sets SENTENCE_TRANSFORMERS_HOME/HF_HOME to <exe_dir>\models.
    # Respect it so packaged app uses external models folder instead of _internal\...\models.
    cache_dir = (os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.environ.get("HF_HOME") or "").strip()
    if not cache_dir:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.normpath(os.path.join(base_dir, "..", "models"))

    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)
    # Match project-style layout: hub cache is directly the models folder.
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return cache_dir


_DEFAULT_EMBEDDER: Optional[Embedder] = None
_WARMUP_DONE = False
_WARMUP_LOCK_THREAD = threading.Lock()


def get_default_embedder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> Embedder:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = Embedder(model_name=model_name, cache_dir=default_cache_dir())
    return _DEFAULT_EMBEDDER


def warmup_embedder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
    """Фоновая загрузка модели в память (daemon-поток), чтобы первый поиск не «висел».

    Безопасно вызывать при каждом рендере app.py: загрузка запускается один раз
    на процесс, повторные вызовы завершаются мгновенно через флаг _WARMUP_DONE.
    """
    global _WARMUP_DONE
    with _WARMUP_LOCK_THREAD:
        if _WARMUP_DONE:
            return
        emb = get_default_embedder(model_name=model_name)
        if emb.is_loaded:
            _WARMUP_DONE = True
            return
        _WARMUP_DONE = True

    def _load():
        try:
            emb.embedding_dim()
        except Exception:
            logger.warning("embedder warmup failed (память разогревается при первом поиске)", exc_info=True)

    threading.Thread(target=_load, name="embedder-warmup", daemon=True).start()

