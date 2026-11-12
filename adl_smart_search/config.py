import os
from dataclasses import dataclass

DEFAULT_AUTO_STRICTNESS = 0.6
DEFAULT_API_BASE_URL = "https://school-test.mos.ru/api/ai/models/v1/message/async/v1"
DEFAULT_CORP_CATALOG_PATH = "./config/corp_catalog.json"
LEGACY_CORP_CATALOG_PATH = "./corp_catalog.json"


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class AppConfig:
    # ADL
    adl_dir: str
    preferred_lang: str = "ru"

    # Index files location (корень дерева индексов: <index_dir>/local, <index_dir>/ckm)
    index_dir: str = "./data"

    # Базовые конфиги (settings.json, llm.json, corp_catalog.json, .env).
    # Обычно это <repo_root>/config; для старой раскладки может совпадать с index_dir.
    config_dir: str = "./config"
    # Пользовательские overrides — вне репозитория с данными.
    user_config_dir: str = ""
    # Корень Git-репозитория (например, C:\emiastemplates).
    repo_root: str = ""
    # Папка с config/ и data/ (например, <repo_root>\Soft\ADL Smart Search).
    data_root: str = ""

    # Вторая база архетипов (например, зеркало CKM). Пусто = не используется.
    ckm_adl_dir: str = ""

    # Форма поиска
    auto_strictness: float = DEFAULT_AUTO_STRICTNESS

    # LLM по умолчанию (UI): base url corp async, если в cfg не задан свой
    api_base_url_default: str = DEFAULT_API_BASE_URL

    # Каталог корпоративных моделей (JSON-файл, локальный, вне git) и список провайдеров
    corp_catalog_path: str = DEFAULT_CORP_CATALOG_PATH
    corp_providers: str = ""  # через запятую; пусто = вывести из каталога

    # Embeddings
    embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Search
    top_k: int = 5
    keyword_bonus: float = 0.5  # вес лексической (BM25-подобной) компоненты, 0..1.5
    misuse_penalty: float = 0.5  # вес понижения за совпадение в блоке misuse, 0..1
    weak_match_penalty: float = 0.25  # вес понижения за совпадение только в имени/якоре, 0..1
    enable_misuse_penalty: bool = True
    enable_weak_match_penalty: bool = True
    max_results: int = 10  # максимум результатов в выдаче
    # Бюджет токенов на ADL-контекст для ИИ (отдельно для вкладок).
    context_tokens_archetypes: int = 10000
    context_tokens_fields: int = 10000
    sim_floor: float = 0.30  # порог схожести для документов с буквальным совпадением
    sim_floor_pure: float = 0.55  # порог схожести для «чисто смысловых» документов
    # Порог «только по смыслу» для документов на другом языке (кросс-языковой поиск
    # без перевода запроса): такие пары систематически дают меньшую схожесть.
    sim_floor_pure_cross: float = 0.30
    # Ниже этой схожести чисто смысловое совпадение считается ненадёжным:
    # результаты показываем, но помечаем «может быть нерелевантным».
    sim_low_confidence: float = 0.45
    simi_pref_eps: float = 1e-4  # микрошаг для переупорядочивания выдачи: -simi и .vN

    # LLM
    llm_mode: str = "api"  # ollama | api
    ollama_model: str = "gemma4:e4b"
    ollama_base_url: str = ""  # e.g. http://192.168.1.10:11434
    # corp_async (МЭШ.ИИ.Хаб)
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = ""
    # openai_compatible — отдельные настройки, чтобы не показывать данные МЭШ
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    # МЭШ.ИИ.Хаб (corp_async) выбран по умолчанию.
    api_kind: str = "corp_async"  # gigachat | corp_async | openai_compatible
    corp_provider: str = ""
    corp_model_kwargs_json: str = ""
    corp_poll_timeout_s: int = 180
    corp_poll_interval_s: float = 1.0
    corp_use_temperature: bool = False
    corp_temperature: float = 0.2
    corp_use_max_tokens: bool = False
    corp_max_tokens: int = 8192
    corp_use_top_p: bool = False
    corp_top_p: float = 0.9
    corp_use_top_k: bool = False
    corp_top_k: int = 40
    corp_use_enable_thinking: bool = False
    corp_enable_thinking: bool = False
    corp_use_presence_penalty: bool = False
    corp_presence_penalty: float = 0.0
    corp_use_repetition_penalty: bool = False
    corp_repetition_penalty: float = 1.0
    corp_aux_use_temperature: bool = False
    corp_aux_temperature: float = 0.2
    corp_aux_use_max_tokens: bool = False
    corp_aux_max_tokens: int = 8192
    corp_aux_use_top_p: bool = False
    corp_aux_top_p: float = 0.9
    corp_aux_use_top_k: bool = False
    corp_aux_top_k: int = 40
    corp_aux_use_enable_thinking: bool = False
    corp_aux_enable_thinking: bool = False
    corp_aux_use_presence_penalty: bool = False
    corp_aux_presence_penalty: float = 0.0
    corp_aux_use_repetition_penalty: bool = False
    corp_aux_repetition_penalty: float = 1.0
    # GigaChat (Sber). credentials хранится только в .env / окружении.
    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-2"
    gigachat_base_url: str = ""
    gigachat_ca_bundle_file: str = ""
    gigachat_verify_ssl_certs: bool = True
    gigachat_timeout_s: float = 120.0
    gigachat_temperature: float = 0.2
    gigachat_max_tokens: int = 4096
    gigachat_context_window: int = 32000
    ollama_use_temperature: bool = False
    ollama_temperature: float = 0.2
    ollama_use_top_p: bool = False
    ollama_top_p: float = 0.9
    ollama_use_repeat_penalty: bool = False
    ollama_repeat_penalty: float = 1.15
    ollama_use_num_predict: bool = False
    ollama_num_predict: int = 900

    # Отдельная модель для перевода и расширения запроса (dev-режим).
    # Провайдер у обеих моделей общий; пустое значение = основная модель.
    llm_aux_model_enabled: bool = False
    ollama_aux_model: str = ""
    api_aux_model: str = ""
    openai_aux_model: str = ""
    corp_aux_provider: str = ""
    corp_aux_model_kwargs_json: str = ""
    gigachat_aux_model: str = ""


def from_env(default_adl_dir: str) -> AppConfig:
    config_dir = os.getenv("ADL_CONFIG_DIR", "./config")
    data_dir = os.getenv("ADL_INDEX_DIR", "./data")
    return AppConfig(
        adl_dir=os.getenv("ADL_DIR", default_adl_dir),
        preferred_lang=os.getenv("ADL_LANG", "ru"),
        index_dir=data_dir,
        config_dir=config_dir,
        user_config_dir=os.getenv("ADL_USER_CONFIG_DIR", ""),
        repo_root=os.getenv("ADL_REPO_ROOT", ""),
        data_root=os.getenv("ADL_DATA_ROOT", ""),
        ckm_adl_dir=os.getenv("ADL_CKM_DIR", ""),
        auto_strictness=float(os.getenv("ADL_AUTO_STRICTNESS", str(DEFAULT_AUTO_STRICTNESS))),
        api_base_url_default=os.getenv("ADL_API_BASE_URL_DEFAULT", DEFAULT_API_BASE_URL),
        corp_catalog_path=os.getenv("ADL_CORP_CATALOG", os.path.join(config_dir, "corp_catalog.json")),
        corp_providers=os.getenv("ADL_CORP_PROVIDERS", ""),
        embed_model_name=os.getenv("ADL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        top_k=int(os.getenv("ADL_TOP_K", "5")),
        keyword_bonus=float(os.getenv("ADL_KEYWORD_BONUS", "0.5")),
        misuse_penalty=float(os.getenv("ADL_MISUSE_PENALTY", "0.5")),
        weak_match_penalty=float(os.getenv("ADL_WEAK_MATCH_PENALTY", "0.25")),
        enable_misuse_penalty=_env_bool("ADL_MISUSE_PENALTY_ENABLED", "1"),
        enable_weak_match_penalty=_env_bool("ADL_WEAK_MATCH_PENALTY_ENABLED", "1"),
        max_results=int(os.getenv("ADL_MAX_RESULTS", "10")),
        context_tokens_archetypes=int(os.getenv("ADL_CONTEXT_TOKENS_ARCHETYPES", "10000")),
        context_tokens_fields=int(os.getenv("ADL_CONTEXT_TOKENS_FIELDS", "10000")),
        sim_floor=float(os.getenv("ADL_SIM_FLOOR", "0.30")),
        sim_floor_pure=float(os.getenv("ADL_SIM_FLOOR_PURE", "0.55")),
        sim_floor_pure_cross=float(os.getenv("ADL_SIM_FLOOR_PURE_CROSS", "0.30")),
        sim_low_confidence=float(os.getenv("ADL_SIM_LOW_CONFIDENCE", "0.45")),
        simi_pref_eps=float(os.getenv("ADL_SIMI_PREF_EPS", "1e-4")),
        # LLM: в .env живут только секреты. Остальные настройки — в data/llm.json
        # (их сохраняет dev-панель) или в дефолтах AppConfig.
        api_key=os.getenv("LLM_API_KEY", ""),
        gigachat_credentials=os.getenv("GIGACHAT_CREDENTIALS", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )

