"""Ранжирование результатов поиска.

Итоговая оценка документа:

    score = (sim + w_lex * lex_norm) * demotion

- ``sim``       — косинусная близость вектора документа и вектора запроса (0..1).
- ``lex_norm``  — нормированная (0..1) BM25-подобная оценка лексического совпадения:
                  учитывает частоту термина (насыщение) и длину документа.
- ``demotion``  — множитель (1.0 — без понижения; <1 — понижение приоритета).

Что изменилось по сравнению с прежней реализацией:

1. Убран штраф покрытия запроса (``keyword_penalty``). Он зависел от числа слов в
   запросе и сдвигал всю шкалу score, из-за чего количество результатов
   «перескакивало» (3 → 138) при небольшой правке запроса, а в смешанном режиме
   один источник полностью вытеснялся из выдачи.
2. Лексический бонус стал BM25-подобным и нормируется на лучший документ
   запроса среди всех источников, поэтому его вклад ограничен и не «перебивает»
   семантику.
3. Штрафы misuse/weak-match больше не вычитаются из score, а понижают приоритет
   (мультипликативно), т.е. не ломают шкалу.
4. Лексический компонент отключается для группы, где язык запроса не совпадает
   с языком документов (например CKM без перевода): буквальные совпадения
   русских слов в английском тексте бессмысленны.
"""

import re
from typing import Dict, List

import numpy as np

from .context_builder import _MISUSE_LABEL, _extract_metadata_lines

# BM25-подобные параметры.
_BM25_K1 = 1.2
_BM25_B = 0.75
# Степень влияния покрытия запроса на лексический бонус: 1.0 — вклад документа
# пропорционален доле совпавших слов запроса (совпадение одного случайного слова
# не поднимает документ наравне с полным совпадением).
_BM25_COVERAGE_ALPHA = 1.0

# Короткие служебные слова: не считаем их ключевыми, даже если они есть в индексе.
_SHORT_STOPWORDS = {
    # русские предлоги/союзы/частицы
    "в", "во", "на", "по", "за", "из", "от", "до", "ко", "со",
    "не", "ни", "и", "а", "но", "или", "же", "ли", "бы",
    "у", "о", "об", "с", "к", "я", "мы", "он", "она", "они",
    # английские
    "of", "in", "on", "at", "to", "by", "is", "as", "an", "or",
    "and", "the", "be", "do", "if", "it", "no", "so", "up", "we",
}

# Совпадение в блоке «НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ» учитывается, только если документ
# в целом релевантен (иначе штраф срабатывает на случайных совпадениях).
_MISUSE_SIM_MIN = 0.3

# Морфология: для длинных слов ищем по префиксу основы («вентиляц*»),
# чтобы «вентиляция» находила «вентиляции»/«вентиляцию».
# Префикс достаточно длинный (6 символов), иначе «настройки» находит
# «настроение», а «вес» — «весь».
_PREFIX_MIN_LEN = 6
_PREFIX_STEM_LEN = 6

_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")


def _strip_known_misuse(text: str, misuse: str) -> str:
    """Убирает текст misuse из текста (без регулярки — это горячий путь на 15k+ документах)."""
    if not text or not misuse:
        return text
    out = text.replace(misuse, " ")
    if out == text:
        # регистр может отличаться — ищем без учёта регистра
        low_text = text.lower()
        low_misuse = misuse.lower()
        pos = low_text.find(low_misuse)
        if pos >= 0:
            out = text[:pos] + " " + text[pos + len(misuse) :]
    return " ".join(out.split())


def _misuse_text_for_doc(doc: dict) -> str:
    misuse = (doc.get("misuse") or "").strip()
    if misuse:
        return misuse
    if doc.get("full_context"):
        _, _, misuse_from_ctx = _extract_metadata_lines(doc.get("full_context") or "")
        return misuse_from_ctx
    return ""


_MISUSE_LINE_RE = re.compile(rf"^{re.escape(_MISUSE_LABEL)}:.*$", flags=re.MULTILINE)


def _full_context_without_misuse(full_context: str) -> str:
    """Убирает строку «НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ» из метаданных."""
    if not full_context:
        return ""
    out = _MISUSE_LINE_RE.sub("", full_context)
    return re.sub(r"\n{3,}", "\n\n", out)


def _search_body_for_doc(doc: dict) -> str:
    if doc.get("full_context"):
        return _full_context_without_misuse(doc.get("full_context") or "")
    return (doc.get("text") or "").strip()


def _search_haystack_for_doc(doc: dict) -> str:
    misuse = _misuse_text_for_doc(doc)
    file_t = doc.get("file") or ""
    anchor_t = _strip_known_misuse(doc.get("anchor") or "", misuse)
    body = _search_body_for_doc(doc)
    return f"{file_t}\n{anchor_t}\n{body}"


def has_cyrillic(text: str) -> bool:
    """True, если в тексте есть кириллица."""
    return bool(_CYRILLIC_RE.search(text or ""))


def _has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _doc_script(doc: dict) -> str:
    """Письменность документа: ``cyr`` / ``lat`` / ``""`` (не определить)."""
    lang = str(doc.get("lang") or "").strip().lower()
    if lang.startswith("ru"):
        return "cyr"
    if lang.startswith("en"):
        return "lat"
    sample = f"{doc.get('anchor') or ''}\n{doc.get('text') or ''}"
    if has_cyrillic(sample):
        return "cyr"
    if _has_latin(sample):
        return "lat"
    return ""


def pure_floor_by_doc_lang(
    docs: List[dict],
    *,
    query: str,
    expanded_query: str | None = None,
    same_floor: float = 0.55,
    cross_floor: float = 0.30,
) -> np.ndarray:
    """Порог «только по смыслу» для каждого документа с учётом языка.

    Кросс-языковые пары (русский запрос → английский документ и наоборот)
    дают систематически меньшую косинусную близость, чем пары на одном языке:
    при пороге 0.55 релевантные CKM-документы отсекаются. Поэтому документам
    на другом языке разрешён более низкий порог ``cross_floor``.

    Схожесть для всех источников теперь считается одним и тем же запросом,
    поэтому и язык запроса здесь один — ``query``/``expanded_query``.
    """
    qx = (expanded_query if expanded_query is not None else query) or ""
    qx = qx.strip() or (query or "").strip()
    same = float(same_floor)
    cross = min(float(cross_floor), same)
    floors = np.full(len(docs), same, dtype=np.float32)
    if has_cyrillic(qx):
        q_script = "cyr"
    elif _has_latin(qx):
        q_script = "lat"
    else:
        return floors
    for i, doc in enumerate(docs):
        doc_script = _doc_script(doc)
        if doc_script and doc_script != q_script:
            floors[i] = cross
    return floors


def is_low_confidence(
    sim: float | None,
    lex: float | None,
    threshold: float,
) -> bool:
    """Слабое чисто смысловое совпадение (без буквального) ниже ``threshold``."""
    try:
        value = float(sim)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value):
        return False
    try:
        if float(lex) > 0.0:  # type: ignore[arg-type]
            return False
    except (TypeError, ValueError):
        pass
    return value < float(threshold)


def _extract_keywords_for_bonus(query: str, token_vocab: set[str] | None = None) -> list[str]:
    query_clean = (query or "").lower()
    tokens = [w for w in re.findall(r"\w+", query_clean) if w]

    vocab = token_vocab or set()
    kws: list[str] = []
    for t in tokens:
        if len(t) >= 4:
            kws.append(t)
        elif 2 <= len(t) <= 3 and (t in vocab) and (t not in _SHORT_STOPWORDS):
            kws.append(t)

    return list(dict.fromkeys(kws))


def _stem_prefix(token: str) -> str:
    """Основа для морфологического матчинга (консервативная обрезка)."""
    if len(token) >= _PREFIX_MIN_LEN:
        return token[:_PREFIX_STEM_LEN]
    return token


def _compile_lexical_patterns(kws: list[str]) -> list[re.Pattern]:
    """Паттерн совпадения для каждого ключевого слова (с учётом морфологии)."""
    pats: list[re.Pattern] = []
    for kw in kws:
        stem = _stem_prefix(kw)
        if stem != kw:
            pats.append(re.compile(rf"\b{re.escape(stem)}\w*\b", flags=re.IGNORECASE))
        else:
            pats.append(re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE))
    return pats


def _scan_lexical_stats(
    docs: List[dict],
    kws: list[str],
    patterns: list[re.Pattern],
) -> dict:
    """Один проход по корпусу: df, tf, длины документов, зоны совпадений.

    Возвращает словарь с:
      df[k]        — число документов, где встречается ключевое слово k;
      tf[i, k]     — число совпадений k в haystack документа i;
      tf_short[i, k] — число совпадений k в имени файла/«якоре» (вне тела);
      tf_misuse[i, k] — число совпадений k в блоке «НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ»;
      dl[i]        — длина документа в токенах;
      in_file/in_anchor/in_searchable/in_misuse[i] — зоны совпадений.
    """
    n = len(docs)
    n_kw = len(kws)
    df = np.zeros(n_kw, dtype=np.float32)
    tf = np.zeros((n, n_kw), dtype=np.float32)
    tf_short = np.zeros((n, n_kw), dtype=np.float32)
    tf_misuse = np.zeros((n, n_kw), dtype=np.float32)
    dl = np.zeros(n, dtype=np.float32)
    in_file: list[list[str]] = [[] for _ in range(n)]
    in_anchor: list[list[str]] = [[] for _ in range(n)]
    in_searchable: list[list[str]] = [[] for _ in range(n)]
    in_misuse: list[list[str]] = [[] for _ in range(n)]

    for i, doc in enumerate(docs):
        misuse_text = _misuse_text_for_doc(doc)
        file_t = doc.get("file") or ""
        anchor_t = _strip_known_misuse(doc.get("anchor") or "", misuse_text)
        body_t = _search_body_for_doc(doc)
        hay = f"{file_t}\n{anchor_t}\n{body_t}"
        short = f"{file_t}\n{anchor_t}"
        dl[i] = float(len(hay.split()))

        for k, pat in enumerate(patterns):
            c_hay = len(pat.findall(hay))
            c_mis = len(pat.findall(misuse_text)) if misuse_text else 0
            if c_hay:
                df[k] += 1.0
                tf[i, k] = c_hay
                c_short = len(pat.findall(short))
                tf_short[i, k] = c_short
                if c_hay - c_short > 0:
                    in_searchable[i].append(kws[k])
                if pat.search(file_t):
                    in_file[i].append(kws[k])
                if pat.search(anchor_t):
                    in_anchor[i].append(kws[k])
            if c_mis:
                # Совпадение в misuse не даёт бонуса, но понижает приоритет документа.
                tf_misuse[i, k] = c_mis
                in_misuse[i].append(kws[k])

    return {
        "df": df,
        "tf": tf,
        "tf_short": tf_short,
        "tf_misuse": tf_misuse,
        "dl": dl,
        "in_file": in_file,
        "in_anchor": in_anchor,
        "in_searchable": in_searchable,
        "in_misuse": in_misuse,
    }


def _bm25_scores(stats: dict, n_docs: int) -> tuple[np.ndarray, np.ndarray]:
    """BM25-подобная оценка и IDF по ключевым словам."""
    tf = stats["tf"]
    dl = stats["dl"]
    df = stats["df"]
    n_kw = tf.shape[1] if tf.ndim == 2 else 0
    if not n_kw or not n_docs:
        return np.zeros(n_docs, dtype=np.float32), np.zeros(n_kw, dtype=np.float32)

    avgdl = float(dl.mean()) if dl.size else 0.0
    if avgdl <= 0:
        avgdl = 1.0

    N = float(n_docs)
    idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))

    norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (dl / avgdl))
    tf_component = (tf * (_BM25_K1 + 1.0)) / (tf + norm[:, None])
    lex = (idf[None, :] * tf_component).sum(axis=1)

    # Покрытие запроса: доля слов запроса, которые встретились в документе.
    if n_kw > 1 and _BM25_COVERAGE_ALPHA > 0:
        coverage = (tf > 0).sum(axis=1) / float(n_kw)
        lex = lex * np.power(coverage, _BM25_COVERAGE_ALPHA)

    return lex.astype(np.float32), idf.astype(np.float32)


def _normalize_lexical(lex: np.ndarray) -> np.ndarray:
    """Нормировка лексической оценки на лучший документ запроса → [0..1]."""
    if lex.size == 0:
        return lex
    top = float(np.max(lex))
    if top <= 0:
        return np.zeros_like(lex)
    return (lex / top).astype(np.float32)


def compute_scores(
    query: str,
    docs: List[dict],
    vectors: np.ndarray,
    embedder,
    keyword_bonus: float = 0.5,
    misuse_penalty: float = 0.5,
    weak_match_penalty: float = 0.25,
    enable_misuse_penalty: bool = True,
    enable_weak_match_penalty: bool = True,
    token_vocab: set[str] | None = None,
    expanded_query: str | None = None,
    enable_lexical: bool = True,
    similarities: np.ndarray | None = None,
    normalize_lexical: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, dict]]:
    """Считает score/similarity/lex/демоушен для набора документов.

    ``keyword_bonus``       — вес лексической компоненты (0..1.5).
    ``misuse_penalty``      — вес понижения за совпадение в блоке misuse (0..1).
    ``weak_match_penalty``  — вес понижения за совпадение только в имени/якоре (0..1).
    ``enable_lexical``      — False → считаем только семантику (без буквальных совпадений).
    ``similarities``        — готовые косинусные близости (если уже посчитаны снаружи).
    ``normalize_lexical``   — False → вернуть сырые BM25-оценки без деления на максимум
                              (нужно для глобальной нормировки между источниками).

    Возвращает:
      scores      — итоговые оценки;
      similarities— косинусная близость;
      bonus       — нормированная лексическая компонента (0..1);
      penalty     — множитель понижения (1.0 — без понижения);
      match_debug — разбор совпадений по документам.
    """
    qx = (expanded_query if expanded_query is not None else query) or ""
    qx = qx.strip() or (query or "").strip()

    n = len(docs)
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy(), empty.copy(), np.ones(0, dtype=np.float32), {}

    if similarities is None:
        query_vec = embedder.encode([qx], normalize_embeddings=True)
        similarities = np.dot(vectors, query_vec.T).flatten().astype(np.float32)
    else:
        similarities = np.asarray(similarities, dtype=np.float32).reshape(-1)
        if similarities.size != n:
            raise ValueError("compute_scores: similarities size does not match docs")

    lex = np.zeros(n, dtype=np.float32)
    demotion = np.ones(n, dtype=np.float32)
    match_debug: Dict[int, dict] = {}

    kws: list[str] = []
    patterns: list[re.Pattern] = []
    if enable_lexical and keyword_bonus > 0:
        kws = _extract_keywords_for_bonus(qx, token_vocab=token_vocab)
        patterns = _compile_lexical_patterns(kws)

    if patterns:
        stats = _scan_lexical_stats(docs, kws, patterns)
        lex_raw, _idf = _bm25_scores(stats, n)
        lex = _normalize_lexical(lex_raw) if normalize_lexical else lex_raw.astype(np.float32)

        # Штрафуем только документы, где запрос зацепился ИСКЛЮЧИТЕЛЬНО за misuse.
        # Если те же слова есть в имени/якоре/теле, документ реально про этот
        # концепт, а блок «не использовать» лишь ограничивает сценарии.
        has_main_hit = (stats["tf"] > 0).any(axis=1)
        mis_any = (
            (stats["tf_misuse"] > 0).any(axis=1)
            & ~has_main_hit
            & (similarities >= _MISUSE_SIM_MIN)
        )
        # Слабое совпадение: термин нашёлся только в имени файла/«якоре», а не в теле документа.
        weak_any = ((stats["tf"] > 0) & (stats["tf"] == stats["tf_short"])).any(axis=1)

        if enable_misuse_penalty:
            demotion = demotion * np.where(mis_any, 1.0 - float(misuse_penalty), 1.0)
        if enable_weak_match_penalty:
            demotion = demotion * np.where(weak_any, 1.0 - float(weak_match_penalty), 1.0)

        for i, doc in enumerate(docs):
            matched = sorted({*stats["in_file"][i], *stats["in_anchor"][i], *stats["in_searchable"][i]})
            if not matched and not mis_any[i] and not weak_any[i]:
                continue
            match_debug[i] = {
                "in_file": list(stats["in_file"][i]),
                "in_anchor": list(stats["in_anchor"][i]),
                "in_searchable": list(stats["in_searchable"][i]),
                "in_misuse": list(stats["in_misuse"][i]),
                "matched_unique": matched,
                "matched_frac": float(matched and len(matched) / max(1, len(kws)) or 0.0),
                "lexical": float(lex[i]),
                "demotion": float(demotion[i]),
                "misuse_hit": bool(mis_any[i]),
                "weak_hit": bool(weak_any[i]),
                "lexical_enabled": True,
            }

    scores = (similarities + float(keyword_bonus) * lex) * demotion
    return scores, similarities, lex, demotion, match_debug


def compute_scores_multilingual(
    query: str,
    docs: List[dict],
    vectors: np.ndarray,
    embedder,
    query_en: str | None = None,
    keyword_bonus: float = 0.5,
    misuse_penalty: float = 0.5,
    weak_match_penalty: float = 0.25,
    enable_misuse_penalty: bool = True,
    enable_weak_match_penalty: bool = True,
    token_vocab: set[str] | None = None,
    expanded_query: str | None = None,
    local_sources: tuple[str, ...] = ("local",),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, dict]]:
    """Мультиязычная версия compute_scores для мульти-источников.

    Схожесть для всех источников считается **одним и тем же** запросом
    (исходный/расширенный ``qx``). Перевод ``query_en`` используется только для
    лексических совпадений в нелокальных источниках: так similarity остаётся
    сравнимой между local и CKM.

    Если для группы не-local перевод отсутствует (запрос остался кириллическим),
    лексическая компонента отключается: буквальные совпадения русских слов в
    английском тексте бессмысленны.

    Лексическая компонента нормируется глобально по всем источникам, поэтому
    ``лексика 1.00`` означает лучший лексический результат среди всех
    кандидатов, а не внутри одной группы.
    """
    qx = (expanded_query if expanded_query is not None else query) or ""
    qx = qx.strip() or (query or "").strip()
    q_en = (query_en or "").strip() or qx

    n = len(docs)
    scores = np.zeros(n, dtype=np.float32)
    similarities = np.zeros(n, dtype=np.float32)
    bonus = np.zeros(n, dtype=np.float32)
    demotion = np.ones(n, dtype=np.float32)
    match_debug: Dict[int, dict] = {}
    if n == 0:
        return scores, similarities, bonus, demotion, match_debug

    # Один вектор запроса на все источники — иначе similarity несравнима.
    shared_sims: np.ndarray | None = None
    if len(vectors) == n:
        query_vec = embedder.encode([qx], normalize_embeddings=True)
        shared_sims = np.dot(vectors, query_vec.T).flatten().astype(np.float32)

    local_set = set(local_sources or ("local",))
    local_idx = [i for i, d in enumerate(docs) if (d.get("source") or "local") in local_set]
    other_idx = [i for i in range(n) if i not in set(local_idx)]

    groups: list[tuple] = []
    for grp, q_use, group_name in (
        (local_idx, qx, "local"),
        (other_idx, q_en, "other"),
    ):
        if not grp:
            continue
        grp_docs = [docs[i] for i in grp]
        grp_vecs = vectors[grp] if len(vectors) else np.array([], dtype=np.float32)
        if not len(grp_docs) or len(grp_vecs) != len(grp_docs):
            continue
        # Лексика активна, только если язык запроса совпадает с языком документов.
        lexical_ok = True if group_name == "local" else not has_cyrillic(q_use)
        sims_grp = shared_sims[grp] if shared_sims is not None else None
        _g_scores, g_simi, g_lex_raw, g_dem, g_debug = compute_scores(
            query=query,
            docs=grp_docs,
            vectors=grp_vecs,
            embedder=embedder,
            keyword_bonus=keyword_bonus,
            misuse_penalty=misuse_penalty,
            weak_match_penalty=weak_match_penalty,
            enable_misuse_penalty=enable_misuse_penalty,
            enable_weak_match_penalty=enable_weak_match_penalty,
            token_vocab=token_vocab,
            expanded_query=q_use,
            enable_lexical=lexical_ok,
            similarities=sims_grp,
            normalize_lexical=False,
        )
        groups.append((grp, group_name, lexical_ok, g_simi, g_lex_raw, g_dem, g_debug))

    global_lex_max = 0.0
    for _grp, _name, _lex_ok, _simi, g_lex_raw, _dem, _debug in groups:
        if g_lex_raw.size:
            global_lex_max = max(global_lex_max, float(np.max(g_lex_raw)))

    for grp, group_name, lexical_ok, g_simi, g_lex_raw, g_dem, g_debug in groups:
        if global_lex_max > 0:
            g_lex = (g_lex_raw / global_lex_max).astype(np.float32)
        else:
            g_lex = np.zeros_like(g_lex_raw, dtype=np.float32)
        g_scores = (g_simi + float(keyword_bonus) * g_lex) * g_dem
        for j, i in enumerate(grp):
            scores[i] = g_scores[j]
            similarities[i] = g_simi[j]
            bonus[i] = g_lex[j]
            demotion[i] = g_dem[j]
            md = g_debug.get(j)
            if md is None:
                # Лексика отключена для этой группы (нет перевода) — сохраним
                # флаг, чтобы dev-режим не показывал «лексика 0.00» как факт.
                if not lexical_ok:
                    match_debug[i] = {"lexical_enabled": False, "group": group_name}
                continue
            md = dict(md)
            md["lexical"] = float(g_lex[j])
            md["lexical_enabled"] = lexical_ok
            md["group"] = group_name
            match_debug[i] = md

    return scores, similarities, bonus, demotion, match_debug
