"""Backward-compatible facade: re-exports from sub-modules + shared utilities."""

import os
import re
from typing import List

import numpy as np

# ── Re-export everything from sub-modules so existing `from .rag import X` works ──
from .context_builder import (  # noqa: F401
    _display_file_name,
    _extract_metadata_lines,
    _extract_relevant_snippets,
    _format_archetype_meta_block,
    _misuse_for_field,
    build_compact_context_for_archetypes,
    build_compact_context_for_fields,
    context_tokens_to_max_chars,
    context_window_to_max_chars,
)
from .prompts import (  # noqa: F401
    PROMPT_KEYS,
    PROMPT_LABELS,
    PROMPT_PART_KEYS,
    PromptConfigError,
    compose_system_prompt,
    load_prompts,
    prompts_path,
    save_prompts,
)
from .scoring import (  # noqa: F401
    _extract_keywords_for_bonus,
    _full_context_without_misuse,
    _misuse_text_for_doc,
    _search_body_for_doc,
    _search_haystack_for_doc,
    _stem_prefix,
    _strip_known_misuse,
    compute_scores,
    compute_scores_multilingual,
    has_cyrillic,
    is_low_confidence,
    pure_floor_by_doc_lang,
)

# ── Shared utilities (not specific to scoring / context / prompts) ──

_VERSION_STEM_RE = re.compile(r"^(.+)\.v(\d+)$", flags=re.IGNORECASE)


def _adl_stem_lower(file_path: str) -> str:
    fn = os.path.basename(file_path or "")
    if fn.lower().endswith(".adl"):
        fn = fn[:-4]
    return (fn or "").lower()


def make_rag_selection_key(indices: List[int]) -> str:
    return ",".join(str(int(i)) for i in sorted(indices))


def simi_family_key_from_file(file_path: str) -> tuple[str, bool]:
    stem_l = _adl_stem_lower(file_path)
    mver = _VERSION_STEM_RE.match(stem_l)
    if mver:
        stem_no_ver = mver.group(1).lower()
    else:
        stem_no_ver = stem_l

    suf = "-simi"
    is_simi = stem_no_ver.endswith(suf) and len(stem_no_ver) > len(suf)
    if is_simi:
        return (stem_no_ver[: -len(suf)], True)
    return (stem_no_ver, False)


def version_logical_key_and_num(file_path: str) -> tuple[str, int]:
    stem_l = _adl_stem_lower(file_path)
    m = _VERSION_STEM_RE.match(stem_l)
    if m:
        return (m.group(1).lower(), int(m.group(2)))
    return (stem_l, 0)


def reorder_picked_with_simi_preference(
    picked: List[int],
    docs: List[dict],
    scores: np.ndarray,
    *,
    eps: float = 1e-4,
) -> tuple[List[int], dict[int, float]]:
    if not picked:
        return [], {}
    arr = np.asarray(scores, dtype=np.float64)
    picked_i = [int(i) for i in picked]
    sort_score: dict[int, float] = {i: float(arr[i]) for i in picked_i}

    eps_f = float(eps)
    if not np.isfinite(eps_f) or eps_f <= 0:
        eps_f = 1e-4

    # --- SIMI: родитель vs специализированный (-simi)
    by_fk: dict[str, list[tuple[int, bool]]] = {}
    for i in picked_i:
        fk, is_simi = simi_family_key_from_file(str(docs[i].get("file") or ""))
        by_fk.setdefault(fk, []).append((i, is_simi))

    for _fk, members in by_fk.items():
        plains = [i for i, is_s in members if not is_s]
        simis = [i for i, is_s in members if is_s]
        if not plains or not simis:
            continue
        m_plain = max(float(arr[i]) for i in plains)
        simis_sorted = sorted(simis, key=lambda j: (-float(arr[j]), j))
        n = len(simis_sorted)
        for p, idx in enumerate(simis_sorted):
            bump = m_plain + eps_f * float(n - p)
            sort_score[idx] = max(float(arr[idx]), bump)

    # --- Версии: .v2 выше .v1 при том же логическом ключе
    by_vk: dict[str, list[tuple[int, int]]] = {}
    for i in picked_i:
        vk, vnum = version_logical_key_and_num(str(docs[i].get("file") or ""))
        by_vk.setdefault(vk, []).append((i, vnum))

    micro = eps_f
    for _vk, members in by_vk.items():
        vnums = {v for _, v in members}
        if len(vnums) < 2:
            continue
        max_v = max(vnums)
        min_v = min(vnums)
        if max_v == min_v:
            continue
        for v in range(max_v, min_v, -1):
            ids_this = [i for i, vv in members if vv == v]
            ids_lower = [i for i, vv in members if vv < v]
            if not ids_this or not ids_lower:
                continue
            m_lower = max(sort_score[i] for i in ids_lower)
            tier_sorted = sorted(ids_this, key=lambda j: (-sort_score[j], j))
            n_t = len(tier_sorted)
            for p, idx in enumerate(tier_sorted):
                bump = m_lower + micro * float(n_t - p)
                sort_score[idx] = max(sort_score[idx], bump)

    pos = {i: k for k, i in enumerate(picked_i)}
    ordered = sorted(picked_i, key=lambda i: (-sort_score[i], pos[i]))
    return ordered, sort_score


def build_token_vocab(docs: List[dict], *, field: str = "anchor", min_len: int = 2, max_len: int = 24) -> set[str]:
    vocab: set[str] = set()
    for d in docs or []:
        txt = (d.get(field) or "")
        if not txt:
            continue
        for t in re.findall(r"\w+", txt.lower()):
            if min_len <= len(t) <= max_len:
                vocab.add(t)
    return vocab


def expand_query(query: str, expansions: dict | None = None) -> str:
    q = (query or "").strip()
    if not q:
        return q
    exp = expansions or {}
    tokens = [t.lower() for t in re.findall(r"\w+", q.lower()) if t]
    add: list[str] = []
    for t in tokens:
        v = exp.get(t)
        if not v:
            continue
        if isinstance(v, str):
            add.append(v)
        elif isinstance(v, list):
            add.extend([str(x) for x in v if str(x).strip()])
    add = [x.strip() for x in add if x and x.strip()]
    if not add:
        return q
    return q + " | " + ", ".join(list(dict.fromkeys(add)))


def expand_query_phrases(
    query: str,
    phrase_expansions: dict[str, list[str]] | None = None,
    *,
    max_add: int = 12,
) -> str:
    q = (query or "").strip()
    if not q:
        return q
    pm = phrase_expansions or {}
    if not pm:
        return q

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", (s or "").lower())).strip()

    base = _norm(q.split("|", 1)[0])
    if not base:
        return q
    base_pad = f" {base} "

    add: list[str] = []
    for k, vals in pm.items():
        kk = _norm(k or "")
        if not kk:
            continue
        if f" {kk} " not in base_pad:
            continue
        if isinstance(vals, list):
            add.extend([str(x) for x in vals if str(x).strip()])
        elif isinstance(vals, str) and vals.strip():
            add.append(vals.strip())

    add = [x.strip() for x in add if x and x.strip()]
    if not add:
        return q

    uniq = list(dict.fromkeys(add))
    if isinstance(max_add, int) and max_add > 0:
        uniq = uniq[:max_add]

    if "|" in q:
        return q + ", " + ", ".join(uniq)
    return q + " | " + ", ".join(uniq)
