"""Build compact context for LLM from archetype/field documents."""

import os
import re
from typing import Dict, List

import numpy as np

_MISUSE_LABEL = "НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ"
_DEFAULT_CONTEXT_WINDOW = 8192  # tokens, conservative default for Ollama / unknown models
_LOCAL_SOURCES = ("local",)
_VECTOR_SNIPPET_TOP_K = 15
# Символов на токен. Для русского текста (GigaChat, Qwen) — ~2.5–3,
# поэтому 5.5 (как для английского) завышало бы бюджет почти вдвое.
_CHARS_PER_TOKEN = 3.0


def context_window_to_max_chars(context_window: int) -> int:
    """
    Convert a model context window (in tokens) to max_chars for context builders.

    Heuristic: ~3.0 chars per token (Russian/English mix), reserve 15% for
    system prompt + user question + template overhead.
    Floor of 4000 chars ensures a usable context even for very small windows.
    """
    cw = max(1, int(context_window or _DEFAULT_CONTEXT_WINDOW))
    return max(4000, int(cw * _CHARS_PER_TOKEN * 0.85))


def context_tokens_to_max_chars(context_tokens: int) -> int:
    """Бюджет токенов именно на ADL-контекст → максимум символов.

    В отличие от ``context_window_to_max_chars`` здесь не резервируем 15%:
    бюджет задаётся уже с учётом системного промпта, истории и ответа.
    """
    tokens = max(1000, int(context_tokens or 10000))
    return max(4000, int(tokens * _CHARS_PER_TOKEN))


def _display_file_name(path: str) -> str:
    fn = os.path.basename(path or "")
    if fn.lower().endswith(".adl"):
        fn = fn[:-4]
    return fn or (path or "")


def _extract_metadata_lines(full_context: str) -> tuple[str, str, str]:
    if not full_context:
        return "", "", ""
    purpose = ""
    use = ""
    misuse = ""
    for ln in full_context.splitlines():
        s = ln.strip()
        if s.startswith("ЦЕЛЬ:"):
            purpose = s[len("ЦЕЛЬ:") :].strip()
        elif s.startswith("ИСПОЛЬЗОВАНИЕ:"):
            use = s[len("ИСПОЛЬЗОВАНИЕ:") :].strip()
        elif s.startswith(f"{_MISUSE_LABEL}:"):
            misuse = s[len(f"{_MISUSE_LABEL}:") :].strip()
        if "СТРУКТУРА" in s:
            break
    return purpose, use, misuse


def _format_archetype_meta_block(*, purpose: str, use: str, misuse: str) -> str:
    lines: List[str] = []
    if purpose:
        lines.append(f"ЦЕЛЬ: {purpose}")
    if use:
        lines.append(f"ИСПОЛЬЗОВАНИЕ: {use}")
    if misuse:
        lines.append(f"{_MISUSE_LABEL}: {misuse}")
    return ("\n".join(lines) + "\n") if lines else ""


def _keywords_with_synonyms(query: str) -> List[str]:
    q = (query or "").lower()
    base = [w for w in re.findall(r"\w+", q) if len(w) > 2]
    kw = set(base)

    if any(x in kw for x in ["пульс", "чсс", "heart", "rate", "pulse"]):
        kw.update(["пульс", "чсс", "heart", "rate", "pulse", "hr", "beat"])

    if any(x in kw for x in ["ивл", "ventilation", "ventilator", "respirator", "вентиляц"]):
        kw.update(["ивл", "ventilation", "ventilator", "respirator", "вентиляция", "дыхание", "resp"])

    if any(x in kw for x in ["одышк", "dyspnea", "breathless", "дых"]):
        kw.update(["одышка", "dyspnea", "breathlessness", "дыхание", "respiratory", "шкала", "степень"])

    stop = {"найди", "архетип", "архетипы", "поле", "поля", "для", "моделирования", "предложи", "пациента"}
    kw = {w for w in kw if len(w) > 2 and w not in stop}
    return sorted(kw)


def _structure_lines(full_context: str) -> List[str]:
    """Строки СТРУКТУРЫ (всё после заголовка «СТРУКТУРА:»)."""
    if not full_context:
        return []
    lines_all = full_context.splitlines()
    start = 0
    for i, ln in enumerate(lines_all):
        if "СТРУКТУРА" in ln:
            start = i + 1
            break
    return lines_all[start:] if start else lines_all


def _split_full_context_blocks(lines: List[str]) -> List[List[str]]:
    """
    Делит строки СТРУКТУРЫ на самодостаточные блоки.

    Блок — строка `- <имя> (atNNNN) | <type>` плюс идущие за ней строки
    «Описание:»/«Значения:»/dv_* до следующего заголовка at-блока.
    """
    blocks: List[List[str]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith("-") and re.search(r"\(at\d+\.?\d*\)", ln):
            block_lines = [ln]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.lstrip().startswith("-") and re.search(r"\(at\d+\.?\d*\)", nxt):
                    break
                low = nxt.lower().strip()
                if low.startswith("описание:") or low.startswith("значения:") or "dv_" in low:
                    block_lines.append(nxt)
                j += 1
            blocks.append(block_lines)
            i = j
        else:
            i += 1
    return blocks


def _select_blocks_lexical(blocks: List[List[str]], kw_l: List[str]) -> List[List[str]]:
    """Лексический отбор (по наличию ключевых слов) — использовался всегда и остаётся фолбэком."""
    def block_score(bl: List[str]) -> int:
        text = "\n".join(bl).lower()
        s = sum(1 for k in kw_l if k and k in text)
        if "значения:" in text:
            s += 2
        if "dv_coded_text" in text:
            s += 1
        return s

    scored_blocks = sorted(((block_score(b), b) for b in blocks), key=lambda x: -x[0])
    return [b for s, b in scored_blocks if s > 0][:25] or [b for _, b in scored_blocks[:15]]


def _select_blocks_by_vector(
    blocks: List[List[str]],
    query_text: str,
    embedder,
    top_k: int = _VECTOR_SNIPPET_TOP_K,
) -> List[List[str]] | None:
    """
    Векторный отбор топ-K блоков по косинусной близости к запросу.

    Возвращает блоки в порядке исходного документа. None — эмбеддинг не удался,
    в этом случае вызывающий использует лексический фолбэк.
    """
    if not blocks:
        return None
    try:
        block_texts = ["\n".join(b) for b in blocks]
        vecs = embedder.encode([(query_text or "").strip()] + block_texts, normalize_embeddings=True)
        if vecs is None or len(vecs) != len(block_texts) + 1:
            return None
        qv = np.asarray(vecs[0], dtype=np.float32)
        bv = np.asarray(vecs[1:], dtype=np.float32)
        if qv.size == 0 or bv.shape[0] != len(block_texts) or bv.shape[1] != qv.size:
            return None
        sims = np.dot(bv, qv)
        count = max(1, min(len(blocks), int(top_k or _VECTOR_SNIPPET_TOP_K)))
        order = np.argsort(sims)[::-1]
        picked = sorted([int(i) for i in order[:count]])
        return [blocks[i] for i in picked]
    except Exception:
        return None


def _query_text_for_doc(
    doc: dict,
    query: str,
    query_en: str,
    expanded_query: str | None,
    local_sources: tuple[str, ...] = _LOCAL_SOURCES,
) -> str:
    """
    Близко к маршрутизации в compute_scores_multilingual: локальные документы скорят
    расширенным русским запросом, документы CKM/прочих источников — английским переводом.
    """
    qx = (expanded_query if expanded_query is not None else query) or ""
    qx = qx.strip() or (query or "").strip()
    q_en = (query_en or "").strip() or qx
    if (doc.get("source") or "local") in (local_sources or _LOCAL_SOURCES):
        return qx
    return q_en


def _extract_relevant_snippets(
    full_context: str,
    keywords: List[str],
    max_lines: int = 80,
    *,
    embedder=None,
    query_text: str | None = None,
    vector_top_k: int = _VECTOR_SNIPPET_TOP_K,
) -> str:
    if not full_context:
        return ""

    kw_l = [k.lower() for k in (keywords or [])]
    lines = _structure_lines(full_context)
    blocks = _split_full_context_blocks(lines)
    if not blocks:
        return "\n".join(lines[:max_lines])

    picked_blocks = None
    if embedder is not None and query_text:
        picked_blocks = _select_blocks_by_vector(blocks, query_text, embedder, top_k=vector_top_k)
    if picked_blocks is None:
        picked_blocks = _select_blocks_lexical(blocks, kw_l)

    out_lines = []
    for b in picked_blocks:
        out_lines.extend(b)
    return "\n".join(out_lines[:max_lines])


def build_compact_context_for_archetypes(
    query: str,
    selected_docs: List[dict],
    *,
    include_metadata: bool = True,
    max_fields_lines: int = 120,
    max_chars: int = 22000,
    embedder=None,
    query_en: str = "",
    expanded_query: str | None = None,
    local_sources: tuple[str, ...] = _LOCAL_SOURCES,
    vector_top_k: int = _VECTOR_SNIPPET_TOP_K,
    max_documents: int | None = None,
) -> str:
    keywords = _keywords_with_synonyms(query)
    out = []
    budget = max(4000, int(max_chars))

    docs = list(selected_docs or [])
    if max_documents and max_documents > 0:
        # selected_docs идут в порядке выдачи, поэтому срез = самые релевантные.
        docs = docs[: int(max_documents)]

    for d in docs:
        fn = _display_file_name(d.get("file", ""))
        full_ctx = d.get("full_context", "") or ""
        purpose, use, misuse = _extract_metadata_lines(full_ctx)
        query_text = _query_text_for_doc(d, query, query_en, expanded_query, local_sources)
        snippet = _extract_relevant_snippets(
            full_ctx,
            keywords=keywords,
            max_lines=max_fields_lines,
            embedder=embedder,
            query_text=query_text,
            vector_top_k=vector_top_k,
        )

        meta_block = ""
        if include_metadata:
            meta_block = _format_archetype_meta_block(purpose=purpose, use=use, misuse=misuse)

        part = f"--- АРХЕТИП: {fn} ---\n{meta_block}{snippet}\n"
        if len(part) <= budget:
            out.append(part)
            budget -= len(part)
        else:
            if budget > 500:
                out.append(part[:budget])
            break

    return "\n".join(out)


def _misuse_for_field(d: dict, archetype_docs_by_file: Dict[str, dict] | None = None) -> str:
    misuse = (d.get("misuse") or "").strip()
    if misuse:
        return misuse
    arch = (archetype_docs_by_file or {}).get(d.get("file") or "")
    if arch:
        _, _, misuse_from_arch = _extract_metadata_lines(arch.get("full_context") or "")
        return misuse_from_arch
    return ""


def build_compact_context_for_fields(
    query: str,
    selected_fields: List[dict],
    *,
    max_items: int = 80,
    max_chars: int = 22000,
    include_containers: bool = False,
    archetype_docs_by_file: Dict[str, dict] | None = None,
) -> str:
    budget = max(3000, int(max_chars))
    out = []
    picked = []
    for d in selected_fields:
        if include_containers:
            picked.append(d)
        else:
            if (d.get("node_kind") or "") == "container":
                continue
            picked.append(d)
        if len(picked) >= max_items:
            break

    seen_archetype_meta: set[str] = set()
    for d in picked:
        fn = _display_file_name(d.get("file", ""))
        field = d.get("field", "")
        at_code = d.get("at_code", "")
        text = (d.get("text") or "").strip()
        rm_type = d.get("rm_type") or ""
        path = d.get("path") or ""

        file_key = d.get("file", "") or fn
        if file_key not in seen_archetype_meta:
            seen_archetype_meta.add(file_key)
            misuse = _misuse_for_field(d, archetype_docs_by_file)
            if misuse:
                archetype_meta = f"--- АРХЕТИП: {fn} ---\n{_MISUSE_LABEL}: {misuse}\n"
                if len(archetype_meta) <= budget:
                    out.append(archetype_meta)
                    budget -= len(archetype_meta)
                elif budget > 500:
                    out.append(archetype_meta[:budget])
                    budget = 0

        extra = ""
        if not include_containers and (d.get("node_kind") != "container"):
            has_desc = bool(d.get("has_desc"))
            if not has_desc:
                pf = (d.get("parent_field") or "").strip()
                pa = (d.get("parent_at_code") or "").strip()
                pr = (d.get("parent_rm_type") or "").strip()
                pd = (d.get("parent_desc") or "").strip()
                pp = (d.get("parent_path") or "").strip()
                if pf or pp:
                    head = pp or pf
                    meta = []
                    if pa:
                        meta.append(pa)
                    if pr:
                        meta.append(pr)
                    meta_s = f" ({', '.join(meta)})" if meta else ""
                    if pd:
                        extra = f"\nКонтейнер: {head}{meta_s} — {pd}\n"
                    else:
                        extra = f"\nКонтейнер: {head}{meta_s}\n"

        title = path if path else field
        part = f"--- ПОЛЕ: {fn} — {title} ({at_code}) [{rm_type}] ---\n{text}{extra}\n"
        if len(part) <= budget:
            out.append(part)
            budget -= len(part)
        else:
            if budget > 500:
                out.append(part[:budget])
            break
    return "\n".join(out)
