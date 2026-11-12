"""Мелкие UI-утилиты: подсказки по коротким токенам, фильтрация документов."""

import re

import numpy as np
import streamlit as st


def short_token_hint(query_text: str, *, vocab: set[str] | None, dev_mode: bool = False) -> None:
    if not dev_mode:
        return
    q_tokens = [t.lower() for t in re.findall(r"\w+", (query_text or "").lower()) if t]
    short_tokens = [t for t in q_tokens if 2 <= len(t) <= 3]
    if not short_tokens:
        return
    v = vocab or set()
    bits = []
    for t in list(dict.fromkeys(short_tokens)):
        bits.append(f"{t}: в базе={'да' if t in v else 'нет'}")
    st.caption("Короткие токены (2–3): " + "; ".join(bits))


def filter_docs(docs_in: list[dict], vecs_in: np.ndarray, keep_fn):
    kept = [i for i, d in enumerate(docs_in or []) if keep_fn(d)]
    if not kept:
        return [], np.array([], dtype=np.float32), []
    return [docs_in[i] for i in kept], vecs_in[kept], kept