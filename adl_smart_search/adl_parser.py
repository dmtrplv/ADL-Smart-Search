import re


def get_block_by_bracket(text: str, start_index: int) -> str:
    open_bracket_pos = text.find("<", start_index)
    if open_bracket_pos == -1:
        return ""
    stack, content = 0, []
    in_quotes = False
    escape = False
    for i in range(open_bracket_pos, len(text)):
        ch = text[i]
        if escape:
            escape = False
        else:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_quotes = not in_quotes

        # Важно: внутри строковых литералов '<' и '>' могут встречаться как текст
        # (например "< 55 лет"), и не должны влиять на баланс скобок.
        if not in_quotes:
            if ch == "<":
                stack += 1
            elif ch == ">":
                stack -= 1
                if stack == 0:
                    return "".join(content[1:])

        content.append(ch)
    return ""


def get_block_by_brace(text: str, start_index: int) -> str:
    """
    Возвращает содержимое блока в фигурных скобках {...}, начиная с первой '{' после start_index.
    Аналогично get_block_by_bracket(), но для фигурных скобок.
    """
    open_pos = text.find("{", start_index)
    if open_pos == -1:
        return ""
    stack, content = 0, []
    in_quotes = False
    escape = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if escape:
            escape = False
        else:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_quotes = not in_quotes

        if not in_quotes:
            if ch == "{":
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    return "".join(content[1:])
        content.append(ch)
    return ""


def parse_ontology_terms(content: str, preferred_lang: str = "ru") -> dict:
    """
    Возвращает словарь at-кодов -> {text, desc} из ontology/term_definitions.
    Берёт preferred_lang, если нет — фоллбек на en, если и его нет — пусто.
    """
    terms_dict: dict = {}
    if "ontology" not in content:
        return terms_dict

    ont_part = content.split("ontology", 1)[-1]

    def _find_lang_marker(lang: str) -> int:
        return ont_part.find(f'["{lang}"] =')

    marker = _find_lang_marker(preferred_lang)
    if marker == -1:
        marker = _find_lang_marker("en")
    if marker == -1:
        return terms_dict

    lang_content = get_block_by_bracket(ont_part, marker)
    codes = re.findall(r'\["(at\d+\.?\d*)"\]', lang_content)
    for code in codes:
        code_pos = lang_content.find(f'["{code}"]')
        block = get_block_by_bracket(lang_content, code_pos)
        t = re.search(r"text\s*=\s*<(.*?)>", block, re.DOTALL)
        d = re.search(r"description\s*=\s*<(.*?)>", block, re.DOTALL)
        terms_dict[code] = {
            "text": t.group(1).strip().strip('"') if t else code,
            "desc": d.group(1).strip().strip('"') if d else "",
        }
    return terms_dict


def has_language(content: str, lang: str = "en") -> bool:
    """Есть ли в ontology пригодный блок term_definitions для языка lang.

    Для CKM проверяем только ``en``: если английской версии нет, архетип
    в индекс не попадает (переводы на другие языки не используем).
    """
    return bool(parse_ontology_terms(_strip_bom(content), preferred_lang=lang))


def _definition_block_names(content: str) -> list[tuple[int, str]]:
    """(индекс строки, имя) верхнеуровневых блоков definition.

    Например, для OBSERVATION это ``data``, ``state``, ``protocol``; для ACTION —
    ``description``, ``protocol``, ``ism_transition``. Нужны, чтобы структура
    в результатах не теряла названия блоков.
    """
    if "definition" not in content:
        return []
    raw_def = content.split("definition", 1)[1].split("ontology", 1)[0]
    lines = raw_def.split("\n")
    depth = 0
    out: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        code = line.split("--", 1)[0].strip()
        if depth == 1 and re.match(r"^[a-z][a-z0-9_]*\s+matches\s*\{", code):
            out.append((i, code.split()[0]))
        depth += code.count("{") - code.count("}")
    return out


def _extract_details_block(content: str) -> str:
    details_pos = content.find("details =")
    if details_pos == -1:
        return ""
    return get_block_by_bracket(content, details_pos)


def extract_details_metadata(content: str, preferred_lang: str = "ru") -> dict:
    """
    Извлекает purpose/use/keywords/misuse из description/details для языка preferred_lang (fallback en).
    Возвращает очищенные строки.
    """
    details_block = _extract_details_block(content)
    if not details_block:
        return {"purpose": "", "use": "", "keywords": "", "misuse": ""}

    lang_pos = details_block.find(f'["{preferred_lang}"] =')
    if lang_pos == -1:
        lang_pos = details_block.find('["en"] =')
    if lang_pos == -1:
        return {"purpose": "", "use": "", "keywords": "", "misuse": ""}

    lang_block = get_block_by_bracket(details_block, lang_pos)

    def _clean_text(x: str) -> str:
        if not x:
            return ""
        parts = [ln.strip() for ln in x.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        parts = [p for p in parts if p]
        return " ".join(parts).strip()

    def _get_raw(key: str) -> str:
        # Важно: не strip('"') — иначе можно сломать парсинг списков вроде keywords <"a","b",...>
        m = re.search(fr"{key}\s*=\s*<(.*?)>", lang_block, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _get_text(key: str) -> str:
        return _clean_text(_get_raw(key))

    def _get_keywords() -> str:
        raw = _get_raw("keywords")
        if not raw:
            return ""
        vals = re.findall(r'"([^"]+)"', raw)
        vals = [v.strip() for v in vals if v.strip() and v.strip() != "..."]
        return ", ".join(vals) if vals else _clean_text(raw)

    return {
        "purpose": _get_text("purpose"),
        "use": _get_text("use"),
        "keywords": _get_keywords(),
        "misuse": _get_text("misuse"),
    }


def _strip_bom(content: str) -> str:
    """CKM-mirror: файлы сконвертированы в UTF-8 with BOM — убираем BOM в начале."""
    return (content or "").lstrip("\ufeff")


def detect_content_lang(content: str, preferred_lang: str = "ru") -> str:
    """
    Фактический язык файла: preferred_lang, если его маркер есть в ontology,
    иначе 'en' (фолбэк парсера). Для файлов без ontology — original_language из шапки.
    """
    lang = (preferred_lang or "ru").strip().lower() or "ru"
    content = _strip_bom(content)
    if "ontology" in content:
        ont_part = content.split("ontology", 1)[-1]
        if f'["{lang}"]' in ont_part:
            return lang
        if '["en"]' in ont_part:
            return "en"
        return "en"
    m = re.search(r"original_language\s*=\s*<\[ISO_639-1::([a-z]{2,5})\]>", content)
    if m:
        return m.group(1).strip().lower()
    return lang


def compact_adl(content: str, preferred_lang: str = "ru") -> str:
    """Генерирует подробный текст по архетипу для LLM."""
    content = _strip_bom(content)
    terms = parse_ontology_terms(content, preferred_lang=preferred_lang)
    metadata = "МЕТАДАННЫЕ:\n"

    meta = extract_details_metadata(content, preferred_lang=preferred_lang)
    if meta.get("purpose"):
        metadata += f"ЦЕЛЬ: {meta['purpose']}\n"
    if meta.get("use"):
        metadata += f"ИСПОЛЬЗОВАНИЕ: {meta['use']}\n"
    if meta.get("keywords"):
        metadata += f"КЛЮЧЕВЫЕ СЛОВА: {meta['keywords']}\n"
    # misuse только для LLM-контекста; в retrieval (anchor/keyword) исключается в rag.py.
    if meta.get("misuse"):
        metadata += f"НЕДОПУСТИМОЕ ИСПОЛЬЗОВАНИЕ: {meta['misuse']}\n"
    metadata += "\n"

    if "definition" not in content:
        return metadata + "Структура не найдена"

    raw_def = content.split("definition", 1)[1].split("ontology", 1)[0]
    lines = raw_def.split("\n")
    block_names = dict(_definition_block_names(content))
    enriched_structure = []

    for i, line in enumerate(lines):
        if i in block_names:
            indent_n = len(line) - len(line.lstrip())
            enriched_structure.append(f"{'  ' * indent_n}{block_names[i]}:")
            continue

        code_match = re.search(r"\[(at\d+\.?\d*)\]", line)
        if not code_match:
            continue

        # RM-класс узла (ELEMENT/CLUSTER/ITEM_TREE/EVENT/...), если удаётся вытащить из definition.
        # CKM: слоты выглядят как "allow_archetype CLUSTER[atNNNN] matches {...}" — распознаём RM-класс после префикса.
        rm_m = re.match(r"^\s*(?:allow_archetype\s+)?([A-Z][A-Z0-9_]+)\s*\[at\d+\.?\d*\]", line)
        rm_class = rm_m.group(1) if rm_m else ""

        code = code_match.group(1)
        term = terms.get(code, {"text": code, "desc": ""})

        # Контекст узла: берём строки до следующего узла на той же/меньшей вложенности.
        # Важно: иначе атрибуты контейнера (например `name matches { ... }`) могут "прилипнуть"
        # к предыдущему ELEMENT и дать ложные типы/значения (internal_codes).
        indent0 = len(line) - len(line.lstrip(" "))
        context_lines = []
        for j in range(i + 1, len(lines)):
            ln = lines[j]
            # Следующий узел archetype definition обычно выглядит как "TYPE[atNNNN] ..."
            if re.search(r"^\s*(?:allow_archetype\s+)?[A-Z][A-Z0-9_]*\s*\[at\d+\.?\d*\]", ln):
                indent_j = len(ln) - len(ln.lstrip(" "))
                if indent_j <= indent0:
                    break
            context_lines.append(ln)
            # safety: ограничим размер контекста
            if len(context_lines) >= 120:
                break
        context_text = "\n".join(context_lines)

        # Для ELEMENT используем только value-block, если он явно присутствует,
        # чтобы не захватывать `name matches ...`/`archetype_node_id ...` и т.п.
        value_context = context_text
        if rm_class == "ELEMENT":
            vm_pos = context_text.find("value matches")
            if vm_pos != -1:
                # вырезаем именно блок фигурных скобок после "value matches"
                vb = get_block_by_brace(context_text, vm_pos)
                if vb:
                    value_context = vb

        dtype_source = value_context if rm_class == "ELEMENT" else context_text
        dtypes = re.findall(r"(DV_[A-Z_]+|C_DV_QUANTITY)", dtype_source)
        clean_types = sorted(list(set([t.replace("C_DV_", "DV_") for t in dtypes])))
        # Всегда сохраняем RM-класс узла (ELEMENT/CLUSTER/SLOT/...), чтобы, например, SLOT не терялся.
        # Далее добавляем DV_* типы (если они есть) для листовых значений.
        dtype_tokens = []
        # ELEMENT выводим только как тип данных (DV_*), без слова ELEMENT (по требованию).
        if rm_class and not (rm_class == "ELEMENT" and clean_types):
            dtype_tokens.append(rm_class)
        dtype_tokens.extend(clean_types)
        dtype_str = " | ".join(dtype_tokens) if dtype_tokens else "ELEMENT"

        # Эвристика для DV_ORDINAL, если тип не указан явно:
        # в некоторых архетипах ordinal задаётся как "1|[local::atXXXX], 2|[local::atYYYY]"
        if "DV_ORDINAL" not in dtype_str and re.search(r"\b\d+\s*\|\s*\[local::\s*at\d+\.?\d*\s*\]", value_context):
            dtype_str = (dtype_str + " | DV_ORDINAL") if dtype_str else "DV_ORDINAL"

        occ_m = re.search(r"occurrences matches \{(.*?)\}", line)
        occ = f" [{occ_m.group(1)}]" if occ_m else ""

        params = []
        if "DV_QUANTITY" in dtype_str:
            q_matches = re.findall(
                r'units = <"(.*?)">(.*?magnitude = <\|(.*?)\|>(.*?precision = <(\d+)>)?)?',
                value_context,
                re.DOTALL,
            )
            for q in q_matches:
                u, _, mag, _, prec = q
                mag_str = f": {mag.replace('..', '—')}" if mag else ""
                prec_str = f" (точн. {prec})" if prec else ""
                params.append(f"Число ({u}){mag_str}{prec_str}")

        if any(x in dtype_str for x in ["DV_ORDINAL", "DV_CODED_TEXT"]):
            # Извлекаем значения только из релевантного value-контекста.
            # Это предотвращает подтягивание value set из `name matches ...` соседних/родительских узлов.
            val_text = value_context

            # Специальный разбор для ordinal в виде "N|[local::atXXXX]"
            ordinal_pairs = re.findall(r"(\d+)\s*\|\s*\[local::\s*(at\d+\.?\d*)\s*\]", val_text)
            if ordinal_pairs:
                for n, v_code in ordinal_pairs:
                    if v_code in terms:
                        params.append(f"{n}: {terms[v_code]['text']}")
            else:
                local_codes: list[str] = []
                if "DV_CODED_TEXT" in dtype_str:
                    # DV_CODED_TEXT: value set обычно объявлен в defining_code как [local:: atXXXX, ...]
                    defining_code_block = ""
                    dc_pos = val_text.find("defining_code")
                    if dc_pos != -1:
                        defining_code_block = get_block_by_brace(val_text, dc_pos) or ""
                    src = defining_code_block or val_text
                    local_blocks = re.findall(r"\[local::([\s\S]*?)\]", src, flags=re.DOTALL)
                    for b in local_blocks:
                        local_codes.extend(re.findall(r"\bat\d+\.?\d*\b", b))

                # Запасной вариант: коды могут встречаться в квадратных скобках [atXXXX]
                bracket_codes = re.findall(r"\[(at\d+\.?\d*)\]", val_text)

                val_codes = list(dict.fromkeys(local_codes + bracket_codes))
                for v_code in val_codes:
                    if v_code == code:
                        continue
                    if v_code not in terms:
                        continue
                    # Для DV_ORDINAL иногда рядом встречается `value matches {N}`
                    w_m = re.search(
                        rf"\[{re.escape(v_code)}\].*?value matches \{{(\d+)\}}", val_text, re.DOTALL
                    )
                    prefix = f"{w_m.group(1)}: " if w_m else ""
                    params.append(f"{prefix}{terms[v_code]['text']}")

        indent = "  " * (len(line) - len(line.lstrip()))
        res = f"{indent}- {term['text']} ({code}) | {dtype_str}{occ}"
        if params:
            res += f"\n{indent}  Значения: {'; '.join(list(dict.fromkeys(params)))}"
        if term["desc"] and term["desc"] not in [term["text"], "@ internal @", "*"]:
            res += f"\n{indent}  Описание: {term['desc']}"
        enriched_structure.append(res)

    return metadata + "СТРУКТУРА:\n" + "\n".join(enriched_structure[:350])


def create_search_anchor(content: str, preferred_lang: str = "ru") -> str:
    """Создает концентрированный текст для векторного поиска."""
    content = _strip_bom(content)
    terms = parse_ontology_terms(content, preferred_lang=preferred_lang)
    main_name = terms.get("at0000", {}).get("text", "Архетип")

    all_fields = []
    for k, v in terms.items():
        if not k.startswith("at"):
            continue
        text = (v.get("text") or "").strip()
        desc = (v.get("desc") or "").strip()
        if text:
            all_fields.append(text)
        if desc and desc not in [text, "@ internal @", "*"]:
            all_fields.append(desc)

    fields_str = ", ".join(all_fields[:30])
    anchor = f"{main_name} {main_name} {main_name} | Поля: {fields_str}"
    blocks = [name for _, name in _definition_block_names(content)]
    if blocks:
        anchor += " | Блоки: " + ", ".join(dict.fromkeys(blocks))
    return anchor


def extract_field_chunks_from_compact(
    compact_text: str,
    file_name: str,
    max_fields: int = 800,
) -> list[dict]:
    """
    Достаёт "поля" из результата compact_adl (блок СТРУКТУРА).
    Возвращает список dict:
      - file, at_code, field, text
    """
    if not compact_text:
        return []

    lines_all = compact_text.splitlines()
    # Старт после заголовка структуры
    start = 0
    for i, ln in enumerate(lines_all):
        if "СТРУКТУРА" in ln:
            start = i + 1
            break
    lines = lines_all[start:] if start else lines_all

    # Возвращаем ТОЛЬКО leaf-поля (например ELEMENT и прочие "значимые" узлы),
    # но контейнеры используем для восстановления иерархии (stack/path/parent_*).
    items: list[dict] = []
    # Для восстановления иерархии будем поддерживать стек контейнеров по "глубине".
    # Глубину оцениваем по числу ведущих пробелов в строке "- ...".
    # В compact_adl отступ формируется как "  " * (кол-во ведущих пробелов в исходном ADL),
    # поэтому абсолютное значение завышено, но относительная иерархия сохраняется.
    stack: list[dict] = []  # {"depth": int, "field": str, "at_code": str, "rm_type": str, "desc": str, "path": str}
    i = 0
    while i < len(lines) and len(items) < max_fields:
        ln = lines[i]
        block_m = re.match(r"^\s*([a-z][a-z0-9_]*):\s*$", ln)
        if block_m:
            leading = len(ln) - len(ln.lstrip(" "))
            while stack and stack[-1]["depth"] >= leading:
                stack.pop()
            name = block_m.group(1)
            parent = stack[-1] if stack else None
            path = f"{parent['path']} > {name}" if parent else name
            stack.append(
                {
                    "depth": leading,
                    "field": name,
                    "at_code": "",
                    "rm_type": "",
                    "desc": "",
                    "path": path,
                }
            )
            i += 1
            continue

        m = re.search(r"-\s+(.*?)\s+\((at\d+\.?\d*)\)\s+\|\s+(.*)$", ln)
        if not m:
            i += 1
            continue

        leading = len(ln) - len(ln.lstrip(" "))
        depth = leading

        field = (m.group(1) or "").strip()
        at_code = (m.group(2) or "").strip()
        type_part = (m.group(3) or "").strip()

        # occurrences в компактном тексте выглядят как " ... [0..1]" в конце строки
        occ = ""
        occ_m = re.search(r"\s+\[([^\]]+)\]\s*$", type_part)
        if occ_m:
            occ = occ_m.group(1).strip()
            type_part = type_part[: occ_m.start()].strip()

        # rm_type/value_types/node_kind
        type_tokens = [t.strip() for t in type_part.split("|") if t.strip()]
        value_types = [t for t in type_tokens if t.startswith("DV_")]
        rm_type = type_tokens[0] if type_tokens else ""
        # SLOT должен оставаться SLOT, даже если где-то в строке встретились DV_* токены.
        if rm_type != "SLOT" and value_types:
            rm_type = "ELEMENT"

        container_types = {
            "CLUSTER",
            "ITEM_TREE",
            "ITEM_LIST",
            "ITEM_TABLE",
            "ITEM_SINGLE",
            "EVENT",
            "POINT_EVENT",
            "INTERVAL_EVENT",
            "HISTORY",
            "SECTION",
            "ACTIVITY",
            "INSTRUCTION",
            "ACTION",
            "OBSERVATION",
            "EVALUATION",
            "ADMIN_ENTRY",
            "COMPOSITION",
            "SLOT",
        }
        node_kind = "container" if (rm_type in container_types) else "value_leaf"
        if rm_type == "ELEMENT":
            node_kind = "value_leaf"

        block_lines = [ln]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if re.search(r"-\s+.*\((at\d+\.?\d*)\)\s+\|", nxt):
                break
            if re.match(r"^\s*([a-z][a-z0-9_]*):\s*$", nxt):
                break
            low = nxt.lower().strip()
            if low.startswith("описание:") or low.startswith("значения:") or "dv_" in low:
                block_lines.append(nxt)
            j += 1

        text = "\n".join(block_lines).strip()

        # Быстрые признаки "насколько поле самодостаточно"
        has_desc = any(bl.strip().lower().startswith("описание:") for bl in block_lines[1:])
        own_desc = ""
        for bl in block_lines[1:]:
            s = bl.strip()
            if s.lower().startswith("описание:"):
                own_desc = s[len("Описание:") :].strip()
                break

        # Обновим стек: снимаем всё не-родительское
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()

        parent = stack[-1] if stack else None
        path_bits = [x["field"] for x in stack] + [field]
        path = " > ".join([p for p in path_bits if p])

        # Для контейнеров полезно хранить описание, чтобы "подтягивать" его к листьям.
        parent_desc = (parent.get("desc") or "") if parent else ""
        parent_field = (parent.get("field") or "") if parent else ""
        parent_at_code = (parent.get("at_code") or "") if parent else ""
        parent_rm_type = (parent.get("rm_type") or "") if parent else ""
        parent_path = (parent.get("path") or "") if parent else ""

        if node_kind != "container":
            items.append(
                {
                    "file": file_name,
                    "at_code": at_code,
                    "field": field,
                    "rm_type": rm_type,
                    "value_types": value_types,
                    "occurrences": occ,
                    "node_kind": node_kind,
                    "depth": depth,
                    "path": path,
                    "parent_field": parent_field,
                    "parent_at_code": parent_at_code,
                    "parent_rm_type": parent_rm_type,
                    "parent_desc": parent_desc,
                    "parent_path": parent_path,
                    "has_desc": has_desc,
                    "text": text,
                }
            )

        # Если это контейнер — положим в стек как потенциального родителя для следующих узлов.
        if node_kind == "container":
            stack.append(
                {
                    "depth": depth,
                    "field": field,
                    "at_code": at_code,
                    "rm_type": rm_type,
                    "desc": own_desc,
                    "path": path,
                }
            )
        i = j

    return items

