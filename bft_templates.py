"""Шаблоны протоколов из БФТ: выбор по типу исследования и рендер в текст для промта."""
import json
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bft_templates.json")
TEMPLATES = {k: v for k, v in json.load(open(PATH, encoding="utf-8")).items() if k != "meta"}

# study_type из тестов -> ключ шаблона (кроме ключей и label, которые ищутся сами)
ALIASES = {
    "xray": "xray_chest",
    "РГ/ФЛГ ОГК": "xray_chest",
    "ФЛГ ОГК": "xray_chest",
    "ММГ": "mammography",
}

REQUIREDNESS = {
    "required": "обязательно",
    "optional": "опционально",
    "not_specified": "обязательность не указана",
}


def find_template(study_type):
    st = (study_type or "").strip()
    if st in TEMPLATES:
        return TEMPLATES[st]
    if st in ALIASES:
        return TEMPLATES[ALIASES[st]]
    for tpl in TEMPLATES.values():
        if tpl["label"].lower() == st.lower():
            return tpl
    return None


def _parent_path(fields, i):
    """Путь родителей поля: по номеру БФТ (2 -> 2.5 -> 2.5.1), для «Описание…» — предыдущее поле."""
    no = fields[i]["bft_no"]
    path = [f["label"] for f in fields[:i]
            if no.startswith(f["bft_no"] + ".") and f["bft_no"].count(".") < no.count(".")]
    if not path and fields[i]["label"].startswith("Описание"):
        prev = [f for f in fields[:i] if f["type"] != "free_text"]
        if prev:
            path = [prev[-1]["label"]]
    return path


def _render_field(f, path):
    name = " → ".join(path + [f["label"]])
    req = (f"условно: {f['condition']}" if f["requiredness"] == "conditional" and f.get("condition")
           else REQUIREDNESS.get(f["requiredness"], f["requiredness"]))
    if f.get("allowed_values"):
        value = " | ".join(f["allowed_values"])
    elif f["type"] == "number":
        value = "число" + (f" ({f['content_hint']})" if f.get("content_hint") else "")
    else:
        value = "текст" + (f" ({f['content_hint']})" if f.get("content_hint") else "")
    line = f"- {name} [{req}]: {value}"
    extra = list(f.get("constraints", [])) + list(f.get("conditions", []))
    # заметки о погрешностях исходной таблицы модели не нужны
    extra += [n for n in f.get("notes", []) if "таблиц" not in n]
    return line + "".join(f"\n  * {e}" for e in extra)


def render_template(study_type):
    tpl = find_template(study_type)
    if tpl is None:
        return f"Шаблон протокола для «{study_type}» в БФТ не найден."
    fields = list(tpl["fields"].values())
    lines = [
        f"Шаблон протокола «{tpl['label']}» "
        f"(БФТ ДЗМ, прил. 11, разд. {tpl['bft_section']}, стр. {', '.join(map(str, tpl['source_pages']))}).",
        "Поля протокола, обязательность и допустимые значения:",
    ]
    for i, f in enumerate(fields):
        if f["type"] == "group":  # группы видны в пути дочерних полей
            continue
        lines.append(_render_field(f, _parent_path(fields, i)))
    return "\n".join(lines)
