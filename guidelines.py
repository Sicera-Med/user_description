"""Справочник «находка → действие» из официальных документов: выбор по типу исследования и рендер в промт.

Данные — папка guidelines/: sources.json (документы) + по файлу на тип исследования
(ct_chest.json, mammography.json, xray_chest.json, ct_head.json). Имя файла = ключ типа.
"""
import glob
import json
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guidelines")
SOURCES = json.load(open(os.path.join(DIR, "sources.json"), encoding="utf-8"))["sources"]

GUIDELINES, ALIASES = {}, {}  # ключ типа -> записи; синоним типа -> ключ
for path in sorted(glob.glob(os.path.join(DIR, "*.json"))):
    key = os.path.splitext(os.path.basename(path))[0]
    if key == "sources":
        continue
    data = json.load(open(path, encoding="utf-8"))
    GUIDELINES[key] = data["entries"]
    for name in [key, data.get("label"), *data.get("aliases", [])]:
        if name:
            ALIASES[name.lower()] = key

_ids = [e["id"] for entries in GUIDELINES.values() for e in entries]
_dups = sorted({i for i in _ids if _ids.count(i) > 1})
if _dups:  # по id подставляется источник со страницей — id должны быть уникальны во всех файлах
    raise ValueError(f"guidelines: повторяющиеся id: {_dups}")
_unknown = sorted({e["source"]["doc"] for entries in GUIDELINES.values() for e in entries} - SOURCES.keys())
if _unknown:
    raise ValueError(f"guidelines: документов нет в sources.json: {_unknown}")


def guideline_key(study_type, template_key=None):
    """Ключ справочника: ключ шаблона БФТ, если для него есть файл, иначе по названию/синониму (КТ ГМ без шаблона БФТ)."""
    if template_key in GUIDELINES:
        return template_key
    return ALIASES.get((study_type or "").strip().lower())


def entries_for(key):
    return GUIDELINES.get(key, [])


def entry_ids(key):
    return {e["id"] for e in entries_for(key)}


def _action(a):
    what = a["code"] or "без кода"
    line = f"{a['type']}: {what} — {a['note']}"
    if a.get("timing"):
        line += f"; срок: {a['timing']}"
    if a.get("condition"):
        line += f"; условие: {a['condition']}"
    return line


def render_guidelines(key):
    """Текст для промта: только записи этого типа исследования, сгруппированные по документу."""
    entries = entries_for(key)
    if not entries:
        return "Справочника по этому типу исследования нет: все действия помечай source = null."
    lines = ["Справочник «находка → действие» из официальных документов. id записи в [] — значение source."]
    for doc in dict.fromkeys(e["source"]["doc"] for e in entries):
        group = [e for e in entries if e["source"]["doc"] == doc]
        src = SOURCES[doc]
        lines.append(f"\n== {src['title']} ({src['org']}, {src['year']})")
        for extra in dict.fromkeys(x for e in group for x in (e.get("scope"), e.get("not_in_source")) if x):
            lines.append(f"! {extra}")
        for e in group:
            lines.append(f"- [{e['id']}] {e['finding']} (стр. {e['source']['pages']})")
            lines += [f"    • {_action(a)}" for a in e["actions"]]
            lines += [f"    ! {x}" for x in [e.get("note"), *e.get("do_not", [])] if x]
    return "\n".join(lines)


def cite(entry_id):
    """Ссылка по id записи: документ, организация, год, страницы. Не найдено — None."""
    for entries in GUIDELINES.values():
        for e in entries:
            if e["id"] == entry_id:
                src = SOURCES[e["source"]["doc"]]
                return {
                    "document": src["title"],
                    "organization": src["org"],
                    "year": src["year"],
                    "pages": e["source"]["pages"],
                    "file": src["file"],
                    "verified_by": e["verified_by"],
                    "text": f"{src['title']} ({src['org']}, {src['year']}), стр. {e['source']['pages']}",
                }
    return None
