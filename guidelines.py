"""Справочник «находка → действие» из официальных документов: выбор по типу исследования и рендер в промт.

Данные — папка guidelines/: sources.json (документы) + по файлу на тип исследования
(ct_chest.json, mammography.json, xray_chest.json, ct_head.json). Имя файла = ключ типа.
"""
import glob
import json
import os
import re

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


def _norm(text):
    """Для поиска по match: нижний регистр, ё → е, «BI-RADS»/«bi rads» → «birads», одиночные пробелы."""
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"bi[\s\-]?rads", "birads", t)
    return re.sub(r"\s+", " ", t)


def select_entries(key, report_text):
    """Записи, у которых хоть одно слово из match есть в протоколе. Ничего не нашлось — весь файл.

    Возвращает (записи, partial): partial=True — показана только часть справочника.
    """
    entries = entries_for(key)
    text = _norm(report_text)
    found = [e for e in entries if any(_norm(m) in text for m in e.get("match", []))]
    if not found or len(found) == len(entries):
        return entries, False
    return found, True


def _action(a):
    what = a["code"] or "без кода"
    line = f"{a['type']}: {what} — {a['note']}"
    if a.get("timing"):
        line += f"; срок: {a['timing']}"
    if a.get("condition"):
        line += f"; условие: {a['condition']}"
    return line


def render_guidelines(key, report_text=None, full=False):
    """Текст для промта: записи этого типа исследования, сгруппированные по документу.

    report_text задан и full=False — только записи, найденные по словам протокола (select_entries).
    """
    if not entries_for(key):
        return "Справочника по этому типу исследования нет: все действия помечай source = null."
    entries, partial = (entries_for(key), False) if full or report_text is None else select_entries(key, report_text)
    lines = ["Справочник «находка → действие» из официальных документов. id записи в [] — значение source."]
    if partial:
        lines.append(f"Показаны только записи, найденные по словам протокола ({len(entries)} из {len(entries_for(key))}). "
                     "Если для значимой находки записи нет или данных не хватает для решения — "
                     "верни \"need_full_guidelines\": true, и тебе покажут весь справочник.")
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


def entry_supports(entry_id, type_, code=None):
    """Есть ли в записи действие этого типа (и с этим кодом, если он задан)."""
    for entries in GUIDELINES.values():
        for e in entries:
            if e["id"] == entry_id:
                return any(a["type"] == type_ and (code is None or a["code"] == code) for a in e["actions"])
    return False


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
