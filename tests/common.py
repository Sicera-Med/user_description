"""Общее для test_models.py и test_guidelines.py: запрос к модели, проверка ожиданий, текст отчёта."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis import (RECOMMENDATION_TYPES, RESEARCH_TYPES, SPECIALISTS,  # noqa: E402
                      attach_sources, build_messages, guidelines_partial, parse_json, validate)
from test_of_models import client, load_tests, report_text  # noqa: E402

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(TESTS_DIR, "results")
CASES = os.path.join(TESTS_DIR, "cases.md")
LABELS = {**SPECIALISTS, **RESEARCH_TYPES}


def cases(only_ids=()):
    tests = load_tests(CASES)
    return [t for t in tests if t["id"] in only_ids] if only_ids else tests


def call(model, messages):
    """Ответ и число токенов промта (если провайдер его вернул)."""
    resp = client.chat_completion(model=model, messages=messages, max_tokens=4096, temperature=0.2)
    usage = getattr(resp, "usage", None)
    return resp.choices[0].message.content or "", getattr(usage, "prompt_tokens", None)


def ask(model, t, use_bft=True, use_guidelines=True):
    """Справочник по словам протокола; если модель просит весь — второй запрос с полным."""
    text = report_text(t)
    msgs = lambda full=False: build_messages(t["study_type"], text, full_guidelines=full,
                                             use_bft=use_bft, use_guidelines=use_guidelines)
    raw, tokens = call(model, msgs())
    if use_guidelines and guidelines_partial(t["study_type"], text) \
            and (parse_json(raw) or {}).get("need_full_guidelines") is True:
        raw, tokens = call(model, msgs(full=True))
    return raw, tokens


def run(model, t, use_bft=True, use_guidelines=True):
    """Один прогон: ответ, ошибки формата, проблемы с ожиданиями, время."""
    start = time.time()
    try:
        raw, tokens = ask(model, t, use_bft, use_guidelines)
    except Exception as e:
        raw, tokens = f"ERROR: {e}", None
    parsed = parse_json(raw)
    errors = validate(parsed, t["study_type"]) if parsed is not None else ["не JSON"]
    parsed = attach_sources(parsed)
    return {
        "raw": raw,
        "parsed": parsed,
        "errors": errors,
        "problems": check(parsed, t.get("expected"), check_sources=use_guidelines),
        "stats": stats(parsed),
        "tokens": tokens,
        "sec": round(time.time() - start, 1),
    }


def _items(parsed):
    """Все пункты ответа: (код или тип варианта, пункт). Вариант без пунктов — сам пункт."""
    out = []
    for o in (parsed or {}).get("options") or []:
        if not isinstance(o, dict):
            continue
        out.append((o.get("type"), o))
        out += [(i.get("code"), i) for i in o.get("items") or [] if isinstance(i, dict)]
    return out


def check(parsed, expected, check_sources=True):
    """None — у теста нет ожиданий или нет ответа; [] — всё ок; иначе список проблем.

    В каждой группе must_include нужен хотя бы один код. Если у элемента есть source и справочник
    в промте был — этот код должен ссылаться на эту запись (подтверждённая ссылка).
    """
    if not expected or parsed is None:
        return None
    got = _items(parsed)
    confirmed = {(c, s) for c, x in got for s in x.get("sources") or [] if s not in (x.get("unconfirmed_sources") or [])}
    codes = {c for c, _ in got}

    def ok(g):
        c = g.get("code") or g.get("type")
        if check_sources and g.get("source"):
            return (c, g["source"]) in confirmed
        return c in codes

    problems = []
    for group in expected.get("must_include", []):
        if not any(ok(g) for g in group):
            names = [(g.get("code") or g.get("type")) + (f"/{g['source']}" if check_sources and g.get("source") else "")
                     for g in group]
            problems.append(f"нет ни одного из: {', '.join(names)}")
    problems += [f"лишнее: {c}" for c in expected.get("must_not", []) if c in codes]
    return problems


def stats(parsed):
    """Сколько пунктов, сколько с подтверждённым источником, сколько неподтверждённых ссылок."""
    items = [x for c, x in _items(parsed) if x.get("code") or not x.get("items")]
    return {
        "items": len(items),
        "confirmed": sum(1 for x in items if x.get("source_refs")),
        "unconfirmed": sum(len(x.get("unconfirmed_sources") or []) for x in items),
    }


def verdict(r):
    if r["parsed"] is None:
        return "нет ответа модели"
    if r["problems"] is None:
        return "—"
    return "ok" if not r["problems"] else "; ".join(r["problems"])


def describe(parsed):
    """Ответ модели обычным текстом: варианты, пункты с причинами, источники."""
    if not isinstance(parsed, dict):
        return ["  Ответ не разобран (не JSON)."]
    lines = []
    for o in parsed.get("options") or []:
        if not isinstance(o, dict):
            continue
        mark = "ОСНОВНОЙ" if o.get("recommended") else "альтернатива"
        lines.append(f"  [{mark}] {RECOMMENDATION_TYPES.get(o.get('type'), o.get('type'))}")
        if o.get("rationale"):
            lines.append(f"      почему: {o['rationale']}")
        for x in [o, *(o.get("items") or [])]:
            if not isinstance(x, dict):
                continue
            if x is not o:
                line = f"      • {LABELS.get(x.get('code'), x.get('code'))}"
                if x.get("reason"):
                    line += f" — {x['reason']}"
                if x.get("timing"):
                    line += f"; срок: {x['timing']}"
                lines.append(line)
            elif not x.get("sources"):
                continue
            refs = x.get("source_refs") or []
            lines += [f"        источник: {r['text']}" for r in refs]
            if x.get("unconfirmed_sources"):
                lines.append(f"        ссылка не подтверждена: {', '.join(x['unconfirmed_sources'])}")
            if not refs:
                lines.append("        источник: предложение модели (не из справочника)")
    if parsed.get("need_full_guidelines"):
        lines.append("  Модель просила полный справочник.")
    return lines or ["  Вариантов нет."]


def case_header(t):
    lines = [f"Протокол:\n{report_text(t)}", ""]
    if t.get("check"):
        lines.append(f"Что проверяем: {t['check']}")
    if t.get("expected", {}).get("basis"):
        lines.append(f"Ожидание: {t['expected']['basis']}")
    return lines + [""]


def run_block(title, r):
    """Блок отчёта по одному прогону."""
    s = r["stats"]
    lines = [f"### {title} ({r['sec']} с, токенов промта: {r['tokens'] or '?'})",
             f"Формат: {'ok' if not r['errors'] else '; '.join(r['errors'])}. Ожидания: {verdict(r)}.",
             f"Пунктов: {s['items']}, с подтверждённым источником: {s['confirmed']}, неподтверждённых ссылок: {s['unconfirmed']}."]
    if r["raw"].startswith("ERROR"):
        lines.append(f"  Ошибка запроса: {r['raw'][:300]}")
    elif r["parsed"] is None:
        lines.append(f"  Ответ (не JSON): {r['raw'][:500]}")
    else:
        lines += describe(r["parsed"])
    return lines + [""]


def save(name, lines):
    """Новый файл на каждый прогон: tests/results/<name>_<дата_время>.md."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Отчёт:", out)
