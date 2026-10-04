"""Сравнение режимов промта на одних и тех же протоколах:
полный (БФТ + справочник) / только справочник / только БФТ / без обоих.

python tests/test_modes.py                       — тесты из tests/site_tests.md
python tests/test_modes.py site_tests.md site_mmg — только перечисленные тесты

Отчёт — человекочитаемый текст: tests/results/modes_<дата_время>.md (новый файл на прогон).
Ожидания теста проверяются по кодам, без источника: без справочника ссылаться не на что.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis import (RECOMMENDATION_TYPES, RESEARCH_TYPES, SPECIALISTS,  # noqa: E402
                      attach_sources, build_messages, guidelines_partial, parse_json, validate)
from test_of_models import MODELS, _call, load_tests, report_text, tests_file  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "tests", "results")

MODES = {
    "БФТ + справочник": (True, True),
    "только справочник": (False, True),
    "только БФТ": (True, False),
    "без БФТ и справочника": (False, False),
}
LABELS = {**SPECIALISTS, **RESEARCH_TYPES}


def ask(model, t, use_bft, use_guidelines):
    """Как в test_of_models.ask: справочник по словам протокола, при need_full_guidelines — полный."""
    text = report_text(t)
    msgs = lambda full=False: build_messages(t["study_type"], text, full_guidelines=full,
                                             use_bft=use_bft, use_guidelines=use_guidelines)
    raw = _call(model, msgs())
    if use_guidelines and guidelines_partial(t["study_type"], text) \
            and (parse_json(raw) or {}).get("need_full_guidelines") is True:
        raw = _call(model, msgs(full=True))
    return raw


def codes(parsed):
    opts = [o for o in (parsed or {}).get("options", []) if isinstance(o, dict)]
    return ({i.get("code") for o in opts for i in o.get("items") or [] if isinstance(i, dict)}
            | {o.get("type") for o in opts})


def check_codes(parsed, expected):
    """В каждой группе must_include есть хоть один код; кодов из must_not нет."""
    if not expected:
        return None
    got = codes(parsed)
    miss = [g for g in expected.get("must_include", [])
            if not any((x.get("code") or x.get("type")) in got for x in g)]
    return ([f"нет ни одного из: {', '.join(x.get('code') or x.get('type') for x in g)}" for g in miss]
            + [f"лишнее: {c}" for c in expected.get("must_not", []) if c in got])


def describe(parsed):
    """Ответ модели обычным текстом: варианты, пункты с причинами, сроки, источники."""
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
                lines.append(f"        ссылка не подтверждена (в записи нет такого действия): {', '.join(x['unconfirmed_sources'])}")
            if not refs:
                lines.append("        источник: предложение модели (не из справочника)")
    findings = [r.get("label") for r in parsed.get("reasons") or [] if isinstance(r, dict) and r.get("label")]
    if findings:
        lines.append("  Находки: " + "; ".join(findings))
    if parsed.get("need_full_guidelines"):
        lines.append("  Модель просила полный справочник.")
    return lines or ["  Вариантов нет."]


def main(tests_path="site_tests.md", *only_ids):
    tests = load_tests(tests_file(tests_path))
    if only_ids:
        tests = [t for t in tests if t["id"] in only_ids]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"modes_{time.strftime('%Y%m%d_%H%M%S')}.md")
    report = [f"# Сравнение режимов промта — {time.strftime('%d.%m.%Y %H:%M')}",
              f"Модели: {', '.join(MODELS)}. Тесты: {tests_path}.", ""]
    summary = []
    for model in MODELS:
        for t in tests:
            report += [f"## {t['id']} · {t['study_type']} · {model}", ""]
            if t.get("source"):
                report.append(f"Источник протокола: {t['source']}")
            report += [f"Протокол:\n{report_text(t)}", ""]
            if t.get("check"):
                report.append(f"Что проверяем: {t['check']}")
            if t.get("expected", {}).get("basis"):
                report.append(f"Ожидание: {t['expected']['basis']}")
            report.append("")
            for mode, (bft, guide) in MODES.items():
                start = time.time()
                try:
                    raw = ask(model, t, bft, guide)
                except Exception as e:
                    raw = f"ERROR: {e}"
                parsed = parse_json(raw)
                errors = validate(parsed, t["study_type"])
                problems = check_codes(parsed, t.get("expected"))
                parsed = attach_sources(parsed)
                sec = round(time.time() - start, 1)
                if parsed is None:  # нет ответа (ошибка запроса или не JSON) — ожидания не проверить
                    verdict = "нет ответа модели"
                else:
                    verdict = "—" if problems is None else ("ok" if not problems else "; ".join(problems))
                report.append(f"### {mode} ({sec} с)")
                report.append(f"Формат: {'ok' if not errors else '; '.join(errors)}. Ожидания: {verdict}.")
                if raw.startswith("ERROR"):
                    report.append(f"  Ошибка запроса: {raw[:200]}")
                else:
                    report += describe(parsed)
                report.append("")
                summary.append((t["id"], mode, "ok" if not errors else "FAIL", verdict, sec))
                print(f"[{model}] {t['id']:14} {mode:22} format={'ok' if not errors else 'FAIL'} ожидания={verdict} {sec}s")
    report += ["## Сводка", "", "| Тест | Режим | Формат | Ожидания | Время, с |", "|---|---|---|---|---|"]
    report += [f"| {i} | {m} | {f} | {v} | {s} |" for i, m, f, v, s in summary]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("Отчёт:", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
