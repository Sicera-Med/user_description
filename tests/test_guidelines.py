"""Со справочником и без: одна модель (GPT), 5 случаев с ожиданиями. БФТ в обоих режимах.

python tests/test_guidelines.py                  — 5 случаев ниже
python tests/test_guidelines.py ct_chest_02      — только перечисленные

Без справочника ожидания проверяются только по кодам (ссылаться не на что),
со справочником — ещё и по ссылке на нужную запись.
Отчёт: tests/results/guidelines_<дата_время>.md (новый файл на прогон).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import case_header, cases, run, run_block, save, verdict  # noqa: E402
from test_models import GPT_MODEL  # noqa: E402

CASE_IDS = ["xray_01", "ct_chest_02", "ct_chest_05", "mmg_01", "ct_head_01"]  # все 4 типа исследований
MODES = {"со справочником": True, "без справочника": False}


def main(*only_ids):
    tests = cases(only_ids or CASE_IDS)
    report = ["# Со справочником и без", "",
              f"Модель: {GPT_MODEL}. Случаев: {len(tests)}. БФТ в обоих режимах.", ""]
    summary = []
    for t in tests:
        report += [f"## {t['id']} · {t['study_type']}", ""] + case_header(t)
        for mode, use_guidelines in MODES.items():
            r = run(GPT_MODEL, t, use_guidelines=use_guidelines)
            report += run_block(mode, r)
            summary.append((t["id"], mode, r))
            print(f"{t['id']:12} {mode:16} format={'ok' if not r['errors'] else 'FAIL'} "
                  f"ожидания={verdict(r)} {r['sec']}s")

    report += ["## Итог", "",
               "| Режим | Формат ok | Ожидания ok | Пунктов с источником | Неподтв. ссылок | Ср. токенов промта | Ср. время, с |",
               "|---|---|---|---|---|---|---|"]
    for mode in MODES:
        rs = [r for _, m, r in summary if m == mode]
        tokens = [r["tokens"] for r in rs if r["tokens"]]
        report.append(
            f"| {mode} | {sum(1 for r in rs if not r['errors'])}/{len(rs)} "
            f"| {sum(1 for r in rs if r['problems'] == [])}/{sum(1 for t in tests if t.get('expected'))} "
            f"| {sum(r['stats']['confirmed'] for r in rs)}/{sum(r['stats']['items'] for r in rs)} "
            f"| {sum(r['stats']['unconfirmed'] for r in rs)} "
            f"| {round(sum(tokens) / len(tokens)) if tokens else '?'} "
            f"| {round(sum(r['sec'] for r in rs) / max(len(rs), 1), 1)} |")
    report += ["", "## По случаям", "", "| Случай | Режим | Формат | Ожидания | Время, с |", "|---|---|---|---|---|"]
    report += [f"| {i} | {m} | {'ok' if not r['errors'] else 'FAIL'} | {verdict(r)} | {r['sec']} |"
               for i, m, r in summary]
    save("guidelines", report)


if __name__ == "__main__":
    main(*sys.argv[1:])
