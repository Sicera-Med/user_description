"""Сравнение моделей на 10 случаях из tests/cases.md (БФТ + справочник, как в проде).

python tests/test_models.py                      — все модели, все случаи
python tests/test_models.py xray_01 mmg_01       — только перечисленные случаи

Отчёт: tests/results/models_<дата_время>.md (новый файл на прогон).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import case_header, cases, run, run_block, save, verdict  # noqa: E402

# 5 открытых моделей 4–8 млрд параметров (у всех есть провайдер в HF Inference на 05.10.2026)
SMALL_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",              # 4B, текущая
    "Qwen/Qwen3-8B",                            # 8B, рассуждающая (<think> вырезается)
    "meta-llama/Llama-3.1-8B-Instruct",         # 8B
    "google/gemma-3-4b-it",                     # 4B
    "CohereLabs/c4ai-command-r7b-12-2024",      # 7B, многоязычная
]
# GPT: открытая gpt-oss-20b (MoE, ~3.6B активных). Другую можно задать в .env: GPT_MODEL=...
GPT_MODEL = os.environ.get("GPT_MODEL", "openai/gpt-oss-20b")
MODELS = SMALL_MODELS + [GPT_MODEL]


def main(*only_ids):
    tests = cases(only_ids)
    report = ["# Сравнение моделей", "",
              f"Модели: {', '.join(MODELS)}.", f"Случаев: {len(tests)}. Режим: БФТ + справочник.", ""]
    summary = []
    for t in tests:
        report += [f"## {t['id']} · {t['study_type']}", ""] + case_header(t)
        for model in MODELS:
            r = run(model, t)
            report += run_block(model, r)
            summary.append((model, t["id"], r))
            print(f"{t['id']:12} {model:40} format={'ok' if not r['errors'] else 'FAIL'} "
                  f"ожидания={verdict(r)} {r['sec']}s")

    report += ["## Итог по моделям", "",
               "| Модель | Формат ok | Ожидания ok | Пунктов с источником | Неподтв. ссылок | Ср. время, с |",
               "|---|---|---|---|---|---|"]
    for model in MODELS:
        rs = [r for m, _, r in summary if m == model]
        exp_ok = sum(1 for r in rs if r["problems"] == [])
        exp_all = sum(1 for t in tests if t.get("expected"))
        items = sum(r["stats"]["items"] for r in rs)
        conf = sum(r["stats"]["confirmed"] for r in rs)
        report.append(f"| {model} | {sum(1 for r in rs if not r['errors'])}/{len(rs)} | {exp_ok}/{exp_all} "
                      f"| {conf}/{items} | {sum(r['stats']['unconfirmed'] for r in rs)} "
                      f"| {round(sum(r['sec'] for r in rs) / max(len(rs), 1), 1)} |")
    report += ["", "## По случаям", "", "| Случай | Модель | Формат | Ожидания | Время, с |", "|---|---|---|---|---|"]
    report += [f"| {i} | {m} | {'ok' if not r['errors'] else 'FAIL'} | {verdict(r)} | {r['sec']} |"
               for m, i, r in summary]
    save("models", report)


if __name__ == "__main__":
    main(*sys.argv[1:])
