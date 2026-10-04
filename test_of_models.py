"""Прогон тестов из tests.md через модели из .env (промт и проверка ответа — analysis.py)."""
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from analysis import attach_sources, build_messages, guidelines_partial, parse_json, validate

load_dotenv()
client = InferenceClient(token=os.environ["HF_TOKEN"])

# MODEL в .env: одна модель или несколько через запятую
MODELS = [m.strip() for m in os.environ["MODEL"].split(",") if m.strip()]


def load_tests(path):
    """Тесты — JSON-список (в .json или в блоке ```json внутри .md)."""
    text = open(path, encoding="utf-8").read()
    if path.endswith(".md"):
        blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
        text = blocks[0] if blocks else text
    text = re.sub(r",\s*\]", "]", text.strip().rstrip(","))  # висячие запятые
    return json.loads(text)


def report_text(t):
    """Описание и заключение одним текстом протокола."""
    parts = []
    if t.get("description"):
        parts.append(f"Описание: {t['description']}")
    if t.get("conclusion"):
        parts.append(f"Заключение: {t['conclusion']}")
    return "\n".join(parts)


def check_expected(parsed, expected):
    """Ожидания из справочника.

    must_include — список групп; в каждой группе должен найтись хотя бы один элемент:
    {"code", "source"} — пункт варианта, {"type", "source"} — сам вариант (например, повторный приём).
    must_not — коды, которых не должно быть ни в одном пункте.
    """
    if not expected:
        return None
    options = [o for o in (parsed or {}).get("options", []) if isinstance(o, dict)]
    items = [i for o in options for i in o.get("items") or [] if isinstance(i, dict)]
    got = {("code", i.get("code"), i.get("source")) for i in items}
    got |= {("type", o.get("type"), o.get("source")) for o in options}
    key = lambda g: ("code", g["code"], g["source"]) if "code" in g else ("type", g["type"], g["source"])
    problems = [f"нет ни одного из {[(g.get('code') or g.get('type')) + '/' + g['source'] for g in group]}"
                for group in expected.get("must_include", []) if not any(key(g) in got for g in group)]
    problems += [f"есть запрещённый код {c}" for c in expected.get("must_not", []) if c in {i.get("code") for i in items}]
    return problems


def _call(model, messages):
    resp = client.chat_completion(
        model=model, messages=messages,
        max_tokens=4096,  # рассуждающим MoE-моделям нужен запас
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""  # при нехватке токенов content бывает None


def ask(model, t):
    """Справочник по словам протокола; если модель просит весь справочник — второй запрос с полным.

    Возвращает (ответ, режим справочника).
    """
    text = report_text(t)
    if not guidelines_partial(t["study_type"], text):
        return _call(model, build_messages(t["study_type"], text)), "весь файл"
    raw = _call(model, build_messages(t["study_type"], text))
    if (parse_json(raw) or {}).get("need_full_guidelines") is True:
        return _call(model, build_messages(t["study_type"], text, full_guidelines=True)), "весь файл по запросу модели"
    return raw, "по словам протокола"


def main(tests_path="tests.md", *only_ids):
    """python test_of_models.py tests.md mmg_101 ct_chest_101 — только перечисленные тесты."""
    tests = load_tests(tests_path)
    if only_ids:
        tests = [t for t in tests if t["id"] in only_ids]
    results = []
    for model in MODELS:
        for t in tests:
            start = time.time()
            try:
                raw, mode = ask(model, t)
            except Exception as e:
                raw, mode = f"ERROR: {e}", None
            parsed = parse_json(raw)
            errors = validate(parsed, t["study_type"])
            expected = check_expected(parsed, t.get("expected"))
            parsed = attach_sources(parsed)  # документ и страницы по source
            results.append({
                "model": model,
                "id": t["id"],
                "raw": raw,
                "parsed": parsed,
                "valid_json": parsed is not None,
                "valid_format": not errors,
                "errors": errors,
                "expected_problems": expected,  # None — у теста нет ожиданий
                "guidelines_mode": mode,
                "sec": round(time.time() - start, 2),
            })
            print(
                f"[{model}] {t['id']}: json={'ok' if parsed else 'FAIL'} "
                f"format={'ok' if not errors else 'FAIL'} "
                + ("" if expected is None else f"guideline={'ok' if not expected else 'FAIL'} ")
                + f"справочник: {mode} {results[-1]['sec']}s"
            )

    out = f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"  # не затирать прошлые прогоны
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
