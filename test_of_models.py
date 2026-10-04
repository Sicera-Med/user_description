"""Прогон тестов из tests.md через модели из .env (промт и проверка ответа — analysis.py)."""
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from analysis import build_messages, parse_json, validate

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


def ask(model, t):
    resp = client.chat_completion(
        model=model,
        messages=build_messages(t["study_type"], report_text(t)),
        max_tokens=4096,  # рассуждающим MoE-моделям нужен запас
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""  # при нехватке токенов content бывает None


def main(tests_path="tests.md"):
    tests = load_tests(tests_path)
    results = []
    for model in MODELS:
        for t in tests:
            start = time.time()
            try:
                raw = ask(model, t)
            except Exception as e:
                raw = f"ERROR: {e}"
            parsed = parse_json(raw)
            errors = validate(parsed)
            results.append({
                "model": model,
                "id": t["id"],
                "raw": raw,
                "parsed": parsed,
                "valid_json": parsed is not None,
                "valid_format": not errors,
                "errors": errors,
                "sec": round(time.time() - start, 2),
            })
            print(
                f"[{model}] {t['id']}: json={'ok' if parsed else 'FAIL'} "
                f"format={'ok' if not errors else 'FAIL'} {results[-1]['sec']}s"
            )

    out = f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"  # не затирать прошлые прогоны
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
