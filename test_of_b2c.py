import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from analysis import RECOMMENDATION_TYPES, RESEARCH_TYPES, SPECIALISTS
from bft_templates import TEMPLATES, find_template, render_template


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTS_PATH = os.path.join(BASE_DIR, "tests.md")
RESULTS_PATH = os.path.join(BASE_DIR, "results_b2c.json")
# Заглушка: ссылка на страницу результата появится, когда будет сайт
RESULTS_URL = "https://example.com/results/{study_id}"

# Названия исследований для пациента — без аббревиатур (ключи — шаблоны БФТ)
PATIENT_STUDY_NAMES = {
    "xray_chest": "рентгенография органов грудной клетки",
    "ct_chest": "компьютерная томография органов грудной клетки",
    "mammography": "маммография",
}


load_dotenv()
client = InferenceClient(token=os.environ["HF_TOKEN"])



# MODEL в .env: одна модель или несколько через запятую
MODELS = [m.strip() for m in os.environ["MODEL"].split(",") if m.strip()]


ABBREVIATIONS = {
    "КТ": "компьютерная томография",
    "НДКТ": "низкодозная компьютерная томография",
    "МРТ": "магнитно-резонансная томография",
    "РГ": "рентгенография",
    "ФЛГ": "флюорография",
    "ММГ": "маммография",
    "ОГК": "органы грудной клетки",
    "ОБП": "органы брюшной полости",
    "ОМТ": "органы малого таза",
    "ГМ": "головной мозг",
    "МЖ": "молочная железа",
    "ППН": "придаточные пазухи носа",
    "ОДА": "опорно-двигательный аппарат",
    "КСС": "костно-суставная система",
    "ШОП": "шейный отдел позвоночника",
    "ГОП": "грудной отдел позвоночника",
    "ПКОП": "пояснично-крестцовый отдел позвоночника",
    "ЗНО": "злокачественное новообразование",
    "ТЭЛА": "тромбоэмболия лёгочной артерии",
    "ВГЛУ": "внутригрудные лимфатические узлы",
    "HU": "единицы Хаунсфилда",
    "КТИ": "кардиоторакальный индекс",
    "Agatston": "кальциевый индекс коронарных артерий",
    "BI-RADS": "категория результатов маммографии",
    "ACR": "категория рентгенологической плотности молочной железы",
    "PGMI": "оценка качества маммографического исследования",
    "ASPECTS": "шкала оценки ишемических изменений головного мозга",
}


PATIENT_SYSTEM_PROMPT = """Ты — медицинский информационный ассистент для пациента.

Пользователь — пациент, который получил результат лучевого исследования.

Твоя задача:
1. Объяснить описание и заключение простым и понятным языком.
2. Объяснить медицинские термины, которые могут быть непонятны пациенту.

Что делать дальше, решает лечащий врач. Его решение система покажет пациенту
отдельно, после твоего ответа. Ты дальнейшие шаги не формулируешь.

Ты выполняешь только информационное объяснение уже существующего медицинского заключения.

Ты НЕ ставишь новые диагнозы.
Ты НЕ назначаешь лечение.
Ты НЕ назначаешь дополнительные исследования.
Ты НЕ определяешь самостоятельно дальнейший маршрут пациента.

ПРАВИЛА

1. Используй только информацию, которая присутствует во входном описании и заключении.

2. Не придумывай:
- новые находки;
- диагнозы;
- симптомы;
- жалобы;
- анамнез;
- причины выявленных изменений;
- возможные заболевания;
- последствия;
- прогноз.

3. Если диагноз или термин прямо присутствует в заключении, его можно повторить,
но необходимо объяснить простым языком.

4. Не превращай рентгенологический признак в новый диагноз.

5. Не назначай самостоятельно:
- лекарственные препараты;
- лечение;
- операции;
- процедуры;
- дополнительные исследования;
- консультации специалистов.

6. Не формируй медицинский маршрут пациента и не пересказывай рекомендации
из заключения как указание к действию: дальнейшие шаги пациенту сообщает врач.

7. Медицинские термины объясняй простыми словами,
но не искажай их медицинский смысл.

8. Не используй пугающие формулировки.
Не называй состояние опасным, тяжёлым или критическим,
если это прямо не написано во входных данных.

9. Не преуменьшай выявленные изменения.

10. Если в исследовании прямо указано отсутствие определённых изменений,
это можно сообщить пациенту простым языком.

11. Не интерпретируй числовые показатели по внешним медицинским нормам.
Можно объяснить, что означает показатель, и повторить его значение,
но нельзя самостоятельно утверждать, что он нормальный, повышенный,
сниженный или патологический, если это прямо не сказано в заключении.

12. Ниже будет передан BFT-шаблон данного типа исследования.
Используй его только для понимания структуры протокола,
значения полей и медицинских формулировок.

13. Не сообщай пациенту:
- об обязательных или отсутствующих полях BFT;
- о структуре BFT;
- о DICOM;
- о внутренней работе системы;
- о технической обработке данных.

14. Не добавляй информацию из BFT в результат пациента,
если эта информация отсутствует в его фактическом описании или заключении.

15. Summary должен представлять собой краткое объяснение всего результата.
Обычно 2–5 предложений.

16. В explanations включай только медицинские термины или формулировки,
которые действительно присутствуют во входном описании или заключении
и могут быть непонятны обычному человеку.

17. Explanation должен быть только простым объяснением исходного термина.
Не добавляй возможные причины, последствия, связанные заболевания или прогноз.

18. Не используй формулировки:
"может быть связано с",
"может быть вызвано",
"обычно возникает при",
"указывает на заболевание",
если такая связь прямо не написана во входных данных.

19. При объяснении термина используй минимально необходимое количество
дополнительных медицинских деталей.


20. Не создавай URL и не вставляй ссылки.

СОКРАЩЕНИЯ:

""" + "\n".join(
    f"- {key}: {value}"
    for key, value in ABBREVIATIONS.items()
) + """

Ответ должен содержать строго один JSON-объект.
Не добавляй markdown, комментарии или текст до или после JSON.

Формат ответа:

{
  "summary": "краткое объяснение результата исследования простым языком",
  "explanations": [
    {
      "term": "медицинский термин из описания или заключения",
      "explanation": "простое объяснение этого термина"
    }
  ]
}

Если специальных медицинских терминов, требующих объяснения, нет:
"explanations": []
"""


def load_tests(path):
    """Загрузка тестов из JSON или JSON-блока внутри Markdown."""

    with open(path, encoding="utf-8") as f:
        text = f.read()

    if path.endswith(".md"):
        blocks = re.findall(
            r"```(?:json)?\s*\n(.*?)```",
            text,
            re.S
        )
        text = blocks[0] if blocks else text

    text = re.sub(
        r",\s*\]",
        "]",
        text.strip().rstrip(",")
    )

    return json.loads(text)


def build_input(t):
    """Формирование входа для B2C-LLM.

    Решение врача сюда не передаётся: LLM только объясняет результат исследования,
    дальнейшие шаги добавляет код из doctor_decision (build_next_step).
    """

    tpl = find_template(t["study_type"])

    parts = [
        f"Тип исследования: "
        f"{tpl['label'] if tpl else t['study_type']}"
    ]

    if t.get("description"):
        parts.append(
            f"Описание:\n{t['description']}"
        )

    if t.get("conclusion"):
        parts.append(
            f"Заключение:\n{t['conclusion']}"
        )

    return "\n\n".join(parts)


def system_prompt_for(study_type):
    """B2C-промпт + BFT-шаблон конкретного исследования."""

    return (
        PATIENT_SYSTEM_PROMPT
        + "\n\n"
        + render_template(study_type)
    )


def ask(model, t):
    """Запрос к LLM для B2C-контура."""

    resp = client.chat_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt_for(
                    t["study_type"]
                ),
            },
            {
                "role": "user",
                "content": build_input(t),
            },
        ],
        max_tokens=4096,
        temperature=0.2,
    )

    return resp.choices[0].message.content or ""  # при нехватке токенов content бывает None


def parse_json(raw):
    """Извлечение JSON из ответа модели."""

    raw = re.sub(
        r"<think>.*?</think>",
        "",
        raw,
        flags=re.S
    )

    match = re.search(
        r"\{.*\}",
        raw,
        re.S
    )

    try:
        return json.loads(
            match.group(0)
        ) if match else None

    except json.JSONDecodeError:
        return None


def validate_b2c_response(data):
    """Проверка структуры ответа LLM."""

    if not isinstance(data, dict):
        return False

    if not isinstance(data.get("summary"), str):
        return False

    explanations = data.get("explanations")
    if not isinstance(explanations, list):
        return False

    for item in explanations:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("term"), str):
            return False
        if not isinstance(item.get("explanation"), str):
            return False

    return True


def build_next_step(decision):
    """Дальнейшие шаги для пациента — из решения врача, не из ответа LLM.

    decision: {"chosen_types": [...], "specialists": [...], "research_types": [...], "comment": ...}
    Коды — из справочников analysis.py, как в ответе модели врачу. Нет решения — None.
    """

    if not decision or not decision.get("chosen_types"):
        return None

    chosen = decision["chosen_types"]
    unknown = [c for c in chosen if c not in RECOMMENDATION_TYPES]
    unknown += [c for c in decision.get("specialists", []) if c not in SPECIALISTS]
    unknown += [c for c in decision.get("research_types", []) if c not in RESEARCH_TYPES]
    if unknown:
        raise ValueError(f"doctor_decision: коды не из справочника: {unknown}")

    items = []
    if "repeat_appointment" in chosen:
        items.append("повторный приём у лечащего врача")
    if "specialist_consult" in chosen:
        who = ", ".join(SPECIALISTS[c].lower() for c in decision.get("specialists", []))
        items.append(f"консультация: {who or 'специалист'}")
    if "additional_research" in chosen:
        what = ", ".join(RESEARCH_TYPES[c] for c in decision.get("research_types", []))
        items.append(f"обследование: {what or 'по назначению врача'}")

    text = "Ваш лечащий врач рекомендует: " + "; ".join(items) + "."
    if decision.get("comment"):
        text += f" Комментарий врача: {decision['comment']}"

    return {
        "required": True,
        "text": text,
        "action": {
            "label": "Записаться на приём"
        },
    }


def study_name(study_type):
    """Название исследования для пациента; нет в словаре — как в данных."""
    tpl = find_template(study_type)
    key = next((k for k, v in TEMPLATES.items() if v is tpl), None) if tpl else None
    return PATIENT_STUDY_NAMES.get(key, study_type)


def build_notification(t):
    """Сообщение 1 (SMS/мессенджер): только факт готовности и ссылка. Без медицинских данных."""

    name = (t.get("patient_name") or "").strip()
    greeting = f"Здравствуйте, {name}!" if name else "Здравствуйте!"
    url = RESULTS_URL.format(study_id=t["id"])
    return (
        f"{greeting} Готов результат исследования: {study_name(t['study_type'])}. "
        f"Результаты доступны по ссылке: {url}"
    )


def build_site_summary(t, parsed):
    """Сообщение 2 (страница на сайте): объяснение от LLM + решение врача (код)."""

    if parsed is None:
        return None

    next_step = build_next_step(t.get("doctor_decision"))
    parts = [parsed["summary"].strip()]
    if parsed["explanations"]:
        parts.append("Что означают термины:\n" + "\n".join(
            f"• {e['term']} — {e['explanation']}" for e in parsed["explanations"]
        ))
    parts.append(next_step["text"] if next_step else
                 "Решение о дальнейших шагах врач ещё не принял — мы сообщим, когда оно появится.")

    return {
        "title": f"Результат исследования: {study_name(t['study_type'])}",
        "explanation": parsed["summary"].strip(),
        "terms": parsed["explanations"],
        "doctor_recommendation": next_step,
        "text": "\n\n".join(parts),
    }


def main(tests_path=DEFAULT_TESTS_PATH):
    tests = load_tests(tests_path)
    for t in tests:  # ошибки в решениях врача — до запросов к модели
        build_next_step(t.get("doctor_decision"))
    results = []

    for model in MODELS:
        for t in tests:
            start = time.time()

            try:
                raw = ask(model, t)
            except Exception as e:
                raw = f"ERROR: {e}"

            model_parsed = parse_json(raw)
            model_schema_valid = validate_b2c_response(model_parsed)


            result = {
                "model": model,
                "id": t["id"],
                "pipeline": "b2c",
                "raw": raw,
                "notification": build_notification(t),  # не зависит от модели
                "site_summary": build_site_summary(t, model_parsed if model_schema_valid else None),
                "valid_json": model_parsed is not None,
                "valid_schema": model_schema_valid,
                "sec": round(
                    time.time() - start,
                    2
                ),
            }

            results.append(result)

            print(
                f"[B2C] [{model}] {t['id']}: "
                f"json={'ok' if result['valid_json'] else 'FAIL'} "
                f"schema={'ok' if result['valid_schema'] else 'FAIL'} "
                f"{result['sec']}s"
            )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nB2C results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])