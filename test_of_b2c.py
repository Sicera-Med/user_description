import json
import os
import re
import sys
import time
from copy import deepcopy

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from bft_templates import find_template, render_template


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTS_PATH = os.path.join(BASE_DIR, "tests.md")
RESULTS_PATH = os.path.join(BASE_DIR, "results_b2c.json")


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
1. Сформировать короткое уведомление о готовности результата.
2. Объяснить описание и заключение простым и понятным языком.
3. Объяснить медицинские термины, которые могут быть непонятны пациенту.
4. Если в ЗАКЛЮЧЕНИИ прямо указана рекомендация записаться на консультацию
   или приём к врачу/специалисту, вынести её в отдельный блок next_step.

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

Если консультация специалиста прямо указана в исходном ЗАКЛЮЧЕНИИ,
это не считается твоим назначением: такую рекомендацию нужно только
аккуратно передать пациенту в next_step.

6. Не формируй самостоятельно медицинский маршрут пациента.
Не делай вывод о необходимости консультации только по находкам, диагнозу,
числовым показателям или BFT. next_step разрешён только тогда, когда
рекомендация на консультацию/приём прямо присутствует в поле "Заключение".

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


20. Анализируй поле "Заключение" на наличие явной рекомендации о консультации
или приёме у врача/специалиста.

Примеры явной рекомендации:
- "Рекомендуется консультация кардиолога."
- "Консультация пульмонолога."
- "Рекомендован приём терапевта."

21. Если такая рекомендация прямо есть в ЗАКЛЮЧЕНИИ:
- верни "next_step.required": true;
- в "next_step.text" передай смысл рекомендации без добавления новых назначений;
- в "next_step.button_label" укажи понятную подпись кнопки.
  Например: "Записаться к кардиологу".

22. Если рекомендации на консультацию/приём в ЗАКЛЮЧЕНИИ нет:
"next_step": null

23. Не создавай next_step только потому, что обнаружена патология.
Например, наличие кардиомегалии само по себе НЕ разрешает тебе добавлять
"Рекомендуется консультация кардиолога", если этой рекомендации нет в заключении.

24. Не создавай URL и не вставляй ссылки. Ссылка на запись добавляется системой
после твоего ответа."

СОКРАЩЕНИЯ:

""" + "\n".join(
    f"- {key}: {value}"
    for key, value in ABBREVIATIONS.items()
) + """

Ответ должен содержать строго один JSON-объект.
Не добавляй markdown, комментарии или текст до или после JSON.

Формат ответа:

{
  "notification": {
    "title": "Готов результат исследования",
    "text": "короткое понятное уведомление для пациента"
  },
  "summary": "краткое объяснение результата исследования простым языком",
  "explanations": [
    {
      "term": "медицинский термин из описания или заключения",
      "explanation": "простое объяснение этого термина"
    }
  ],
  "next_step": {
    "required": true,
    "text": "рекомендация, которая прямо указана в заключении",
    "button_label": "Записаться к специалисту"
  }
}

Если специальных медицинских терминов, требующих объяснения, нет:
"explanations": []

Если в заключении нет явной рекомендации на консультацию/приём:
"next_step": null
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

    Подтверждённый маршрут пациента сюда не передаётся.
    LLM только объясняет результат исследования.
    Маршрут и ссылка добавляются позже программно.
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

    notification = data.get("notification")
    if not isinstance(notification, dict):
        return False

    if not isinstance(notification.get("title"), str):
        return False

    if not isinstance(notification.get("text"), str):
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

    next_step = data.get("next_step")
    if next_step is not None:
        if not isinstance(next_step, dict):
            return False
        if next_step.get("required") is not True:
            return False
        if not isinstance(next_step.get("text"), str) or not next_step["text"].strip():
            return False
        button_label = next_step.get("button_label")
        if button_label is not None and not isinstance(button_label, str):
            return False

    return True


def build_next_step(parsed):
    """Добавляет единую ссылку только к рекомендации, найденной LLM в заключении.

    LLM не генерирует URL. Если next_step отсутствует, ссылка не добавляется.
    """

    route = parsed.get("next_step")

    if not isinstance(route, dict):
        return None

    if route.get("required") is not True:
        return None

    text_value = route.get("text")
    button_label = route.get("button_label", "Записаться на приём")

    if not isinstance(text_value, str) or not text_value.strip():
        return None

    if not isinstance(button_label, str) or not button_label.strip():
        button_label = "Записаться на приём"

    return {
        "required": True,
        "text": text_value.strip(),
        "action": {
            "label": button_label.strip()
        },
    }


def attach_next_step(parsed):
    """Заменяет LLM-блок next_step на финальный блок с системной ссылкой."""

    if parsed is None:
        return None

    result = deepcopy(parsed)
    result["next_step"] = build_next_step(parsed)

    return result


def main(tests_path=DEFAULT_TESTS_PATH):
    tests = load_tests(tests_path)
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

            if model_schema_valid:
                final_parsed = attach_next_step(model_parsed)
            else:
                final_parsed = None

            result = {
                "model": model,
                "id": t["id"],
                "pipeline": "b2c",
                "raw": raw,
                "parsed": final_parsed,
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