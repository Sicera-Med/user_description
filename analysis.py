"""Анализ протокола: промт, выбор шаблона БФТ по типу исследования, разбор и проверка ответа модели."""
import json
import re

from bft_templates import TEMPLATES, find_template, render_template

# Справочники кодов, которыми модель отвечает
RECOMMENDATION_TYPES = {
    "repeat_appointment": "Повторный приём у лечащего врача",
    "specialist_consult": "Консультация специалиста",
    "additional_research": "Дополнительное исследование",
}
SPECIALISTS = {
    "therapist": "Терапевт",
    "pulmonologist": "Пульмонолог",
    "oncologist": "Онколог",
    "cardiologist": "Кардиолог",
    "neurologist": "Невролог",
    "neurosurgeon": "Нейрохирург",
    "orthopedist": "Травматолог-ортопед",
    "gastroenterologist": "Гастроэнтеролог",
    "endocrinologist": "Эндокринолог",
    "urologist": "Уролог",
}
RESEARCH_TYPES = {
    "ct": "Компьютерная томография",
    "ct_contrast": "КТ с контрастированием",
    "mri": "МРТ",
    "pet_ct": "ПЭТ-КТ",
    "ultrasound": "УЗИ",
    "xray": "Рентгенография",
    "biopsy": "Биопсия",
    "bronchoscopy": "Бронхоскопия",
    "lab_tests": "Лабораторные анализы",
}

# Словарь сокращений: глоссарий БФТ ДЗМ (стр. 4) + шкалы и обозначения из текста БФТ
ABBREVIATIONS = {
    # области и модальности
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
    # патологии и анатомия
    "ЗНО": "злокачественное новообразование",
    "ТЭЛА": "тромбоэмболия лёгочной артерии",
    "ВГЛУ": "внутригрудные лимфатические узлы",
    "ГИБВ": "гиперинтенсивность белого вещества",
    "ВКК": "вентрикуло-краниальный коэффициент",
    "ПМА/СМА/ЗМА": "передняя/средняя/задняя мозговая артерия",
    "ВББ": "вертебро-базилярный бассейн",
    "МОС": "металлоостеосинтез",
    "Th, L, C": "грудные, поясничные, шейные позвонки (Th12 — 12-й грудной)",
    "LM, LAD, LCx, RCA": "ствол ЛКА, передняя нисходящая, огибающая, правая коронарная артерия",
    # измерения и шкалы
    "HU": "единицы Хаунсфилда (рентгеновская плотность)",
    "КТИ": "кардиоторакальный индекс (0–1)",
    "Agatston": "кальциевый индекс коронарных артерий",
    "CAC-DRS (CAC DRS A0–A3)": "класс коронарного кальция по индексу Agatston, 0–3",
    "Genant": "степень компрессионной деформации позвонка",
    "BI-RADS": "категория находок в молочной железе, 0–6, ставится по каждой железе",
    "ACR A–D": "рентгенологическая плотность ткани молочной железы",
    "PGMI": "качество укладки при маммографии (P, G, M, I)",
    "ASPECTS": "шкала ишемии в бассейне СМА, 0–10",
    "AO (A/B/C)": "классификация переломов",
    "IASLC": "классификация групп лимфоузлов средостения",
}

SYSTEM_PROMPT = """Ты — ассистент врача (система поддержки принятия решений).
Пользователь — врач. Он присылает протокол лучевого исследования.
Твоя задача — предложить врачу, что делать дальше. Решение принимает врач,
ты не ставишь диагноз и не назначаешь лечение.

Правила:
1. Опирайся только на то, что написано в протоколе. Не придумывай находки, размеры, жалобы.
2. Выбери ОДИН основной вариант (recommendation) из кодов ниже и упорядочи все три кода
   от наиболее к наименее подходящему (options_order). Первый в options_order = recommendation.
3. Если recommendation = specialist_consult — укажи specialists (только коды из списка).
   Если recommendation = additional_research — укажи research_types (только коды из списка).
   Если подходящего кода нет — выбери ближайший и не придумывай новых кодов.
4. reasons — находки из протокола, на которых основан выбор, от главной к второстепенной:
   code — короткий латинский идентификатор в snake_case, label — находка по-русски (до 80 символов).
   Минимум одна причина.
5. Пиши по-русски, медицинской терминологией, без англицизмов (не "бенигный", а "доброкачественный").
6. Шкалы (BI-RADS, CAC-DRS и т.п.) трактуй строго по их категориям, не завышай и не занижай.
7. Ниже дан шаблон протокола для этого типа исследования. Называй находки так, как в шаблоне.
   Поля, которых нет в протоколе, не считай нормой и не додумывай.

Варианты (recommendation):
""" + "\n".join(f"- {k}: {v}" for k, v in RECOMMENDATION_TYPES.items()) + """

Специалисты (specialists):
""" + "\n".join(f"- {k}: {v}" for k, v in SPECIALISTS.items()) + """

Исследования (research_types):
""" + "\n".join(f"- {k}: {v}" for k, v in RESEARCH_TYPES.items()) + """

Сокращения:
""" + "\n".join(f"- {k}: {v}" for k, v in ABBREVIATIONS.items()) + """

Ответ — строго один JSON-объект, без текста и markdown вокруг:
{
  "recommendation": "repeat_appointment | specialist_consult | additional_research",
  "options_order": ["код 1", "код 2", "код 3"],
  "specialists": ["cardiologist"],
  "research_types": [],
  "reasons": [{"code": "cardiomegaly", "label": "Кардиомегалия, КТИ 0,6"}]
}"""



def template_key(study_type):
    """Ключ шаблона БФТ по типу исследования: ключ JSON, синоним из ALIASES или label."""
    tpl = find_template(study_type)
    return next((k for k, v in TEMPLATES.items() if v is tpl), None) if tpl else None


def build_messages(study_type, report_text, patient_context=None):
    """Системный промт + шаблон только этого типа исследования; в сообщении — протокол."""
    key = template_key(study_type)
    label = TEMPLATES[key]["label"] if key else study_type
    template = render_template(key) if key else f"Шаблон протокола для «{label}» в БФТ не найден."
    user = [f"Тип исследования: {label}"]
    ctx = patient_context or {}
    if ctx.get("age") is not None or ctx.get("sex"):
        sex = {"m": "мужской", "f": "женский"}.get(ctx.get("sex"), "не указан")
        user.append(f"Пациент: возраст {ctx.get('age', 'не указан')}, пол {sex}")
    user.append(f"Протокол:\n{report_text}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + template},
        {"role": "user", "content": "\n\n".join(user)},
    ]


def parse_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)  # рассуждения reasoning-моделей
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def validate(parsed):
    """Список нарушений формата ответа; пустой — ответ корректен."""
    if not isinstance(parsed, dict):
        return ["ответ модели не JSON-объект"]
    errors = []
    rec = parsed.get("recommendation")
    if rec not in RECOMMENDATION_TYPES:
        errors.append(f"неизвестный recommendation: {rec!r}")
    order = parsed.get("options_order") or []
    if order and order[0] != rec:
        errors.append("options_order начинается не с recommendation")
    unknown = [c for c in order if c not in RECOMMENDATION_TYPES]
    unknown += [c for c in parsed.get("specialists") or [] if c not in SPECIALISTS]
    unknown += [c for c in parsed.get("research_types") or [] if c not in RESEARCH_TYPES]
    if unknown:
        errors.append(f"коды не из справочника: {unknown}")
    if rec == "specialist_consult" and not parsed.get("specialists"):
        errors.append("specialist_consult без specialists")
    if rec == "additional_research" and not parsed.get("research_types"):
        errors.append("additional_research без research_types")
    reasons = parsed.get("reasons")
    if not isinstance(reasons, list) or not any(
        isinstance(r, dict) and str(r.get("label", "")).strip() for r in reasons
    ):
        errors.append("нет reasons")
    return errors
