"""Анализ протокола: промт, шаблон БФТ и справочник рекомендаций по типу исследования, разбор и проверка ответа."""
import json
import re

from bft_templates import TEMPLATES, find_template, render_template
from guidelines import cite, entry_ids, guideline_key, render_guidelines, select_entries

# Справочники кодов, которыми модель отвечает
RECOMMENDATION_TYPES = {
    "repeat_appointment": "Повторный приём у лечащего врача",
    "urgent_hospitalization": "Экстренная госпитализация",
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
    "vascular_surgeon": "Сердечно-сосудистый (сосудистый) хирург",
    "radiotherapist": "Радиотерапевт",
}
RESEARCH_TYPES = {
    "ct": "Компьютерная томография",
    "ct_contrast": "КТ с контрастированием",
    "mri": "МРТ",
    "pet_ct": "ПЭТ-КТ",
    "ultrasound": "УЗИ",
    "mammography": "Маммография",
    "ldct": "Низкодозная КТ (НДКТ)",
    "ct_angiography": "КТ-ангиография",
    "mri_contrast": "МРТ с контрастированием",
    "adrenal_ct": "КТ надпочечников по протоколу с вымыванием или МРТ с химическим сдвигом",
    "echocardiography": "Эхокардиография (ЭхоКГ)",
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
2. Предложи от 1 до 3 вариантов (options), каждый тип — не больше одного раза,
   от наиболее к наименее подходящему. Ровно один вариант отметь "recommended": true —
   основной; остальные — "recommended": false (альтернативы, из которых врач тоже может выбрать).
   Не добавляй вариант «для полноты», если он не следует из протокола.
3. Внутри варианта перечисли в items всё, что нужно, — несколько специалистов
   или несколько исследований, если находок несколько или одной недостаточно:
   - specialist_consult — коды из списка «Специалисты»;
   - additional_research — коды из списка «Исследования»;
   - repeat_appointment, urgent_hospitalization — items пустой.
   У каждого item укажи reason — какая находка из протокола требует именно его.
   Без повторов. Если подходящего кода нет — выбери ближайший и не придумывай новых кодов.
4. reasons — все значимые находки протокола, от главной к второстепенной:
   code — короткий латинский идентификатор в snake_case, label — находка по-русски (до 80 символов).
   Минимум одна причина.
5. Пиши по-русски, медицинской терминологией, без англицизмов (не "бенигный", а "доброкачественный").
6. Шкалы (BI-RADS, CAC-DRS и т.п.) трактуй строго по их категориям, не завышай и не занижай.
7. Ниже дан шаблон протокола для этого типа исследования. Называй находки так, как в шаблоне.
   Поля, которых нет в протоколе, не считай нормой и не додумывай.
8. Ниже дан справочник «находка → действие» из официальных документов.
   - Если находка есть в справочнике, бери действия и сроки оттуда и укажи source = id записи.
   - Учитывай условия (!) и ограничения записи: например, область применения шкалы.
     Если условие не выполнено или по протоколу его нельзя проверить — скажи об этом в rationale.
   - Действия, которых нет в справочнике, добавляй только если они следуют из протокола,
     и ставь source = null: врач увидит, что это предложение модели, а не документа.
   - timing — срок из справочника; если в справочнике срока нет — null, не придумывай.
   - Если справочник показан частично и для значимой находки нет записи или данных не хватает
     для решения — добавь в ответ "need_full_guidelines": true (остальные поля заполни как сможешь).

Варианты (options[].type):
""" + "\n".join(f"- {k}: {v}" for k, v in RECOMMENDATION_TYPES.items()) + """

Специалисты (items для specialist_consult):
""" + "\n".join(f"- {k}: {v}" for k, v in SPECIALISTS.items()) + """

Исследования (items для additional_research):
""" + "\n".join(f"- {k}: {v}" for k, v in RESEARCH_TYPES.items()) + """

Сокращения:
""" + "\n".join(f"- {k}: {v}" for k, v in ABBREVIATIONS.items()) + """

Ответ — строго один JSON-объект, без текста и markdown вокруг:
{
  "options": [
    {
      "type": "specialist_consult",
      "recommended": true,
      "items": [
        {"code": "cardiologist", "reason": "Кардиомегалия, КТИ 0,6", "timing": null, "source": null},
        {"code": "...", "reason": "другая находка из протокола", "timing": null, "source": null}
      ],
      "source": null,
      "rationale": "почему этот вариант, 1–2 предложения"
    },
    {
      "type": "additional_research",
      "recommended": false,
      "items": [{"code": "ldct", "reason": "Солидный очаг 9 мм", "timing": "через 3 месяца", "source": "lungrads_4a"}],
      "source": null,
      "rationale": "..."
    }
  ],
  "reasons": [{"code": "cardiomegaly", "label": "Кардиомегалия, КТИ 0,6"}]
}"""



def template_key(study_type):
    """Ключ шаблона БФТ по типу исследования: ключ JSON, синоним из ALIASES или label."""
    tpl = find_template(study_type)
    return next((k for k, v in TEMPLATES.items() if v is tpl), None) if tpl else None


def guidelines_partial(study_type, report_text):
    """True — по словам протокола в промт попадёт только часть справочника."""
    key = guideline_key(study_type, template_key(study_type))
    return bool(key) and select_entries(key, report_text)[1]


def build_messages(study_type, report_text, patient_context=None, full_guidelines=False):
    """Системный промт + шаблон этого типа исследования + справочник; в сообщении — протокол.

    Справочник — по словам из match (full_guidelines=False) или целиком (True).
    """
    key = template_key(study_type)
    label = TEMPLATES[key]["label"] if key else study_type
    template = render_template(key) if key else f"Шаблон протокола для «{label}» в БФТ не найден."
    user = [f"Тип исследования: {label}"]
    ctx = patient_context or {}
    if ctx.get("age") is not None or ctx.get("sex"):
        sex = {"m": "мужской", "f": "женский"}.get(ctx.get("sex"), "не указан")
        user.append(f"Пациент: возраст {ctx.get('age', 'не указан')}, пол {sex}")
    user.append(f"Протокол:\n{report_text}")
    guide = render_guidelines(guideline_key(study_type, key), report_text, full=full_guidelines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + template + "\n\n" + guide},
        {"role": "user", "content": "\n\n".join(user)},
    ]


def parse_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)  # рассуждения reasoning-моделей
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


ITEM_CODES = {
    "specialist_consult": SPECIALISTS,
    "additional_research": RESEARCH_TYPES,
    "repeat_appointment": {},
    "urgent_hospitalization": {},
}


def validate(parsed, study_type=None):
    """Список нарушений формата ответа; пустой — ответ корректен.

    study_type задан — source должен быть id записи справочника этого типа исследования или null.
    """
    if not isinstance(parsed, dict):
        return ["ответ модели не JSON-объект"]
    errors = []
    options = parsed.get("options")
    if not isinstance(options, list) or not 1 <= len(options) <= 3:
        errors.append("options: нужно от 1 до 3 вариантов")
        options = options if isinstance(options, list) else []
    types = [o.get("type") for o in options if isinstance(o, dict)]
    if len(types) != len(options):
        errors.append("options: вариант не объект")
    for t in types:
        if t not in RECOMMENDATION_TYPES:
            errors.append(f"неизвестный type: {t!r}")
    if len(set(types)) != len(types):
        errors.append("options: тип повторяется")
    recommended = [o for o in options if isinstance(o, dict) and o.get("recommended") is True]
    if len(recommended) != 1:
        errors.append(f"recommended=true должен быть ровно у одного варианта, а не у {len(recommended)}")
    elif options and options[0] is not recommended[0]:
        errors.append("основной вариант (recommended) должен идти первым")
    known = entry_ids(guideline_key(study_type, template_key(study_type))) if study_type else None
    for o in options:
        if not isinstance(o, dict) or o.get("type") not in ITEM_CODES:
            continue
        if known is not None:
            bad = [x.get("source") for x in [o, *(o.get("items") or [])]
                   if isinstance(x, dict) and x.get("source") not in (None, *known)]
            if bad:
                errors.append(f"{o['type']}: source не из справочника: {bad}")
        allowed = ITEM_CODES[o["type"]]
        items = o.get("items") or []
        codes = [i.get("code") for i in items if isinstance(i, dict)]
        if not allowed:  # варианты без пунктов: повторный приём, экстренная госпитализация
            if items:
                errors.append(f"{o['type']}: items должен быть пустым")
            continue
        if not codes:
            errors.append(f"{o['type']}: пустой items")
        unknown = [c for c in codes if c not in allowed]
        if unknown:
            errors.append(f"{o['type']}: коды не из справочника: {unknown}")
        if len(set(codes)) != len(codes):
            errors.append(f"{o['type']}: коды повторяются")
        if any(not str(i.get("reason", "")).strip() for i in items if isinstance(i, dict)):
            errors.append(f"{o['type']}: у item нет reason")
    reasons = parsed.get("reasons")
    if not isinstance(reasons, list) or not any(
        isinstance(r, dict) and str(r.get("label", "")).strip() for r in reasons
    ):
        errors.append("нет reasons")
    return errors


def attach_sources(parsed):
    """Добавляет к каждому варианту и пункту source_ref — документ и страницы по id из справочника.

    Страницы берутся из guidelines/*.json, а не из ответа модели: модель называет только id.
    source = null — source_ref = null (предложение модели, не из документа).
    """
    if not isinstance(parsed, dict):
        return parsed
    for o in parsed.get("options") or []:
        if not isinstance(o, dict):
            continue
        for x in [o, *(o.get("items") or [])]:
            if isinstance(x, dict) and "source" in x:
                x["source_ref"] = cite(x["source"]) if x["source"] else None
    return parsed
