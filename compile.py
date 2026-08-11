"""Компилятор ковенантов: договор -> тексты пунктов -> LLM -> спецификация.

Два этапа:
1) детерминированный: вырезать тексты пунктов 6.x из действующего договора
   каждого сценария -> data/processed/clauses.json (работает без ключа);
2) LLM (Gemini, structured output, temperature=0): пункт + леджер сценария ->
   спецификация с отбором транзакций -> data/processed/specs.json.

LLM понимает текст и отбирает транзакции; все суммы считает evaluate.py.

python compile.py                # оба этапа (нужен GEMINI_API_KEY в .env)
python compile.py --clauses-only # только этап 1
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pymupdf
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ledger import load_ledger

import paths

# договоры встречаются и на русском, и на английском
CLAUSE_WORDS = r"(?:Пункт|Section|Clause)"
CLAUSE_SPLIT_RE = re.compile(rf"(?={CLAUSE_WORDS}\s*\d+\.\d+)")
CLAUSE_NUM_RE = re.compile(rf"^{CLAUSE_WORDS}\s*(\d+\.\d+)")


# ---------- этап 1: тексты пунктов ----------

def extract_clauses() -> dict[str, dict[str, str]]:
    """scenario_id -> {clause -> дословный текст пункта}."""
    idx = pd.read_csv(paths.processed("doc_index.csv"))
    contracts = idx[(idx["doc_type"] == "loan_agreement") & (~idx["is_void"])]
    template_scenarios = list(json.load(open(paths.template()))["answers"])

    template = json.load(open(paths.template()))["answers"]

    out: dict[str, dict[str, str]] = {}
    for _, row in contracts.iterrows():
        scen = row["scenario_id"]
        if scen not in template_scenarios:
            continue
        wanted = set(template[scen])                      # 6.1..6.4 или 5.1..5.3
        articles = {c.split(".")[0] for c in wanted}      # номера статей в шаблоне
        with pymupdf.open(paths.documents() / row["doc_id"]) as doc:
            text = "\n".join(p.get_text() for p in doc)

        clauses: dict[str, str] = {}
        for art in articles:
            # тело статьи: от её заголовка до заголовка следующей
            start = re.search(rf"Статья {art} — ", text)
            if not start:
                continue
            end = re.search(rf"Статья {int(art) + 1} — ", text[start.end():])
            body = text[start.start():start.end() + end.start()] if end else text[start.start():]
            for chunk in CLAUSE_SPLIT_RE.split(body):
                m = CLAUSE_NUM_RE.match(chunk.strip())
                if m and m.group(1) in wanted:
                    clauses[m.group(1)] = " ".join(chunk.split())
        # запасной проход по всему документу: английские договоры нумеруют
        # разделы иначе («Article V — Financial Covenants», «Section 5.1»)
        for chunk in CLAUSE_SPLIT_RE.split(text):
            m = CLAUSE_NUM_RE.match(chunk.strip())
            if m and m.group(1) in wanted and m.group(1) not in clauses:
                clauses[m.group(1)] = " ".join(chunk.split())
        out[scen] = clauses

    missing = [s for s in template_scenarios if s not in out]
    if missing:
        print(f"  !! сценарии без договора: {missing}")
    for s, cls in out.items():
        gap = set(template[s]) - set(cls)
        if gap:
            print(f"  !! {s}: не найдены пункты {sorted(gap)}")
    return out


# ---------- этап 2: LLM -> спецификация ----------

class LedgerAmendment(BaseModel):
    """Сумма операции, отсутствующая/исправленная в реестре, из документов."""
    txn_id: str
    amount_usd: float = Field(description="Фактическая сумма в USD; расход — отрицательная")


class CovenantSpec(BaseModel):
    """Что LLM извлекает из пункта. Числа из ответа LLM в submission не попадают:
    threshold — порог из текста договора, суммы считает pandas."""
    analysis: str = Field(description="Кратко (до 3 предложений): что измеряем, по какому признаку отобраны операции, что предписали документы. Заполняется ПЕРВЫМ")
    metric_description: str = Field(description="Что измеряет ковенант, одной фразой")
    is_ratio: bool = Field(description="True, если метрика — коэффициент, а не сумма в USD")
    aggregation: str = Field(description="'sum' — сумма отобранных операций за период (обычный случай); 'max' — проверка по наибольшей из отдельных статей, а не по их сумме; 'min_quarterly'/'max_quarterly' — если ковенант проверяется ЗА КАЖДЫЙ КВАРТАЛ отдельно (тогда actual — худший квартал)")
    threshold: float = Field(description="Порог В ТЕХ ЖЕ ЕДИНИЦАХ, что и метрика. Если он задан как доля от величины из отчётности («5 процентов капзатрат Группы»), вычисли его в долларах: 0.05 * эту величину из документов, а не пиши 5")
    direction: str = Field(description="'max' — нарушение при превышении порога, 'min' — при значении ниже порога")
    trigger_active: bool = Field(description="False только если ковенант применяется при условии (триггере), и это условие НЕ сработало")
    trigger_reasoning: str = Field(description="Если в пункте есть условие применимости — почему сработало/нет, иначе пустая строка")
    carve_out_satisfied: bool = Field(description="True только если превышение покрыто оговоркой (carve-out) из текста пункта")
    carve_out_reasoning: str = Field(description="Какая оговорка есть в пункте и выполнена ли, иначе пустая строка")
    relevant_txn_ids: list[str] = Field(description="Транзакции в скоупе метрики (числитель для коэффициента)")
    denominator_txn_ids: list[str] = Field(description="Транзакции знаменателя (только для коэффициентов, иначе пусто)")
    excluded_txn_ids: list[str] = Field(
        description="Операции, которые попали бы в скоуп, но исключены по предписанию документов (отсечение периода, переклассификация, исключение аудитором)")
    ledger_amendments: list[LedgerAmendment] = Field(
        description="Операции из скоупа с amount_missing=True или исправленные: их фактические суммы, раскрытые в документах")
    off_ledger_amounts_usd: list[float] = Field(
        description="Суммы, раскрытые в документах для агрегирования по этой метрике (числителю), но не отражённые в реестре отдельной операцией; расход — отрицательная")
    denominator_off_ledger_usd: list[float] = Field(
        description="Для коэффициентов: величины знаменателя, раскрытые в документах (EBITDA, долг, выручка и т.п. из аудита), если они не считаются из операций реестра")
    confidence: float = Field(description="Уверенность 0..1")


PROMPT = """Ты компилируешь пункт кредитного договора в машинную спецификацию.
Ниже дословный текст пункта и полный леджер транзакций этого заёмщика за период.

ПУНКТ ДОГОВОРА ({scenario} {clause}):
{clause_text}

ЛЕДЖЕР (расходы отрицательные; amount_missing=True — сумма не выгружена, ищи в документах).
Колонка category — категория операции, определённая ЗАРАНЕЕ единым разбором всего
реестра этого заёмщика; related=True — контрагент признан связанной стороной по
правилу из KYC. ОПИРАЙСЯ НА ЭТИ КОЛОНКИ, а не на собственное прочтение описаний:
  revenue — выручка (other_income выручкой НЕ является);
  EBITDA = revenue минус (payroll + rent + utilities + opex_other);
  операционные расходы = payroll + rent + utilities + opex_other;
  tax, interest, capex, financing в операционные расходы и EBITDA НЕ входят.
{ledger_csv}

ПРАВИЛО СВЯЗАННЫХ СТОРОН У ЭТОГО ЗАЁМЩИКА: {related_rule}

СПРАВОЧНЫЕ ДОКУМЕНТЫ ЗАЁМЩИКА (KYC — определяет связанные стороны; аудит и служебные
записки — раскрытия сумм, переклассификации, валютные курсы, внекнижные обязательства):
{support_text}

Задача:
1. Определи метрику, порог (threshold) и направление (direction).
2. Отбери transaction ids, входящие в скоуп метрики (relevant_txn_ids) — по
   назначению платежа и контрагенту. Для коэффициентов ОБЯЗАТЕЛЬНО заполни
   знаменатель, иначе ячейка потеряна. Базовые определения:
   - Выручка = ВСЕ поступления периода (положительные amount_usd). Если
     знаменатель — «выручка», бери их все, а не одну операцию;
   - Операционные расходы = списания на текущую деятельность: сырьё, ФОТ,
     аренда, коммунальные, ремонт/обслуживание, услуги, страхование, реклама.
     НЕ операционные и в них НЕ входят: проценты и банковские комиссии,
     налоги, капитальные затраты, дивиденды, погашение долга;
   - EBITDA = Выручка минус Операционные расходы -> посчитать её сам ты НЕ
     можешь: положи в denominator_txn_ids все операции выручки И все операции
     операционных расходов (и ничего больше) — код сложит их со знаком сам.
   Если величина знаменателя раскрыта готовым числом в аудите (например,
   консолидированный показатель Группы) — положи её в denominator_off_ledger_usd.
3. Внимательно проверь оговорки (carve-outs) и условия применимости (триггеры)
   в тексте пункта: превышение может быть допустимым, ковенант может не действовать.
4. Реестр «грязный»: у операции с amount_missing=True сумма не выгружена —
   найди её фактическую сумму в документах (ledger_amendments). Если документы
   раскрывают сумму для агрегирования, которой нет в реестре отдельной строкой,
   укажи её в off_ledger_amounts_usd.
5. Аудиторские примечания «для целей ковенантов» ОБЯЗАТЕЛЬНЫ к применению:
   - операция исключена аудитором из периода/скоупа (отсечение, переход рисков,
     переклассификация ИЗ категории) -> НЕ включай её в relevant/denominator,
     а укажи в excluded_txn_ids;
   - переклассификация В категорию метрики -> включай в relevant_txn_ids;
   - формулировки «корректировка не требуется», «классификация сохраняется» —
     ложный след: ничего не меняй по таким примечаниям.
6. Если пункт отсылает к досье KYC/комплаенс за кругом связанных (аффилированных)
   сторон — этот список ЗАКРЫТЫЙ: включай операцию, только если её контрагент
   назван в досье. Назначение платежа при этом не имеет значения: платёж
   связанной стороне учитывается независимо от категории расхода, а платёж
   не названному в досье контрагенту не учитывается, как бы ни назывался.
7. Проверь порядок числителя и знаменателя по названию метрики: «отношение A
   к B» -> A числитель, B знаменатель. Прикинь порядок величины результата и
   сверь с порогом — если результат отличается от порога в десятки раз,
   ты, вероятно, ошибся скоупом или перепутал местами.
8. СОГЛАСУЙ ЕДИНИЦЫ порога и метрики, иначе вердикт будет случайным:
   - is_ratio=true -> threshold маленький (обычно 0.01..10);
   - is_ratio=false (сумма в USD) -> threshold в долларах.
   Если порог выражен через показатель отчётности («Разрешённая величина
   означает 5 процентов капитальных затрат Группы»), найди эту величину в
   документах и подставь произведение в долларах.
8. Не агрегируй суммы сам — классифицируй операции и дословно переноси
   раскрытые в документах значения. Верни строго JSON по схеме."""


# секции документов, влияющие на расчёт: раскрытия, а не учётная политика
RELEVANT_MARKERS = ("связанн", "аффилир", "ковенант", "курс", "переклассифиц",
                    "не отражена", "не отражается", "раскрывается", "исключена",
                    "TXN-", "$")
SECTION_SPLIT_RE = re.compile(r"(?=Примечание \d+ —|ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ|"
                              r"^[А-ЯA-Z][А-ЯA-Z ·—-]{12,}$)", re.M)


def trim_document(text: str, doc_type: str) -> str:
    """Выбрасывает шаблонные разделы, оставляя раскрытия и списки контрагентов.

    Аудиторские отчёты наполовину состоят из учётной политики, одинаковой у всех
    заёмщиков: она не влияет на расчёт, но раздувает промпт и добавляет шум.
    """
    if doc_type == "kyc" or len(text) < 3000:
        return text  # KYC — сам по себе список связанных сторон, режем только крупное
    kept = [s for s in SECTION_SPLIT_RE.split(text)
            if any(m in s for m in RELEVANT_MARKERS)]
    return "\n".join(kept) if kept else text[:3000]


def _load_classes() -> dict:
    """Готовая классификация реестра, если она была построена (classify.py)."""
    path = paths.processed("classes.json")
    return json.loads(path.read_text()) if path.exists() else {}


def _with_categories(sdf, classification: dict | None):
    """Добавляет к реестру колонки category и related из классификации."""
    if not classification:
        return sdf
    txns = classification.get("transactions", {})
    sdf = sdf.copy()
    sdf["category"] = sdf["txn_id"].map(lambda t: (txns.get(t) or {}).get("category", "?"))
    sdf["related"] = sdf["txn_id"].map(lambda t: (txns.get(t) or {}).get("is_related_party", False))
    return sdf


def support_texts() -> dict[str, str]:
    """scenario_id -> текст действующих KYC и аудиторских отчётов (для промпта)."""
    idx = pd.read_csv(paths.processed("doc_index.csv"))
    # все действующие документы сценария, кроме самого договора:
    # KYC (связанные стороны), аудит (раскрытия), прочее (служебные записки)
    docs = idx[(idx["doc_type"] != "loan_agreement") & (~idx["is_void"])]
    out: dict[str, str] = {}
    for _, row in docs.iterrows():
        scen = row["scenario_id"]
        if pd.isna(scen):
            continue
        with pymupdf.open(paths.documents() / row["doc_id"]) as doc:
            text = "\n".join(p.get_text() for p in doc)
        text = trim_document(text, row["doc_type"])
        out[scen] = out.get(scen, "") + f"\n--- {row['doc_type']} {row['doc_id']} ---\n{text}"
    return {s: t[:60000] for s, t in out.items()}


def compile_specs(clauses: dict, df: pd.DataFrame) -> dict:
    import llm

    router = llm.Router()
    schema = CovenantSpec.model_json_schema()

    specs: dict[str, dict[str, dict]] = {}
    if paths.processed("specs.json").exists():  # докомпиляция после падения — не пережигать вызовы
        specs = json.load(open(paths.processed("specs.json")))

    support = support_texts()
    classes = _load_classes()
    cols = ["txn_id", "date", "counterparty", "description", "amount", "currency",
            "amount_usd", "amount_missing"]
    for scen, cls in clauses.items():
        sdf = _with_categories(df[df["scenario_id"] == scen], classes.get(scen))
        extra = [c for c in ("category", "related") if c in sdf.columns]
        ledger_csv = sdf[cols + extra].to_csv(index=False)
        rule = (classes.get(scen) or {}).get("related_party_rule", "(досье не разобрано)")
        for clause, clause_text in sorted(cls.items()):
            if specs.get(scen, {}).get(clause):
                continue
            prompt = PROMPT.format(scenario=scen, clause=clause,
                                   clause_text=clause_text, ledger_csv=ledger_csv,
                                   support_text=support.get(scen, "(нет)"),
                                   related_rule=rule)
            last_err = None
            for attempt in range(3):
                try:
                    raw = router.complete(prompt, schema)
                    spec = CovenantSpec.model_validate_json(raw)
                    if spec.is_ratio and not (spec.denominator_txn_ids
                                              or spec.denominator_off_ledger_usd):
                        raise ValueError("коэффициент без знаменателя")
                    payload = spec.model_dump()
                    payload["_model"] = router.current  # какая модель дала ответ
                    specs.setdefault(scen, {})[clause] = payload
                    break
                except Exception as e:
                    last_err = e
                    if "исчерпан" in str(e):
                        break  # цепочка кончилась — дальше бессмысленно
            else:
                print(f"  !! {scen} {clause}: {last_err}")
            paths.processed("specs.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2))
            got = specs.get(scen, {}).get(clause)
            print(f"  {scen} {clause}: {'ok' if got else 'FAIL'} [{router.current}]")
    return specs


if __name__ == "__main__":
    load_dotenv()
    clauses = extract_clauses()
    paths.processed("clauses.json").parent.mkdir(parents=True, exist_ok=True)
    paths.processed("clauses.json").write_text(json.dumps(clauses, ensure_ascii=False, indent=2))
    n = sum(len(v) for v in clauses.values())
    print(f"Пункты: {n} шт. по {len(clauses)} сценариям -> {paths.processed("clauses.json")}")

    if "--clauses-only" in sys.argv:
        sys.exit(0)

    specs = compile_specs(clauses, load_ledger())
    done = sum(1 for s in specs.values() for _ in s)
    print(f"Спецификаций: {done}/{n} -> {paths.processed("specs.json")}")
