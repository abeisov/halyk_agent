"""Классификация реестра заёмщика — один проход на заёмщика, не на пункт.

Категории в реестре отсутствуют: их приходится выводить из назначения платежа
и контрагента. Раньше это делалось внутри компиляции каждого пункта, то есть
одни и те же операции классифицировались трижды и каждый раз по-разному —
отсюда знаменатели, промахивавшиеся в десятки раз.

Здесь классификация выполняется один раз, результат кэшируется и используется
всеми пунктами заёмщика. Отдельно разбирается KYC: там задан не список
связанных сторон, а ПРАВИЛО — доли участия контрагентов и порог, начиная с
которого контрагент признаётся связанным (порог свой у каждого заёмщика).
"""

import json

import pandas as pd
import pymupdf
from pydantic import BaseModel, Field

import paths

# Категории подобраны под то, что реально требуют ковенанты: выручка и
# операционные расходы образуют EBITDA, остальное из неё исключается.
CATEGORIES = [
    "revenue",          # выручка от основной деятельности
    "other_income",     # возвраты, компенсации, проценты к получению — НЕ выручка
    "payroll",          # оплата труда
    "rent",             # аренда и содержание помещений
    "utilities",        # коммунальные услуги
    "opex_other",       # прочие операционные расходы
    "capex",            # капитальные затраты
    "tax",              # налоги и сборы
    "interest",         # проценты и банковские комиссии
    "financing",        # привлечение и погашение финансирования
    "other",            # всё остальное
]


class TxnClass(BaseModel):
    txn_id: str
    category: str = Field(description=f"Одна из: {', '.join(CATEGORIES)}")
    is_related_party: bool = Field(
        description="True, только если контрагент признан связанной стороной по "
                    "правилу из KYC (доля участия >= порога, указанного в досье)")


class LedgerClassification(BaseModel):
    related_party_rule: str = Field(
        description="Дословно: порог владения из KYC и какие контрагенты его достигают")
    related_party_counterparties: list[str] = Field(
        description="Контрагенты, признанные связанными сторонами по этому правилу")
    transactions: list[TxnClass]


PROMPT = """Классифицируй операции реестра заёмщика {scenario}. Категории нужны
для проверки кредитных ковенантов, поэтому границы важны.

ПРАВИЛА КАТЕГОРИЙ:
- revenue — выручка от основной деятельности (продажи, услуги, перевалка, аренда
  переданного в субаренду). ТОЛЬКО операционные продажи;
- other_income — поступления, которые выручкой НЕ являются: возвраты налогов,
  страховые возмещения и скидки страховых брокеров, проценты по депозитам,
  возвраты авансов и депозитов, кредит-ноты, компенсации, сторнирование начислений;
- payroll / rent / utilities — оплата труда, аренда и содержание помещений,
  коммунальные услуги (это операционные расходы, но их часто ограничивают отдельно);
- opex_other — прочие операционные расходы: сырьё, ремонт, обслуживание, услуги,
  страхование, реклама, консультанты;
- capex — приобретение и модернизация основных средств, оборудования, строительство;
- tax — налоги, сборы, пошлины (НЕ операционные расходы);
- interest — проценты по займам и банковские комиссии (НЕ операционные расходы);
- financing — получение и погашение займов, вклады в капитал, дивиденды;
- other — что не подходит ни к чему.
EBITDA = revenue минус (payroll + rent + utilities + opex_other). Категории
tax, interest, capex, financing, other_income в EBITDA НЕ входят.

СВЯЗАННЫЕ СТОРОНЫ: в досье KYC задан ПОРОГ доли участия (например «30.0% и более»)
и перечислены контрагенты с их долями. Связанной стороной является контрагент,
чья доля ДОСТИГАЕТ порога. Контрагент с долей ниже порога связанной стороной НЕ
является, как бы ни назывался. Назначение платежа роли не играет.
Если у заёмщика несколько досье, действует то, где указаны доли конкретных
контрагентов; методические инструкции без данных о клиенте игнорируй.

ДОСЬЕ KYC:
{kyc_text}

РЕЕСТР (расходы отрицательные):
{ledger_csv}

Верни классификацию КАЖДОЙ операции реестра. Ни одну не пропускай."""


def kyc_text(scenario: str) -> str:
    idx = pd.read_csv(paths.processed("doc_index.csv"))
    docs = idx[(idx["scenario_id"] == scenario) & (idx["doc_type"] == "kyc")
               & (~idx["is_void"])]
    parts = []
    for _, row in docs.iterrows():
        with pymupdf.open(paths.documents() / row["doc_id"]) as doc:
            parts.append("\n".join(p.get_text() for p in doc))
    return "\n---\n".join(parts) if parts else "(досье не найдено)"


def classify_scenario(scenario: str, df: pd.DataFrame, router) -> dict:
    cols = ["txn_id", "date", "counterparty", "description", "amount_usd"]
    sdf = df[df["scenario_id"] == scenario]
    prompt = PROMPT.format(scenario=scenario, kyc_text=kyc_text(scenario)[:12000],
                           ledger_csv=sdf[cols].to_csv(index=False))
    raw = router.complete(prompt, LedgerClassification.model_json_schema())
    result = LedgerClassification.model_validate_json(raw)

    known = set(sdf["txn_id"])
    by_txn = {t.txn_id: t.model_dump() for t in result.transactions if t.txn_id in known}
    missing = known - set(by_txn)
    for txn_id in missing:  # пропущенные не теряем, помечаем нейтрально
        by_txn[txn_id] = {"txn_id": txn_id, "category": "other", "is_related_party": False}
    if missing:
        print(f"  .. {scenario}: модель пропустила {len(missing)} операций -> other")
    return {
        "related_party_rule": result.related_party_rule,
        "related_party_counterparties": result.related_party_counterparties,
        "transactions": by_txn,
    }


def classify_all(df: pd.DataFrame, router, scenarios: list[str]) -> dict:
    """scenario -> классификация; уже посчитанные берутся из кэша."""
    path = paths.processed("classes.json")
    out = json.loads(path.read_text()) if path.exists() else {}
    for scen in scenarios:
        if scen in out:
            continue
        try:
            out[scen] = classify_scenario(scen, df, router)
            print(f"  {scen}: классифицировано {len(out[scen]['transactions'])} операций, "
                  f"связанных сторон {len(out[scen]['related_party_counterparties'])}")
        except Exception as e:
            print(f"  !! {scen}: классификация не удалась: {e}")
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def category_summary(classification: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Суммы по категориям — для проверки глазами и для промпта компилятора."""
    from decimal import Decimal

    rows = []
    for txn_id, t in classification["transactions"].items():
        amt = df.loc[df["txn_id"] == txn_id, "amount_usd"]
        rows.append({"category": t["category"],
                     "related": t["is_related_party"],
                     "amount": amt.iloc[0] if len(amt) else Decimal(0)})
    out = pd.DataFrame(rows)
    return out.groupby("category")["amount"].agg(["count", "sum"])
