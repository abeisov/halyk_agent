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

DOCS_DIR = Path("data/public/documents")
DOC_INDEX = Path("data/processed/doc_index.csv")
CLAUSES_JSON = Path("data/processed/clauses.json")
SPECS_JSON = Path("data/processed/specs.json")
TEMPLATE = Path("data/public/submission_template.json")

SECTION_RE = re.compile(r"Статья 6 — ")
NEXT_SECTION_RE = re.compile(r"Статья 7 — ")
CLAUSE_SPLIT_RE = re.compile(r"(?=Пункт 6\.\d+)")
CLAUSE_NUM_RE = re.compile(r"^Пункт (6\.\d+)")


# ---------- этап 1: тексты пунктов ----------

def extract_clauses() -> dict[str, dict[str, str]]:
    """scenario_id -> {clause -> дословный текст пункта}."""
    idx = pd.read_csv(DOC_INDEX)
    contracts = idx[(idx["doc_type"] == "loan_agreement") & (~idx["is_void"])]
    template_scenarios = list(json.load(open(TEMPLATE))["answers"])

    out: dict[str, dict[str, str]] = {}
    for _, row in contracts.iterrows():
        scen = row["scenario_id"]
        if scen not in template_scenarios:
            continue
        with pymupdf.open(DOCS_DIR / row["doc_id"]) as doc:
            text = "\n".join(p.get_text() for p in doc)
        m6 = SECTION_RE.search(text)
        m7 = NEXT_SECTION_RE.search(text, m6.end()) if m6 else None
        if not (m6 and m7):
            raise ValueError(f"{scen}: не нашёл границы Статьи 6 в {row['doc_id']}")
        body = text[m6.start():m7.start()]
        clauses = {}
        for chunk in CLAUSE_SPLIT_RE.split(body):
            m = CLAUSE_NUM_RE.match(chunk.strip())
            if m:
                clauses[m.group(1)] = " ".join(chunk.split())
        out[scen] = clauses

    missing = [s for s in template_scenarios if s not in out]
    if missing:
        raise ValueError(f"Сценарии без договора: {missing}")
    return out


# ---------- этап 2: LLM -> спецификация ----------

class CovenantSpec(BaseModel):
    """Что LLM извлекает из пункта. Числа из ответа LLM в submission не попадают:
    threshold — порог из текста договора, суммы считает pandas."""
    metric_description: str = Field(description="Что измеряет ковенант, одной фразой")
    is_ratio: bool = Field(description="True, если метрика — коэффициент, а не сумма в USD")
    threshold: float = Field(description="Числовой порог из текста пункта")
    direction: str = Field(description="'max' — нарушение при превышении порога, 'min' — при значении ниже порога")
    trigger_active: bool = Field(description="False только если ковенант применяется при условии (триггере), и это условие НЕ сработало")
    trigger_reasoning: str = Field(description="Если в пункте есть условие применимости — почему сработало/нет, иначе пустая строка")
    carve_out_satisfied: bool = Field(description="True только если превышение покрыто оговоркой (carve-out) из текста пункта")
    carve_out_reasoning: str = Field(description="Какая оговорка есть в пункте и выполнена ли, иначе пустая строка")
    relevant_txn_ids: list[str] = Field(description="Транзакции в скоупе метрики (числитель для коэффициента)")
    denominator_txn_ids: list[str] = Field(description="Транзакции знаменателя (только для коэффициентов, иначе пусто)")
    confidence: float = Field(description="Уверенность 0..1")


PROMPT = """Ты компилируешь пункт кредитного договора в машинную спецификацию.
Ниже дословный текст пункта и полный леджер транзакций этого заёмщика за период.

ПУНКТ ДОГОВОРА ({scenario} {clause}):
{clause_text}

ЛЕДЖЕР (txn_id, date, counterparty, description, amount, currency, amount_usd; расходы отрицательные):
{ledger_csv}

СПРАВОЧНЫЕ ДОКУМЕНТЫ ЗАЁМЩИКА (KYC — определяет связанные стороны; аудит — дополнение
о соблюдении ковенантов, переклассификации, валютные курсы):
{support_text}

Задача:
1. Определи метрику, порог (threshold) и направление (direction).
2. Отбери transaction ids, входящие в скоуп метрики (relevant_txn_ids) — по
   назначению платежа и контрагенту. Для коэффициентов отдельно знаменатель.
3. Внимательно проверь оговорки (carve-outs) и условия применимости (триггеры)
   в тексте пункта: превышение может быть допустимым, ковенант может не действовать.
4. Не считай суммы — только классифицируй. Верни строго JSON по схеме."""


def support_texts() -> dict[str, str]:
    """scenario_id -> текст действующих KYC и аудиторских отчётов (для промпта)."""
    idx = pd.read_csv(DOC_INDEX)
    docs = idx[idx["doc_type"].isin(["kyc", "audit_report"]) & (~idx["is_void"])]
    out: dict[str, str] = {}
    for _, row in docs.iterrows():
        scen = row["scenario_id"]
        if pd.isna(scen):
            continue
        with pymupdf.open(DOCS_DIR / row["doc_id"]) as doc:
            text = "\n".join(p.get_text() for p in doc)
        out[scen] = out.get(scen, "") + f"\n--- {row['doc_type']} {row['doc_id']} ---\n{text}"
    return {s: t[:60000] for s, t in out.items()}


def compile_specs(clauses: dict, df: pd.DataFrame) -> dict:
    import os
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Нет GEMINI_API_KEY в .env — этап LLM пропущен. "
                         "Ключ: aistudio.google.com")
    client = genai.Client(api_key=api_key)
    model = os.environ.get("MODEL_MAIN", "gemini-2.5-flash")

    specs: dict[str, dict[str, dict]] = {}
    if SPECS_JSON.exists():  # докомпиляция после падения — не пережигать вызовы
        specs = json.load(open(SPECS_JSON))

    support = support_texts()
    cols = ["txn_id", "date", "counterparty", "description", "amount", "currency", "amount_usd"]
    for scen, cls in clauses.items():
        ledger_csv = df[df["scenario_id"] == scen][cols].to_csv(index=False)
        for clause, clause_text in sorted(cls.items()):
            if specs.get(scen, {}).get(clause):
                continue
            prompt = PROMPT.format(scenario=scen, clause=clause,
                                   clause_text=clause_text, ledger_csv=ledger_csv,
                                   support_text=support.get(scen, "(нет)"))
            last_err = None
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={
                            "temperature": 0,
                            "response_mime_type": "application/json",
                            "response_schema": CovenantSpec,
                        },
                    )
                    spec = CovenantSpec.model_validate_json(resp.text)
                    specs.setdefault(scen, {})[clause] = spec.model_dump()
                    break
                except Exception as e:  # ретрай с паузой, потом дальше
                    last_err = e
                    time.sleep(5 * (attempt + 1))
            else:
                print(f"  !! {scen} {clause}: LLM не ответил: {last_err}")
            SPECS_JSON.write_text(json.dumps(specs, ensure_ascii=False, indent=2))
            print(f"  {scen} {clause}: "
                  f"{'ok' if specs.get(scen, {}).get(clause) else 'FAIL'}")
    return specs


if __name__ == "__main__":
    load_dotenv()
    clauses = extract_clauses()
    CLAUSES_JSON.parent.mkdir(parents=True, exist_ok=True)
    CLAUSES_JSON.write_text(json.dumps(clauses, ensure_ascii=False, indent=2))
    n = sum(len(v) for v in clauses.values())
    print(f"Пункты: {n} шт. по {len(clauses)} сценариям -> {CLAUSES_JSON}")

    if "--clauses-only" in sys.argv:
        sys.exit(0)

    specs = compile_specs(clauses, load_ledger())
    done = sum(1 for s in specs.values() for _ in s)
    print(f"Спецификаций: {done}/{n} -> {SPECS_JSON}")
