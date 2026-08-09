"""Курсы валют к USD — из документов датасета, а не из захардкоженной таблицы.

Отдельной таблицы курсов организаторы не дают: курс восстанавливается из
раскрытия в аудите вида «счёт на сумму 72,146.75 EUR урегулирован платежом
в долларах США в размере $83,690.23» -> 1 EUR = 1.16 USD.

Извлечение детерминированное (регулярки), LLM не нужен.
"""

import re
from decimal import Decimal

import pymupdf

import paths

NUM = r"[\d][\d,\s]*(?:\.\d+)?"

# «... 72,146.75 EUR ... в размере $83,690.23» и обратный порядок
PATTERNS = [
    re.compile(rf"({NUM})\s*([A-Z]{{3}})\b[^.]{{0,160}}?\$\s*({NUM})"),
    re.compile(rf"\$\s*({NUM})[^.]{{0,160}}?({NUM})\s*([A-Z]{{3}})\b"),
]


def _num(s: str) -> Decimal:
    return Decimal(s.replace(",", "").replace(" ", ""))


def extract_rates_by_scenario(currencies: set[str] | None = None
                              ) -> dict[str, dict[str, Decimal]]:
    """scenario_id -> {currency -> курс}.

    Курс задан «по курсу фактического расчёта по операциям периода», поэтому у
    каждого заёмщика он свой: применять общую медиану ко всем нельзя.
    Документ привязывается к сценарию через doc_index.
    """
    import pandas as pd

    try:
        idx = pd.read_csv(paths.processed("doc_index.csv"))
    except Exception:
        return {}
    doc_scen = dict(zip(idx["doc_id"], idx["scenario_id"]))

    out: dict[str, dict[str, list[Decimal]]] = {}
    for pdf in sorted(paths.documents().glob("*.pdf")):
        scen = doc_scen.get(pdf.name)
        if not isinstance(scen, str):
            continue
        try:
            with pymupdf.open(pdf) as doc:
                text = "\n".join(p.get_text() for p in doc)
        except Exception:
            continue
        for i, pat in enumerate(PATTERNS):
            for m in pat.finditer(text):
                usd, amount, code = ((m.group(3), m.group(1), m.group(2)) if i == 0
                                     else (m.group(1), m.group(2), m.group(3)))
                if code == "USD" or (currencies is not None and code not in currencies):
                    continue
                try:
                    rate = _num(usd) / _num(amount)
                except (ArithmeticError, ValueError):
                    continue
                if Decimal("0.2") <= rate <= Decimal("5"):
                    out.setdefault(scen, {}).setdefault(code, []).append(rate)

    return {s: {c: sorted(v)[len(v) // 2] for c, v in cur.items()}
            for s, cur in out.items()}


def extract_rates(currencies: set[str] | None = None,
                  min_ratio: Decimal = Decimal("0.2"),
                  max_ratio: Decimal = Decimal("5")) -> dict[str, Decimal]:
    """currency -> курс к USD по раскрытиям в документах.

    currencies — валюты, встречающиеся в реестре; всё остальное отбрасывается
    как случайное совпадение трёх заглавных букв.
    """
    found: dict[str, list[Decimal]] = {}
    for pdf in sorted(paths.documents().glob("*.pdf")):
        try:
            with pymupdf.open(pdf) as doc:
                text = "\n".join(p.get_text() for p in doc)
        except Exception:
            continue
        for i, pat in enumerate(PATTERNS):
            for m in pat.finditer(text):
                usd, amount, code = ((m.group(3), m.group(1), m.group(2)) if i == 0
                                     else (m.group(1), m.group(2), m.group(3)))
                if code == "USD" or (currencies is not None and code not in currencies):
                    continue
                try:
                    rate = _num(usd) / _num(amount)
                except (ArithmeticError, ValueError):
                    continue
                if min_ratio <= rate <= max_ratio:  # отсев случайных пар чисел
                    found.setdefault(code, []).append(rate)

    rates = {"USD": Decimal("1")}
    for code, values in found.items():
        values.sort()
        rates[code] = values[len(values) // 2]  # медиана — устойчива к мусору
    return rates


if __name__ == "__main__":
    for code, rate in sorted(extract_rates().items()):
        print(f"1 {code} = {rate} USD")
