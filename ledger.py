"""Нормализатор реестра: master_ledger_2025.csv -> DataFrame + amount_usd.

Курсы валют к USD заданы в документах датасета, а не в реестре.
Публичный набор: аудиторский отчёт data/public/documents/6f1c06f8479a.pdf,
«Примечание 9 — Валютные курсы»: суммы в валютах, отличных от USD,
пересчитываются по курсу фактического расчёта; приведён расчёт
72,146.75 EUR, урегулированный платежом $83,690.23 => 1 EUR = 1.16 USD (точно).
"""

from decimal import Decimal
from pathlib import Path

import pandas as pd

import paths

# запасные курсы, если из документов извлечь не удалось (публичный датасет)
FALLBACK_FX: dict[str, Decimal] = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.16"),
}

# TXN-P1-0039 -> P1; порядковый номер в конце отбрасывается
TXN_ID_RE = r"^TXN-(.+)-\d+$"


def load_ledger(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or paths.ledger(), dtype=str)
    # «грязные» строки: пустая сумма означает, что фактическая сумма
    # раскрыта в документах (поправка придёт из спецификации ковенанта)
    df["amount_missing"] = df["amount"].isna()
    df["amount"] = df["amount"].fillna("0").map(Decimal)

    df["scenario_id"] = df["txn_id"].str.extract(TXN_ID_RE)
    if df["scenario_id"].isna().any():
        bad = df.loc[df["scenario_id"].isna(), "txn_id"].tolist()
        raise ValueError(f"txn_id вне формата TXN-<scenario>-<NNNN>: {bad[:5]}")
    df["scenario_id"] = _align_to_template(df["scenario_id"])

    rates = fx_rates(set(df["currency"]))
    by_scen = _scenario_rates(set(df["currency"]))
    df["amount_usd"] = df.apply(
        lambda r: r["amount"] * by_scen.get(r["scenario_id"], {}).get(
            r["currency"], rates[r["currency"]]), axis=1
    )
    return df


def _scenario_rates(currencies: set[str]) -> dict[str, dict[str, Decimal]]:
    """Курс, раскрытый в документах самого заёмщика, важнее общего."""
    import fx

    try:
        by_scen = fx.extract_rates_by_scenario(currencies)
    except Exception as e:
        print(f"  !! курсы по сценариям недоступны: {e}")
        return {}
    if by_scen:
        pairs = {f"{s}:{c}={v:.4f}" for s, cur in by_scen.items() for c, v in cur.items()}
        print(f"  курсы заёмщиков: {', '.join(sorted(pairs))}")
    return by_scen


def fx_rates(currencies: set[str]) -> dict[str, Decimal]:
    """Курсы из документов; чего не нашлось — из запасной таблицы."""
    import fx

    try:
        rates = fx.extract_rates(currencies)
    except Exception as e:
        print(f"  !! извлечение курсов упало: {e}")
        rates = {"USD": Decimal("1")}
    for code in currencies - set(rates):
        if code in FALLBACK_FX:
            print(f"  !! курс {code} не найден в документах, беру запасной "
                  f"{FALLBACK_FX[code]}")
            rates[code] = FALLBACK_FX[code]
        else:
            raise ValueError(
                f"Нет курса к USD для валюты {code}: не найден в документах и "
                f"нет запасного. Считать в валюте операции нельзя — промах в разы.")
    return rates


def _align_to_template(series: pd.Series) -> pd.Series:
    """Приводит извлечённый префикс к сценарию из шаблона.

    Часть заёмщиков нумерует операции составно: TXN-KC-CAP-29 — сценарий «KC»,
    а «CAP» уже категория. Берём самое длинное совпадение с шаблоном.
    """
    import json

    try:
        scenarios = set(json.load(open(paths.template()))["answers"])
    except Exception:
        return series
    cache: dict[str, str] = {}

    def fix(prefix: str) -> str:
        if prefix in cache:
            return cache[prefix]
        best = prefix
        if prefix not in scenarios:
            matches = [s for s in scenarios if prefix.startswith(s + "-") or prefix == s]
            if matches:
                best = max(matches, key=len)
        cache[prefix] = best
        return best

    return series.map(fix)


def account_map(df: pd.DataFrame) -> dict[str, str]:
    """account_id -> scenario_id; падает, если соответствие неоднозначно."""
    pairs = df[["account_id", "scenario_id"]].drop_duplicates()
    dupes = pairs[pairs.duplicated("account_id", keep=False)]
    if not dupes.empty:
        raise ValueError(f"account_id с несколькими сценариями:\n{dupes}")
    return dict(pairs.itertuples(index=False))


if __name__ == "__main__":
    df = load_ledger()
    acc_map = account_map(df)

    print("Колонки:", list(df.columns))
    print("Строк:", len(df))
    print("Валюты:", sorted(df["currency"].unique()))
    print("Сценариев:", df["scenario_id"].nunique())
    print("Счетов (account_id -> scenario_id однозначно):", len(acc_map))
    print(df.head(5).to_string())
