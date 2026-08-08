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

LEDGER_CSV = Path("data/public/master_ledger_2025.csv")

# currency -> курс к USD (источник: см. докстринг модуля)
FX_RATES: dict[str, Decimal] = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.16"),
}

# TXN-P1-0039 -> P1; порядковый номер в конце отбрасывается
TXN_ID_RE = r"^TXN-(.+)-\d+$"


def load_ledger(path: Path = LEDGER_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # «грязные» строки: пустая сумма означает, что фактическая сумма
    # раскрыта в документах (поправка придёт из спецификации ковенанта)
    df["amount_missing"] = df["amount"].isna()
    df["amount"] = df["amount"].fillna("0").map(Decimal)

    df["scenario_id"] = df["txn_id"].str.extract(TXN_ID_RE)
    if df["scenario_id"].isna().any():
        bad = df.loc[df["scenario_id"].isna(), "txn_id"].tolist()
        raise ValueError(f"txn_id вне формата TXN-<scenario>-<NNNN>: {bad[:5]}")

    unknown = set(df["currency"]) - set(FX_RATES)
    if unknown:
        raise ValueError(f"Нет курса к USD для валют: {sorted(unknown)}")
    df["amount_usd"] = df.apply(
        lambda r: r["amount"] * FX_RATES[r["currency"]], axis=1
    )
    return df


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
