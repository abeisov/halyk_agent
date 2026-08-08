"""Расчёт ячеек по спецификациям: actual + status + контрфактическая улика.

Чистый pandas/Decimal, ноль LLM. Числа в submission приходят только отсюда.

actual: сумма amount_usd отобранных транзакций по модулю (для коэффициента —
отношение модулей сумм), 2 знака. status: сравнение с порогом с учётом
триггера применимости и carve-out. Улика — единственная транзакция, чьё
удаление меняет вердикт; для коэффициентов и агрегатов — null.
"""

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

from ledger import load_ledger

SPECS_JSON = Path("data/processed/specs.json")

TWO = Decimal("0.01")


def _sum_usd(df: pd.DataFrame, txn_ids: list[str]) -> Decimal:
    sel = df[df["txn_id"].isin(txn_ids)]
    return abs(sum(sel["amount_usd"], Decimal(0)))


def _breached(actual: Decimal, threshold: Decimal, direction: str) -> bool:
    return actual > threshold if direction == "max" else actual < threshold


def _verdict(actual: Decimal, spec: dict) -> str:
    if not spec.get("trigger_active", True):
        return "COMPLIANT"
    if _breached(actual, Decimal(str(spec["threshold"])), spec["direction"]):
        return "COMPLIANT" if spec.get("carve_out_satisfied") else "BREACH"
    return "COMPLIANT"


def _compute_actual(df: pd.DataFrame, spec: dict) -> Decimal:
    num = _sum_usd(df, spec["relevant_txn_ids"])
    if spec["is_ratio"]:
        den = _sum_usd(df, spec["denominator_txn_ids"])
        if den == 0:
            raise ZeroDivisionError("пустой знаменатель коэффициента")
        return (num / den).quantize(TWO, ROUND_HALF_UP)
    return num.quantize(TWO, ROUND_HALF_UP)


def _evidence(df: pd.DataFrame, spec: dict, base_status: str) -> str | None:
    """Контрфактически: удаление какой одной транзакции меняет вердикт."""
    if spec["is_ratio"]:
        return None
    candidates = []
    for txn_id in spec["relevant_txn_ids"]:
        rest = [t for t in spec["relevant_txn_ids"] if t != txn_id]
        alt = _verdict(_sum_usd(df, rest).quantize(TWO, ROUND_HALF_UP), spec)
        if alt != base_status:
            candidates.append(txn_id)
    return candidates[0] if len(candidates) == 1 else None


def evaluate_cell(df: pd.DataFrame, spec: dict) -> dict:
    actual = _compute_actual(df, spec)
    status = _verdict(actual, spec)
    return {
        "status": status,
        "actual": float(actual),
        "evidence_txn_id": _evidence(df, spec, status),
    }


def evaluate_all(specs: dict, df: pd.DataFrame) -> dict:
    """scenario -> clause -> ячейка (или None, если спека нет/расчёт упал)."""
    out: dict[str, dict[str, dict | None]] = {}
    for scen, clauses in specs.items():
        sdf = df[df["scenario_id"] == scen]
        for clause, spec in clauses.items():
            try:
                cell = evaluate_cell(sdf, spec) if spec else None
            except Exception as e:
                print(f"  !! {scen} {clause}: {e}")
                cell = None
            out.setdefault(scen, {})[clause] = cell
    return out


if __name__ == "__main__":
    specs = json.load(open(SPECS_JSON))
    cells = evaluate_all(specs, load_ledger())
    for scen, clauses in sorted(cells.items()):
        for clause, cell in sorted(clauses.items()):
            if cell is None:
                print(f"{scen}:{clause}  FAIL")
            else:
                print(f"{scen}:{clause}  {cell['status']:10s} "
                      f"{cell['actual']:>15,.2f}  {cell['evidence_txn_id']}")
