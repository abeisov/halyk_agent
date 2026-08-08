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


def _apply_amendments(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Суммы, раскрытые в документах вместо пустых/неверных строк реестра."""
    amendments = spec.get("ledger_amendments") or []
    if not amendments:
        return df
    df = df.copy()
    for a in amendments:
        df.loc[df["txn_id"] == a["txn_id"], "amount_usd"] = Decimal(str(a["amount_usd"]))
    return df


def _signed_sum(df: pd.DataFrame, txn_ids: list[str]) -> Decimal:
    sel = df[df["txn_id"].isin(txn_ids)]
    return sum(sel["amount_usd"], Decimal(0))


def _sum_usd(df: pd.DataFrame, txn_ids: list[str], extra: list | None = None) -> Decimal:
    total = _signed_sum(df, txn_ids)
    for x in (extra or []):  # внекнижные суммы из документов
        total += Decimal(str(x))
    return abs(total)


def _breached(actual: Decimal, threshold: Decimal, direction: str) -> bool:
    return actual > threshold if direction == "max" else actual < threshold


def _verdict(actual: Decimal, spec: dict) -> str:
    if not spec.get("trigger_active", True):
        return "COMPLIANT"
    if _breached(actual, Decimal(str(spec["threshold"])), spec["direction"]):
        return "COMPLIANT" if spec.get("carve_out_satisfied") else "BREACH"
    return "COMPLIANT"


def _compute_actual(df: pd.DataFrame, spec: dict) -> Decimal:
    df = _apply_amendments(df, spec)
    if spec.get("aggregation") == "max":
        # ковенант проверяется по наибольшей из статей, а не по их сумме
        vals = [abs(v) for v in df[df["txn_id"].isin(spec["relevant_txn_ids"])]["amount_usd"]]
        vals += [abs(Decimal(str(x))) for x in (spec.get("off_ledger_amounts_usd") or [])]
        return max(vals).quantize(TWO, ROUND_HALF_UP) if vals else Decimal(0)
    num = _sum_usd(df, spec["relevant_txn_ids"], spec.get("off_ledger_amounts_usd"))
    if spec["is_ratio"]:
        den = _sum_usd(df, spec["denominator_txn_ids"],
                       spec.get("denominator_off_ledger_usd"))
        if den == 0:
            raise ZeroDivisionError("пустой знаменатель коэффициента")
        return (num / den).quantize(TWO, ROUND_HALF_UP)
    return num.quantize(TWO, ROUND_HALF_UP)


def _evidence(df: pd.DataFrame, spec: dict, base_status: str) -> str | None:
    """Контрфактически: удаление какой одной транзакции меняет вердикт."""
    df = _apply_amendments(df, spec)
    candidates = []
    for txn_id in spec["relevant_txn_ids"]:
        alt = dict(spec, relevant_txn_ids=[t for t in spec["relevant_txn_ids"] if t != txn_id])
        if _verdict(_compute_actual(df, alt), alt) != base_status:
            candidates.append(txn_id)
    # исключённая по предписанию документов операция — улика, если её
    # ВКЛЮЧЕНИЕ обратно меняет вердикт
    for txn_id in spec.get("excluded_txn_ids") or []:
        alt = dict(spec, relevant_txn_ids=spec["relevant_txn_ids"] + [txn_id])
        if _verdict(_compute_actual(df, alt), alt) != base_status:
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
