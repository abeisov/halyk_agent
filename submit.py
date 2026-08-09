"""Сборка submission.json из шаблона + валидация перед отправкой.

Сериализатор читает submission_template.json и заполняет существующие ячейки.
Ключи не создаются и не удаляются. Пустых ячеек не остаётся: для ячейки без
расчёта — дефолт (COMPLIANT — частотный класс публичного GT, actual=1.0).
"""

import json
import os
from pathlib import Path

import pandas as pd

import paths

DEFAULT_CELL = {"status": "COMPLIANT", "actual": 1.00, "evidence_txn_id": None}


def models_used(specs: dict) -> str:
    """Модели, фактически давшие ответы, — по убыванию числа ячеек.

    Цепочка фолбэка может задействовать несколько провайдеров, поэтому поле
    заполняется по факту, а не тем, что записано в настройках.
    """
    counts: dict[str, int] = {}
    for clauses in specs.values():
        for spec in clauses.values():
            tag = (spec or {}).get("_model")
            if not tag:
                continue
            name = tag.split("/", 1)[1] if "/" in tag else tag  # без метки ключа
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return os.environ.get("MODEL_MAIN", "gemini-3.6-flash")
    return ", ".join(sorted(counts, key=lambda m: -counts[m]))


def build_submission(cells: dict, out_path: Path, specs: dict | None = None) -> dict:
    sub = json.load(open(paths.template()))
    sub["team"] = os.environ.get("TEAM", "halyk-covenant-agent")
    sub["contact_email"] = os.environ.get("CONTACT_EMAIL", "anuar.beisov1992@gmail.com")
    sub["model"] = models_used(specs or {})

    fallbacks = []
    for scen, clauses in sub["answers"].items():
        for clause in clauses:
            cell = cells.get(scen, {}).get(clause)
            if not cell:
                cell = dict(DEFAULT_CELL)
                fallbacks.append(f"{scen}:{clause}")
            clauses[clause] = {
                "status": cell["status"],
                "actual": round(float(cell["actual"]), 2),
                "evidence_txn_id": cell["evidence_txn_id"],
            }
    if fallbacks:
        print(f"ДЕФОЛТЫ в {len(fallbacks)} ячейках: {', '.join(fallbacks)}")
    out_path.write_text(json.dumps(sub, ensure_ascii=False, indent=2))
    return sub


def validate(sub: dict, ledger_df: pd.DataFrame) -> list[str]:
    errors = []
    template = json.load(open(paths.template()))
    known_txns = set(ledger_df["txn_id"])

    for field in ("team", "contact_email", "model"):
        if not sub.get(field):
            errors.append(f"пустое поле {field}")

    if set(sub["answers"]) != set(template["answers"]):
        errors.append("набор сценариев не совпадает с шаблоном")
    for scen, clauses in template["answers"].items():
        if set(sub["answers"].get(scen, {})) != set(clauses):
            errors.append(f"{scen}: набор пунктов не совпадает с шаблоном")
            continue
        for clause, cell in sub["answers"][scen].items():
            where = f"{scen}:{clause}"
            if cell.get("status") not in ("COMPLIANT", "BREACH"):
                errors.append(f"{where}: status = {cell.get('status')!r}")
            a = cell.get("actual")
            if not isinstance(a, (int, float)) or isinstance(a, bool) or a <= 0:
                errors.append(f"{where}: actual = {a!r} (нужно положительное число)")
            elif round(a, 2) != a:
                errors.append(f"{where}: actual = {a} (больше 2 знаков)")
            ev = cell.get("evidence_txn_id")
            if ev is not None and ev not in known_txns:
                errors.append(f"{where}: evidence {ev!r} нет в реестре")
    return errors
