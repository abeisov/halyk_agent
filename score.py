"""Замер submission.json по ground_truth.json (формула из ТЗ, часть I.4).

Ячейка: status 0.50 (ошибка обнуляет ячейку целиком),
actual 0.30 x max(0, 1 - e/0.05), e = |ваше - ключ| / |ключ|,
evidence 0.20: точный ID; при null в ключе — убывает по шкале actual.

python score.py [submission.json] [ground_truth.json]
"""

import json
import sys


def score_cell(sub: dict, key: dict) -> dict:
    out = {"status": 0.0, "actual": 0.0, "evidence": 0.0, "e": None}

    if sub.get("status") != key["status"]:
        return out  # неверный вердикт обнуляет ячейку
    out["status"] = 0.50

    actual = sub.get("actual")
    decay = 0.0
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        e = abs(actual - key["actual"]) / abs(key["actual"])
        out["e"] = e
        decay = max(0.0, 1 - e / 0.05)
    out["actual"] = 0.30 * decay

    if key["evidence_txn_id"] is None:
        out["evidence"] = 0.20 * decay
    elif sub.get("evidence_txn_id") == key["evidence_txn_id"]:
        out["evidence"] = 0.20
    return out


def main(sub_path: str, gt_path: str) -> None:
    sub = json.load(open(sub_path))["answers"]
    gt = json.load(open(gt_path))["scenarios"]

    rows, miss_status, errs, ev_hit, ev_total = [], 0, [], 0, 0
    for scen, cov in gt.items():
        for clause, key in cov["covenants"].items():
            cell = sub.get(scen, {}).get(clause, {})
            s = score_cell(cell, key)
            total = s["status"] + s["actual"] + s["evidence"]
            rows.append((scen, clause, total, s, key, cell))
            if s["status"] == 0:
                miss_status += 1
            if s["e"] is not None:
                errs.append(s["e"])
            if key["evidence_txn_id"] is not None:
                ev_total += 1
                ev_hit += s["evidence"] == 0.20

    n = len(rows)
    print(f"ОБЩИЙ БАЛЛ: {sum(r[2] for r in rows):.3f} / {n}  "
          f"(средний по ячейке {sum(r[2] for r in rows) / n:.3f})")
    print(f"status: {n - miss_status}/{n} верных")
    if errs:
        print(f"actual: средняя отн. ошибка {sum(errs) / len(errs):.4f} "
              f"(по {len(errs)} ячейкам с верным status и числовым actual)")
    print(f"evidence: {ev_hit}/{ev_total} точных попаданий (ячейки с не-null ключом)")

    print(f"\n{'ячейка':10s} {'балл':>5s}  {'status':>6s} {'actual':>6s} {'evid':>5s}   разбор")
    for scen, clause, total, s, key, cell in sorted(rows, key=lambda r: r[2]):
        note = ""
        if s["status"] == 0:
            note = f"status: у нас {cell.get('status')!r}, ключ {key['status']!r}"
        elif s["e"] and s["e"] > 0.001:
            note = f"e={s['e']:.3f}: у нас {cell.get('actual')}, ключ {key['actual']}"
        if key["evidence_txn_id"] and s["status"] > 0 and s["evidence"] == 0:
            note += f" | evid: у нас {cell.get('evidence_txn_id')}, ключ {key['evidence_txn_id']}"
        print(f"{scen}:{clause:6s} {total:5.2f}  {s['status']:6.2f} {s['actual']:6.2f} "
              f"{s['evidence']:5.2f}   {note}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "submission.json",
         args[1] if len(args) > 1 else "data/public/ground_truth.json")
