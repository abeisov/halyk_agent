"""Регрессия на публичном наборе: балл сейчас против сохранённого эталона.

Показывает не только итог, но и какие ячейки правка починила, а какие сломала —
без этого улучшения делаются вслепую.

    python regress.py            # сравнить с эталоном
    python regress.py --save     # запомнить текущий результат как эталон
"""

import json
import subprocess
import sys
from pathlib import Path

BASELINE = Path("data/processed/regress_baseline.json")
OUT = Path("/tmp/regress_current.json")


def current_cells() -> dict[str, float]:
    """cell -> балл, посчитанный тем же кодом, что и score.py."""
    subprocess.run([sys.executable, "run.py", "--input", "data/public",
                    "--output", str(OUT), "--skip-llm"],
                   capture_output=True, check=False)
    sys.path.insert(0, ".")
    from score import score_cell

    sub = json.load(open(OUT))["answers"]
    gt = json.load(open("data/public/ground_truth.json"))["scenarios"]
    out = {}
    for scen, cov in gt.items():
        for clause, key in cov["covenants"].items():
            s = score_cell(sub.get(scen, {}).get(clause, {}), key)
            out[f"{scen}:{clause}"] = round(s["status"] + s["actual"] + s["evidence"], 3)
    return out


def main() -> None:
    cells = current_cells()
    total = sum(cells.values())
    print(f"СЕЙЧАС: {total:.2f} / {len(cells)}  (ячеек с полным баллом: "
          f"{sum(1 for v in cells.values() if v >= 0.999)})")

    if "--save" in sys.argv:
        BASELINE.write_text(json.dumps(cells, ensure_ascii=False, indent=2))
        print(f"эталон сохранён: {BASELINE}")
        return

    if not BASELINE.exists():
        print("эталона нет — сохрани его: python regress.py --save")
        return

    base = json.loads(BASELINE.read_text())
    fixed = [(c, base[c], v) for c, v in cells.items() if v > base.get(c, 0) + 1e-9]
    broke = [(c, base[c], v) for c, v in cells.items() if v < base.get(c, 0) - 1e-9]
    print(f"ЭТАЛОН: {sum(base.values()):.2f}  ->  разница {total - sum(base.values()):+.2f}\n")
    for label, rows in (("ПОЧИНИЛОСЬ", fixed), ("СЛОМАЛОСЬ", broke)):
        print(f"{label}: {len(rows)}")
        for c, was, now in sorted(rows, key=lambda r: r[1] - r[2]):
            print(f"   {c:10s} {was:.2f} -> {now:.2f}")


if __name__ == "__main__":
    main()
