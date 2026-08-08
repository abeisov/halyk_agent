"""Пути датасета в одном месте: переключаются на приватный набор одним флагом.

По умолчанию — публичный датасет. Боевой прогон:
    python run.py --input data/private
"""

from pathlib import Path

INPUT_DIR = Path("data/public")
PROCESSED_DIR = Path("data/processed")


def set_input(path: str | Path) -> None:
    """Переключает датасет; производные файлы у каждого набора свои."""
    global INPUT_DIR, PROCESSED_DIR
    INPUT_DIR = Path(path)
    PROCESSED_DIR = Path("data/processed") / INPUT_DIR.name
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Папки датасета нет: {INPUT_DIR}")
    missing = [f for f in (ledger(), template(), documents()) if not f.exists()]
    if missing:
        raise SystemExit(f"В {INPUT_DIR} не хватает: {[str(m) for m in missing]}")


def ledger() -> Path:
    """Реестр транзакций — единственный CSV в папке датасета."""
    named = INPUT_DIR / "master_ledger_2025.csv"
    if named.exists():
        return named
    csvs = sorted(INPUT_DIR.glob("*.csv"))
    return csvs[0] if csvs else named


def template() -> Path:
    return INPUT_DIR / "submission_template.json"


def documents() -> Path:
    return INPUT_DIR / "documents"


def ground_truth() -> Path:
    return INPUT_DIR / "ground_truth.json"


def processed(name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / name
