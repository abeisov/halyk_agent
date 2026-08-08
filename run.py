"""Оркестратор: от папки данных до submission.json одной командой.

python run.py                      # полный прогон
python run.py --skip-llm           # без LLM (использует кэш specs.json, если есть)

Падение любого этапа не роняет прогон: незаполненные ячейки получают дефолт
в submit.py. Пустая ячейка = гарантированный ноль, дефолт = шанс на 0.50.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

import paths


def main() -> None:
    load_dotenv(".env")
    args = sys.argv[1:]
    if "--input" in args:
        paths.set_input(args[args.index("--input") + 1])
    submission = Path(args[args.index("--output") + 1]) if "--output" in args \
        else Path("submission.json")
    print(f"Датасет: {paths.INPUT_DIR}  ->  {submission}")

    print("[1/5] Реестр...")
    from ledger import load_ledger
    df = load_ledger()

    print("[2/5] Атрибуция документов...")
    try:
        from docs import build_doc_index
        build_doc_index().to_csv(paths.processed("doc_index.csv"), index=False)
    except Exception as e:
        print(f"  !! атрибуция упала: {e} (использую старый doc_index.csv)")

    print("[3/5] Компиляция ковенантов...")
    specs = {}
    try:
        from compile import extract_clauses, compile_specs
        clauses = extract_clauses()
        paths.processed("clauses.json").write_text(json.dumps(clauses, ensure_ascii=False, indent=2))
        if "--skip-llm" in sys.argv:
            specs = json.loads(paths.processed("specs.json").read_text()) if paths.processed("specs.json").exists() else {}
            print(f"  LLM пропущен, кэш: {sum(len(v) for v in specs.values())} спецификаций")
        else:
            specs = compile_specs(clauses, df)
    except SystemExit as e:
        print(f"  !! {e}")
        specs = json.loads(paths.processed("specs.json").read_text()) if paths.processed("specs.json").exists() else {}
    except Exception as e:
        print(f"  !! компиляция упала: {e}")
        specs = json.loads(paths.processed("specs.json").read_text()) if paths.processed("specs.json").exists() else {}

    print("[4/5] Расчёт ячеек...")
    from evaluate import evaluate_all
    cells = evaluate_all(specs, df)

    print("[5/5] Сборка и валидация...")
    from submit import build_submission, validate
    sub = build_submission(cells, submission)
    errors = validate(sub, df)
    if errors:
        print("ВАЛИДАЦИЯ ПРОВАЛЕНА:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {submission} собран и валиден.")

    if paths.ground_truth().exists():
        print("\n--- score по публичному GT ---")
        import score
        score.main(str(submission), str(paths.ground_truth()))


if __name__ == "__main__":
    main()
