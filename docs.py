"""Атрибуция PDF: documents/*.pdf -> scenario_id через account_id.

Имена файлов — хэши. Принадлежность определяется по содержимому:
регулярка ACC-\\d+ по тексту -> словарь account_id -> scenario_id из реестра.
Результат — data/processed/doc_index.csv.
"""

import re
from collections import Counter
from pathlib import Path

import pymupdf
import pandas as pd

import paths
from ledger import load_ledger, account_map

ACC_RE = re.compile(r"ACC-\d+")

# тип документа по характерным фразам первой страницы (эвристика, не критично)
DOC_TYPE_MARKERS = [
    ("loan_agreement", ["договор банковского займа", "кредитный договор", "loan agreement"]),
    ("audit_report", ["аудитор", "независимого аудитора", "audit"]),
    ("kyc", ["kyc", "досье", "know your customer"]),
    ("other", []),
]


def extract_text(pdf_path: Path) -> tuple[str, int]:
    with pymupdf.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc), doc.page_count


# маркеры недействующей (подменной) редакции документа
VOID_MARKERS = ["недействующая редакция", "не применяется", "superseded"]


def is_void(text: str) -> bool:
    head = text[:1500].lower()
    return any(m in head for m in VOID_MARKERS)


def guess_doc_type(text: str) -> str:
    head = text[:3000].lower()
    for doc_type, markers in DOC_TYPE_MARKERS:
        if any(m in head for m in markers):
            return doc_type
    return "other"


def build_doc_index() -> pd.DataFrame:
    acc_map = account_map(load_ledger())
    rows = []
    for pdf in sorted(paths.documents().glob("*.pdf")):
        text, n_pages = extract_text(pdf)
        accs = Counter(ACC_RE.findall(text))
        # сценарии, на счета которых ссылается документ (по убыванию частоты)
        scens = []
        for acc, _ in accs.most_common():
            scen = acc_map.get(acc)
            if scen and scen not in scens:
                scens.append(scen)
        rows.append({
            "doc_id": pdf.name,
            "scenario_id": scens[0] if scens else None,
            "all_scenarios": ";".join(scens),
            "accounts_found": ";".join(a for a, _ in accs.most_common()),
            "doc_type": guess_doc_type(text),
            "is_void": is_void(text),
            "pages": n_pages,
            "chars": len(text),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json

    df = build_doc_index()
    out_csv = paths.processed("doc_index.csv")
    df.to_csv(out_csv, index=False)

    template = json.load(open(paths.template()))
    scenarios = list(template["answers"])

    print(f"Документов: {len(df)}, без атрибуции: {df['scenario_id'].isna().sum()}")
    print(f"Пустой текстовый слой: {(df['chars'] < 100).sum()}")
    print("\nТипы документов:")
    print(df["doc_type"].value_counts().to_string())

    print(f"Недействующих редакций (is_void): {df['is_void'].sum()}")

    print("\nПокрытие сценариев из шаблона (действующие документы):")
    attributed = df[df["scenario_id"].notna() & ~df["is_void"]]
    for s in scenarios:
        mine = attributed[attributed["scenario_id"] == s]
        types = dict(mine["doc_type"].value_counts())
        n_la = types.get("loan_agreement", 0)
        flag = "" if n_la == 1 else f"  <-- договоров: {n_la}, ожидался 1"
        print(f"  {s:5s} документов: {len(mine):2d}  {types}{flag}")

    outside = attributed[~attributed["scenario_id"].isin(scenarios)]
    print(f"\nДокументы вне шаблонных сценариев: {len(outside)}")
    print(f"Индекс записан: {out_csv}")
