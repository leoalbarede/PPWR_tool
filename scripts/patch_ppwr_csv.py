"""
Patch ppwr_audit_results.csv without re-running the LLM audit.

- Correct inverted PFAS answers when evidence states compliance (limit met).
- Recover PFAS findings from PDF text when CSV is N/A but documents support it.
- Add Supplier folder column to disambiguate duplicate supplier numbers.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader

# Make the repo root importable when run as scripts/patch_ppwr_csv.py.
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evidence_validator import (  # noqa: E402
    CHECK_PFAS,
    correct_inverted_raw_answer,
    enrich_pass_through_pfas_evidence,
    evidence_declarant_mismatch,
    evidence_indicates_compliant,
    evidence_matches_check,
    recover_pfas_from_markdown,
    validate_point_finding,
)

_PASS_THROUGH_DECLARATION_RE = re.compile(
    r"present\s+declaration\s+is\s+valid\s+for\s+material",
    re.IGNORECASE,
)

DOCS_DIR = os.path.join(ROOT_DIR, "docs")
DEFAULT_CSV = os.path.join(ROOT_DIR, "data", "ppwr_audit_results.csv")

SUPPLIER_FOLDER_RE = re.compile(r"^(?P<supplier_no>.+?)\s*-\s*(?P<supplier>.+)$")


def _extract_supplier_no(raw_prefix: str) -> str:
    prefix = raw_prefix.strip()
    if "_" in prefix:
        return prefix.rsplit("_", 1)[-1].strip()
    return prefix


def parse_supplier_folder(folder_name: str) -> Optional[Tuple[str, str]]:
    match = SUPPLIER_FOLDER_RE.match(folder_name.strip())
    if not match:
        return None
    supplier_no = _extract_supplier_no(match.group("supplier_no"))
    return supplier_no, match.group("supplier").strip()


def _collect_pdfs(folder_path: str) -> List[str]:
    return sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    )


def discover_suppliers(docs_dir: str = DOCS_DIR) -> List[Tuple[str, str, str, str, List[str]]]:
    suppliers: List[Tuple[str, str, str, str, List[str]]] = []
    for entry in sorted(os.listdir(docs_dir)):
        folder_path = os.path.join(docs_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        parsed = parse_supplier_folder(entry)
        if parsed:
            supplier_no, supplier_name = parsed
            suppliers.append(("", entry.strip(), supplier_no, supplier_name, _collect_pdfs(folder_path)))
            continue
        doc_list = entry.strip()
        for sub_entry in sorted(os.listdir(folder_path)):
            sub_path = os.path.join(folder_path, sub_entry)
            if not os.path.isdir(sub_path):
                continue
            sub_parsed = parse_supplier_folder(sub_entry)
            if not sub_parsed:
                continue
            supplier_no, supplier_name = sub_parsed
            suppliers.append(
                (doc_list, sub_entry.strip(), supplier_no, supplier_name, _collect_pdfs(sub_path))
            )
    return suppliers


def pdf_to_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception as exc:
        print(f"  ⚠️  Could not read PDF {os.path.basename(pdf_path)}: {exc}")
        return ""


def _source_hints(row: Dict[str, str]) -> List[str]:
    hints: List[str] = []
    for key, val in row.items():
        if "source document" in key.lower() and val and val not in ("NONE", "nan"):
            hints.append(val.strip())
    return hints


def assign_supplier_folders(
    rows: List[Dict[str, str]],
    suppliers: List[Tuple[str, str, str, str, List[str]]],
) -> None:
    by_key: Dict[Tuple[str, str, str], List[Tuple[str, List[str]]]] = defaultdict(list)
    for doc_list, folder, sup_no, sup_name, pdfs in suppliers:
        by_key[(doc_list, sup_no, sup_name)].append((folder, pdfs))

    assigned: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    for row in rows:
        key = (row.get("Doc list", ""), row.get("Supplier No.", ""), row.get("Supplier", ""))
        options = by_key.get(key, [])
        if not options:
            row.setdefault("Supplier folder", "")
            continue
        if row.get("Supplier folder"):
            continue
        if len(options) == 1:
            row["Supplier folder"] = options[0][0]
            assigned[key].append(options[0][0])
            continue

        hints = _source_hints(row)
        for folder, pdfs in options:
            if folder in assigned[key]:
                continue
            pdf_names = {os.path.basename(p) for p in pdfs}
            if hints and any(h in pdf_names for h in hints):
                row["Supplier folder"] = folder
                assigned[key].append(folder)
                break

        if not row.get("Supplier folder"):
            for folder, _pdfs in options:
                if folder not in assigned[key]:
                    row["Supplier folder"] = folder
                    assigned[key].append(folder)
                    break


def patch_row_pfas(
    row: Dict[str, str],
    pdf_lookup: Dict[Tuple[str, str, str, str], List[str]],
) -> None:
    pfas_ev = (row.get("PFAS evidence") or "").strip()
    pfas_raw = (row.get("PPWR PFAS content") or "N/A").strip()
    sup_name = (row.get("Supplier") or "").strip()

    doc_list = row.get("Doc list", "")
    folder = row.get("Supplier folder", "")
    sup_no = row.get("Supplier No.", "")
    pdfs = pdf_lookup.get((doc_list, folder, sup_no, sup_name), [])
    if not pdfs:
        for key, paths in pdf_lookup.items():
            if key[0] == doc_list and key[2] == sup_no and key[3] == sup_name:
                pdfs = paths
                row["Supplier folder"] = key[1]
                break

    if pdfs and sup_name:
        for pdf_path in pdfs:
            markdown = pdf_to_text(pdf_path)
            if not _PASS_THROUGH_DECLARATION_RE.search(markdown):
                continue
            candidate = enrich_pass_through_pfas_evidence(
                pfas_ev if pfas_ev not in ("NONE", "nan", "") else "NONE",
                markdown,
                sup_name,
            )
            needs_better_quote = (
                not pfas_ev
                or pfas_ev in ("NONE", "nan")
                or not evidence_matches_check(pfas_ev, CHECK_PFAS)
                or (
                    candidate
                    and candidate not in ("NONE", "nan", "")
                    and candidate != pfas_ev
                )
            )
            if needs_better_quote:
                enriched = candidate
                if enriched and enriched not in ("NONE", "nan", ""):
                    pfas_ev = enriched
                    row["PFAS evidence"] = enriched
                    row["PFAS source document"] = os.path.basename(pdf_path)
                    break

    if pfas_ev and pfas_ev not in ("NONE", "nan"):
        markdown_for_check = ""
        if pdfs:
            markdown_for_check = pdf_to_text(pdfs[0])
        if sup_name and evidence_declarant_mismatch(pfas_ev, sup_name, markdown_for_check):
            row["PPWR PFAS content"] = "N/A"
            row["PFAS evidence"] = "NONE"
            row["PFAS source document"] = ""
            return
        row["PPWR PFAS content"] = correct_inverted_raw_answer(CHECK_PFAS, pfas_raw, pfas_ev)
        return

    for pdf_path in pdfs:
        markdown = pdf_to_text(pdf_path)
        recovered = recover_pfas_from_markdown(markdown, os.path.basename(pdf_path))
        if recovered is None:
            continue
        validated = validate_point_finding(recovered, markdown, CHECK_PFAS, sup_name)
        if validated.answer not in ("yes", "no"):
            continue
        row["PPWR PFAS content"] = correct_inverted_raw_answer(
            CHECK_PFAS, validated.answer, validated.evidence
        )
        row["PFAS evidence"] = validated.evidence
        row["PFAS source document"] = validated.source_document
        break


def patch_csv(csv_path: str = DEFAULT_CSV, docs_dir: str = DOCS_DIR) -> int:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "Supplier folder" not in fieldnames:
        insert_at = fieldnames.index("Supplier") if "Supplier" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "Supplier folder")

    suppliers = discover_suppliers(docs_dir)
    assign_supplier_folders(rows, suppliers)
    pdf_lookup = {
        (doc_list, folder, sup_no, sup_name): pdfs
        for doc_list, folder, sup_no, sup_name, pdfs in suppliers
    }

    changed = 0
    for row in rows:
        before = (row.get("PPWR PFAS content"), row.get("PFAS evidence"), row.get("Supplier folder"))
        patch_row_pfas(row, pdf_lookup)
        after = (row.get("PPWR PFAS content"), row.get("PFAS evidence"), row.get("Supplier folder"))
        if before != after:
            changed += 1

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Patched {changed} row(s). CSV written: {csv_path} ({len(rows)} suppliers)")
    return changed


if __name__ == "__main__":
    patch_csv()
