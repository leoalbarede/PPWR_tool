"""
PPWR supplier audit — binary compliance checks on regulatory PDFs per CMO folder.

Scans docs/<Supplier No> - <Supplier>/ for PDFs, runs hybrid RAG (analyzer_docling),
and exports a CSV with heavy-metals and SoC findings grounded in source text only.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from analyzer_docling import ask_pdf_multi_section

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "ppwr_audit_results.csv")

SUPPLIER_FOLDER_RE = re.compile(r"^(?P<supplier_no>.+?)\s*-\s*(?P<supplier>.+)$")

PPWR_INSTRUCTION = """
You are a PPWR regulatory document auditor. Answer ONLY from the CONTEXT passages below.

STRICT RULES (zero hallucination):
- Use ONLY explicit statements found in the CONTEXT. Do not infer, extrapolate, or use outside knowledge.
- If the CONTEXT does not explicitly support an answer, write N/A for that field.
- Evidence MUST be a verbatim quote copied from the CONTEXT (short excerpt, max 300 characters).
- If no quote supports the answer, set Answer to N/A and Evidence to NONE.
- Source document is the filename provided in the task header.
- For concentrations, copy the exact value and unit as written (e.g. mg/kg, ppm, mg). Do not convert units.

Output format — use EXACTLY these labels for each numbered point:

Point N:
Answer: YES | NO | N/A
Evidence: "<verbatim quote>" or NONE
Concentration: <exact value and unit from context> or N/A
Notes: <optional brief clarification strictly based on context, or NONE>
""".strip()

PPWR_SECTIONS: List[Tuple[str, str]] = [
    (
        "PPWR — Heavy metals (Pb, Cd, Hg, Cr6+)",
        """
1. Heavy metals concentration limit (PPWR): Is the sum of lead (Pb), cadmium (Cd), mercury (Hg),
   and hexavalent chromium (Cr6+) in packaging and its components below 100 mg (or 100 ppm / mg/kg
   as stated in the document)? Answer YES if explicitly compliant below the limit, NO if explicitly
   above or non-compliant, N/A if not stated.
   If YES or NO, report the total or individual concentrations when explicitly given.
""".strip(),
    ),
    (
        "PPWR — Substances of Concern (SoC)",
        """
2. Substances of Concern (SoC) in packaging materials and components: Is the presence of SoC
   explicitly mentioned (NOT PFAS — PFAS is a separate topic)? SoC includes substances of very
   high concern (SVHC), CMRs, or other restricted substances under PPWR Article 5 other than PFAS.
   Answer YES if SoC are present or declared above detection limits, NO if explicitly absent or
   not detected, N/A if SoC are not mentioned at all in the document.
   If YES or NO with a quantitative statement, report the concentration(s) exactly as written.
""".strip(),
    ),
]

CSV_COLUMNS = [
    "Supplier No.",
    "Supplier",
    "PPWR compliant with heavy metals concentration limit",
    "PPWR SoC content",
    "Concentration",
    "Heavy metals evidence",
    "Heavy metals source document",
    "SoC evidence",
    "SoC source document",
]

# User-facing export (core columns only)
CSV_COLUMNS_CORE = [
    "Supplier No.",
    "Supplier",
    "PPWR compliant with heavy metals concentration limit",
    "PPWR SoC content",
    "Concentration",
]


@dataclass
class PointFinding:
    answer: str = "N/A"
    evidence: str = "NONE"
    concentration: str = "N/A"
    notes: str = "NONE"
    source_document: str = ""


@dataclass
class PdfFinding:
    source_file: str
    heavy_metals: PointFinding = field(default_factory=PointFinding)
    soc: PointFinding = field(default_factory=PointFinding)


@dataclass
class SupplierFinding:
    supplier_no: str
    supplier: str
    pdf_findings: List[PdfFinding] = field(default_factory=list)


def _extract_supplier_no(raw_prefix: str) -> str:
    """S00122_0100114617 → 0100114617 (numeric ID after underscore)."""
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


def discover_suppliers(docs_dir: str = DOCS_DIR) -> List[Tuple[str, str, List[str]]]:
    """Return (supplier_no, supplier_name, pdf_paths) for each subfolder under docs/."""
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    suppliers: List[Tuple[str, str, List[str]]] = []
    for entry in sorted(os.listdir(docs_dir)):
        folder_path = os.path.join(docs_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        parsed = parse_supplier_folder(entry)
        if not parsed:
            print(f"⚠️  Skipping folder (unexpected name): {entry}")
            continue
        supplier_no, supplier_name = parsed
        pdfs = sorted(
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".pdf")
        )
        suppliers.append((supplier_no, supplier_name, pdfs))
    return suppliers


def _normalize_answer(raw: str) -> str:
    text = (raw or "").strip().upper()
    if text.startswith("YES"):
        return "yes"
    if text.startswith("NO"):
        return "no"
    return "N/A"


def _extract_section_body(answer_text: str, title_keywords: Tuple[str, ...]) -> str:
    """Extract content between === Section title === markers."""
    matches = list(re.finditer(r"===\s*(.+?)\s*===", answer_text))
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        if any(kw.lower() in title.lower() for kw in title_keywords):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(answer_text)
            return answer_text[start:end].strip()
    return ""


def _parse_point_block(block: str) -> PointFinding:
    finding = PointFinding()

    answer_m = re.search(r"Answer:\s*(YES|NO|N/A)\b", block, re.IGNORECASE)
    if answer_m:
        finding.answer = _normalize_answer(answer_m.group(1))

    evidence_m = re.search(r'Evidence:\s*("([^"]*)"|NONE)', block, re.IGNORECASE | re.DOTALL)
    if evidence_m:
        if evidence_m.group(1).upper() == "NONE":
            finding.evidence = "NONE"
        else:
            finding.evidence = (evidence_m.group(2) or "").strip() or "NONE"

    conc_m = re.search(r"Concentration:\s*(.+?)(?:\nNotes:|\Z)", block, re.IGNORECASE | re.DOTALL)
    if conc_m:
        conc = conc_m.group(1).strip()
        if conc.lower().startswith("<exact") or conc.lower() == "n/a":
            finding.concentration = "N/A"
        else:
            finding.concentration = conc

    notes_m = re.search(r"Notes:\s*(.+?)(?:\nPoint|\Z)", block, re.IGNORECASE | re.DOTALL)
    if notes_m:
        finding.notes = notes_m.group(1).strip()

    if finding.answer == "N/A" or finding.evidence == "NONE":
        finding.answer = "N/A"
        finding.concentration = "N/A"

    return finding


def parse_rag_answer(answer_text: str, source_file: str) -> PdfFinding:
    """Parse LLM section output into structured heavy-metals and SoC findings."""
    result = PdfFinding(source_file=source_file)

    hm_body = _extract_section_body(answer_text, ("Heavy metals", "Pb, Cd, Hg"))
    soc_body = _extract_section_body(answer_text, ("SoC", "Substances of Concern"))

    if hm_body:
        result.heavy_metals = _parse_point_block(hm_body)
        result.heavy_metals.source_document = source_file
    if soc_body:
        result.soc = _parse_point_block(soc_body)
        result.soc.source_document = source_file

    return result


def analyze_pdf(pdf_path: str, api_key: str, k: int = 5) -> PdfFinding:
    source_file = os.path.basename(pdf_path)
    instruction = (
        f"{PPWR_INSTRUCTION}\n\n"
        f"SOURCE DOCUMENT FILENAME: {source_file}\n"
    )
    rag = ask_pdf_multi_section(
        pdf_path=pdf_path,
        instruction=instruction,
        sections=PPWR_SECTIONS,
        api_key=api_key,
        k=k,
    )
    return parse_rag_answer(rag.get("answer", ""), source_file)


def _pick_best(findings: List[PointFinding]) -> PointFinding:
    """Prefer answers backed by evidence; YES/NO over N/A."""
    ranked = sorted(
        findings,
        key=lambda f: (
            0 if f.answer in ("yes", "no") and f.evidence not in ("", "NONE") else 1,
            0 if f.answer in ("yes", "no") else 1,
        ),
    )
    return ranked[0] if ranked else PointFinding()


def _format_concentration(hm: PointFinding, soc: PointFinding, with_citations: bool) -> str:
    parts: List[str] = []
    if hm.concentration not in ("", "N/A"):
        part = f"Heavy metals: {hm.concentration}"
        if with_citations and hm.evidence not in ("", "NONE"):
            part += f' [{hm.source_document}: "{hm.evidence}"]'
        parts.append(part)
    if soc.concentration not in ("", "N/A"):
        part = f"SoC: {soc.concentration}"
        if with_citations and soc.evidence not in ("", "NONE"):
            part += f' [{soc.source_document}: "{soc.evidence}"]'
        parts.append(part)
    if not parts and with_citations:
        cite_parts: List[str] = []
        if hm.evidence not in ("", "NONE"):
            cite_parts.append(f'Heavy metals [{hm.source_document}: "{hm.evidence}"]')
        if soc.evidence not in ("", "NONE"):
            cite_parts.append(f'SoC [{soc.source_document}: "{soc.evidence}"]')
        if cite_parts:
            return "; ".join(cite_parts)
    return "; ".join(parts) if parts else "N/A"


def aggregate_supplier(
    supplier_no: str,
    supplier: str,
    pdf_findings: List[PdfFinding],
    with_citations_in_concentration: bool = True,
) -> Dict[str, str]:
    hm_list = [p.heavy_metals for p in pdf_findings]
    soc_list = [p.soc for p in pdf_findings]

    hm = _pick_best(hm_list)
    soc = _pick_best(soc_list)

    concentration = _format_concentration(hm, soc, with_citations=with_citations_in_concentration)

    return {
        "Supplier No.": supplier_no,
        "Supplier": supplier,
        "PPWR compliant with heavy metals concentration limit": hm.answer,
        "PPWR SoC content": soc.answer,
        "Concentration": concentration,
        "Heavy metals evidence": hm.evidence,
        "Heavy metals source document": hm.source_document,
        "SoC evidence": soc.evidence,
        "SoC source document": soc.source_document,
    }


def run_audit(
    docs_dir: str = DOCS_DIR,
    output_path: str = DEFAULT_OUTPUT,
    api_key: Optional[str] = None,
    k: int = 5,
    supplier_filter: Optional[str] = None,
    include_evidence_columns: bool = False,
) -> List[Dict[str, str]]:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required (set in .env or pass --api-key).")

    suppliers = discover_suppliers(docs_dir)
    if supplier_filter:
        needle = supplier_filter.lower()
        suppliers = [
            s for s in suppliers
            if needle in s[0].lower() or needle in s[1].lower()
        ]

    rows: List[Dict[str, str]] = []
    for supplier_no, supplier_name, pdf_paths in suppliers:
        print(f"\n{'=' * 60}\nSupplier: {supplier_name} ({supplier_no})")
        if not pdf_paths:
            print("  No PDFs found — row will be N/A.")
            rows.append(
                {
                    "Supplier No.": supplier_no,
                    "Supplier": supplier_name,
                    "PPWR compliant with heavy metals concentration limit": "N/A",
                    "PPWR SoC content": "N/A",
                    "Concentration": "N/A",
                    "Heavy metals evidence": "NONE",
                    "Heavy metals source document": "",
                    "SoC evidence": "NONE",
                    "SoC source document": "",
                }
            )
            continue

        pdf_findings: List[PdfFinding] = []
        for pdf_path in pdf_paths:
            print(f"  Analyzing: {os.path.basename(pdf_path)}")
            try:
                pdf_findings.append(analyze_pdf(pdf_path, key, k=k))
            except Exception as exc:
                print(f"  ❌ Error on {pdf_path}: {exc}")

        row = aggregate_supplier(
            supplier_no,
            supplier_name,
            pdf_findings,
            with_citations_in_concentration=not include_evidence_columns,
        )
        rows.append(row)
        print(
            f"  → Heavy metals: {row['PPWR compliant with heavy metals concentration limit']} | "
            f"SoC: {row['PPWR SoC content']} | Concentration: {row['Concentration']}"
        )

    columns = CSV_COLUMNS if include_evidence_columns else CSV_COLUMNS_CORE
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ CSV written: {output_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="PPWR binary compliance audit on supplier PDFs.")
    parser.add_argument(
        "--docs-dir",
        default=DOCS_DIR,
        help=f"Root folder with supplier subfolders (default: {DOCS_DIR})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI API key (overrides .env)")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks per section (default 5)")
    parser.add_argument(
        "--supplier",
        default=None,
        help="Filter by supplier number or name substring",
    )
    parser.add_argument(
        "--with-evidence-columns",
        action="store_true",
        help="Add separate evidence and source-document columns for audit trail",
    )
    parser.add_argument(
        "--list-suppliers",
        action="store_true",
        help="List supplier folders and PDF counts, then exit",
    )
    args = parser.parse_args()

    if args.list_suppliers:
        for supplier_no, supplier_name, pdfs in discover_suppliers(args.docs_dir):
            print(f"{supplier_no} | {supplier_name} | {len(pdfs)} PDF(s)")
        return

    run_audit(
        docs_dir=args.docs_dir,
        output_path=args.output,
        api_key=args.api_key,
        k=args.k,
        supplier_filter=args.supplier,
        include_evidence_columns=args.with_evidence_columns,
    )


if __name__ == "__main__":
    main()
