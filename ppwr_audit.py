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

from analyzer_docling import ask_pdf_multi_section, get_document_markdown
from evidence_validator import (
    CHECK_HEAVY_METALS,
    CHECK_PFAS,
    CHECK_SOC,
    CHECK_SVHC,
    correct_inverted_raw_answer,
    dedupe_findings,
    recover_pfas_from_markdown,
    validate_point_finding,
)

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
- YES or NO is forbidden without a verbatim Evidence quote — use N/A instead.
- The Evidence quote MUST appear word-for-word in the CONTEXT. If you cannot copy an exact quote, use N/A.
- Answer N/A if the only relevant text describes legal/regulatory requirements (e.g. PPWR Article 5 limits)
  rather than the supplier's own declaration about their packaging materials or products.
- Each numbered point requires its OWN evidence quote. Never reuse evidence from another point or topic.
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
   Answer N/A if the only quote describes PPWR/legal requirements, not the supplier's own material.
   If YES or NO, report the total or individual concentrations when explicitly given.
   Evidence must mention Pb, Cd, Hg, Cr6+ or heavy metals — not SoC/SVHC alone.
""".strip(),
    ),
    (
        "PPWR — Substances of Concern (SoC)",
        """
2. Substances of Concern (SoC) in packaging per Article 3(2)(a) of Regulation (EU) 2025/40 (PPWR).
   NOT PFAS (separate check) and NOT standalone SVHC Article 9 declarations unless about SoC criteria below.
   A substance is SoC if it meets ANY of: (i) listed in Annex XIV REACH; (ii) classified STOT with PBT/vPvB;
   (iii) listed in Annex XVII REACH; (iv) classified CMR; (v) adverse effect on environment or human health
   upon release from packaging; (vi) impedes recyclability or re-use; (vii) contaminates recycling streams
   or final recycled products.
   Answer YES if one or more SoC are present or declared in the supplier's packaging materials.
   Answer NO if the supplier explicitly states no Substances of Concern are present in their packaging
   (none of criteria (i)–(vii) apply to substances in their materials).
   Answer N/A if SoC / Article 3(2)(a) is not mentioned at all.
   Answer N/A if the only quote describes legal requirements only, not the supplier's own material.
   If YES or NO with quantitative values, report concentrations exactly as written.
   Evidence must mention Substances of Concern, Article 3, REACH Annex XIV/XVII, CMR, STOT, PBT, vPvB,
   recyclability, or recycling streams — NOT heavy metals or PFAS alone.
""".strip(),
    ),
    (
        "PPWR — PFAS",
        """
3. PFAS (per- and polyfluoroalkyl substances) in packaging materials: Is PFAS presence or concentration
   explicitly mentioned? EU PPWR limits: <25 µg/kg per individual PFAS, <250 µg/kg sum of targeted PFAS,
   <50 mg/kg total PFAS including polymeric PFAS (total fluorine).
   SEMANTICS (inverted — read carefully):
   - Answer YES only if PFAS are present, detected, or declared ABOVE regulatory limits (non-compliant).
   - Answer NO if the supplier explicitly states PFAS are absent, not detected, below limits, or that
     PFAS limits/requirements are MET (e.g. "limit for PFAS … is met", "requirements for PFAS … are met",
     "PFAS are not used", TF/TOF analysis confirms compliance).
   - Answer N/A if PFAS are not mentioned at all.
   Answer N/A if the only quote describes PPWR/legal requirements only, not the supplier's own material.
   If YES or NO with quantitative values, report concentrations exactly as written.
   Evidence must mention PFAS, perfluoro, polyfluoro, total fluorine, or Article 5.5 — NOT heavy metals or generic SoC alone.
""".strip(),
    ),
    (
        "PPWR — SVHC (Article 9)",
        """
4. SVHC (Substances of Very High Concern) under PPWR Article 9: Is SVHC presence in packaging materials
   explicitly mentioned? Concentration limit: 0.1% w/w.
   Answer YES if SVHC are present or declared above 0.1% w/w, NO if explicitly absent, not detected,
   or declared below 0.1% w/w, N/A if SVHC are not mentioned at all.
   Answer N/A if the only quote describes PPWR/legal requirements, not the supplier's own material.
   If YES or NO with quantitative values, report concentrations exactly as written.
   Evidence must mention SVHC, substances of very high concern, or Article 9 — NOT PFAS or heavy metals alone.
""".strip(),
    ),
]

CSV_COLUMNS = [
    "Supplier No.",
    "Doc list",
    "Supplier folder",
    "Supplier",
    "PPWR compliant with heavy metals concentration limit",
    "PPWR SoC content",
    "PPWR PFAS content",
    "PPWR SVHC content",
    "Concentration",
    "Heavy metals evidence",
    "Heavy metals source document",
    "SoC evidence",
    "SoC source document",
    "PFAS evidence",
    "PFAS source document",
    "SVHC evidence",
    "SVHC source document",
]

# User-facing export (core columns only)
CSV_COLUMNS_CORE = [
    "Supplier No.",
    "Doc list",
    "Supplier folder",
    "Supplier",
    "PPWR compliant with heavy metals concentration limit",
    "PPWR SoC content",
    "PPWR PFAS content",
    "PPWR SVHC content",
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
    pfas: PointFinding = field(default_factory=PointFinding)
    svhc: PointFinding = field(default_factory=PointFinding)


@dataclass
class SupplierFinding:
    doc_list: str
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


def _collect_pdfs(folder_path: str) -> List[str]:
    return sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    )


def discover_suppliers(
    docs_dir: str = DOCS_DIR,
) -> List[Tuple[str, str, str, str, List[str]]]:
    """
    Return (doc_list, supplier_folder, supplier_no, supplier_name, pdf_paths).

    Supports nested layout docs/<Doc list>/<Supplier folder>/PDFs
    and legacy flat layout docs/<Supplier folder>/PDFs.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    suppliers: List[Tuple[str, str, str, str, List[str]]] = []
    for entry in sorted(os.listdir(docs_dir)):
        folder_path = os.path.join(docs_dir, entry)
        if not os.path.isdir(folder_path):
            continue

        parsed = parse_supplier_folder(entry)
        if parsed:
            supplier_no, supplier_name = parsed
            pdfs = _collect_pdfs(folder_path)
            suppliers.append(("", entry.strip(), supplier_no, supplier_name, pdfs))
            continue

        doc_list = entry.strip()
        for sub_entry in sorted(os.listdir(folder_path)):
            sub_path = os.path.join(folder_path, sub_entry)
            if not os.path.isdir(sub_path):
                continue
            sub_parsed = parse_supplier_folder(sub_entry)
            if not sub_parsed:
                print(f"⚠️  Skipping folder (unexpected name): {doc_list}/{sub_entry}")
                continue
            supplier_no, supplier_name = sub_parsed
            pdfs = _collect_pdfs(sub_path)
            suppliers.append((doc_list, sub_entry.strip(), supplier_no, supplier_name, pdfs))

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


def _extract_point_block(section_body: str, point_num: int) -> str:
    """Keep only the block for Point N when the LLM returns multiple points."""
    if not section_body:
        return ""
    markers = list(
        re.finditer(
            rf"(?:^|\n)(?:Point\s+(\d+)\s*:|\s*(\d+)\.\s+)",
            section_body,
            re.IGNORECASE,
        )
    )
    for i, match in enumerate(markers):
        num = int(match.group(1) or match.group(2))
        if num != point_num:
            continue
        start = match.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(section_body)
        return section_body[start:end].strip()
    return section_body.strip()


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

    if finding.answer in ("yes", "no") and finding.evidence in ("", "NONE"):
        finding.answer = "N/A"
        finding.concentration = "N/A"

    return finding


def parse_rag_answer(answer_text: str, source_file: str) -> PdfFinding:
    """Parse LLM section output into structured findings for all PPWR checks."""
    result = PdfFinding(source_file=source_file)

    section_map = [
        ("heavy_metals", ("Heavy metals", "Pb, Cd, Hg"), 1),
        ("soc", ("SoC", "Substances of Concern"), 2),
        ("pfas", ("PFAS",), 3),
        ("svhc", ("SVHC", "Article 9"), 4),
    ]
    for attr, keywords, point_num in section_map:
        body = _extract_section_body(answer_text, keywords)
        if body:
            finding = _parse_point_block(_extract_point_block(body, point_num))
            finding.source_document = source_file
            setattr(result, attr, finding)

    return result


def _validate_pdf_finding(finding: PdfFinding, source_markdown: str) -> PdfFinding:
    finding.heavy_metals = validate_point_finding(
        finding.heavy_metals, source_markdown, CHECK_HEAVY_METALS
    )
    finding.soc = validate_point_finding(finding.soc, source_markdown, CHECK_SOC)
    finding.pfas = validate_point_finding(finding.pfas, source_markdown, CHECK_PFAS)
    if finding.pfas.answer == "N/A":
        recovered = recover_pfas_from_markdown(source_markdown, finding.source_file)
        if recovered is not None:
            finding.pfas = validate_point_finding(recovered, source_markdown, CHECK_PFAS)
    finding.svhc = validate_point_finding(finding.svhc, source_markdown, CHECK_SVHC)
    dedupe_findings(
        {
            CHECK_HEAVY_METALS: finding.heavy_metals,
            CHECK_SOC: finding.soc,
            CHECK_PFAS: finding.pfas,
            CHECK_SVHC: finding.svhc,
        }
    )
    return finding


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
        retry_na_sections=True,
    )
    finding = parse_rag_answer(rag.get("answer", ""), source_file)
    source_markdown = get_document_markdown(pdf_path)
    return _validate_pdf_finding(finding, source_markdown)


def _has_evidence(finding: PointFinding) -> bool:
    return finding.evidence not in ("", "NONE")


def _pick_best(findings: List[PointFinding]) -> PointFinding:
    """Prefer yes/no answers backed by verbatim evidence; otherwise N/A."""
    backed = [f for f in findings if f.answer in ("yes", "no") and _has_evidence(f)]
    if backed:
        return backed[0]
    return PointFinding()


def _format_concentration(findings: Dict[str, PointFinding], with_citations: bool) -> str:
    labels = {
        "heavy_metals": "Heavy metals",
        "soc": "SoC",
        "pfas": "PFAS",
        "svhc": "SVHC",
    }
    parts: List[str] = []
    for key, label in labels.items():
        f = findings[key]
        if f.concentration not in ("", "N/A"):
            part = f"{label}: {f.concentration}"
            if with_citations and f.evidence not in ("", "NONE"):
                part += f' [{f.source_document}: "{f.evidence}"]'
            parts.append(part)
    if not parts and with_citations:
        cite_parts: List[str] = []
        for key, label in labels.items():
            f = findings[key]
            if f.evidence not in ("", "NONE"):
                cite_parts.append(f'{label} [{f.source_document}: "{f.evidence}"]')
        if cite_parts:
            return "; ".join(cite_parts)
    return "; ".join(parts) if parts else "N/A"


def aggregate_supplier(
    doc_list: str,
    supplier_folder: str,
    supplier_no: str,
    supplier: str,
    pdf_findings: List[PdfFinding],
    with_citations_in_concentration: bool = True,
) -> Dict[str, str]:
    hm_list = [p.heavy_metals for p in pdf_findings]
    soc_list = [p.soc for p in pdf_findings]
    pfas_list = [p.pfas for p in pdf_findings]
    svhc_list = [p.svhc for p in pdf_findings]

    hm = _pick_best(hm_list)
    soc = _pick_best(soc_list)
    pfas = _pick_best(pfas_list)
    svhc = _pick_best(svhc_list)

    concentration = _format_concentration(
        {"heavy_metals": hm, "soc": soc, "pfas": pfas, "svhc": svhc},
        with_citations=with_citations_in_concentration,
    )

    pfas_answer = correct_inverted_raw_answer(CHECK_PFAS, pfas.answer, pfas.evidence)

    return {
        "Supplier No.": supplier_no,
        "Doc list": doc_list,
        "Supplier folder": supplier_folder,
        "Supplier": supplier,
        "PPWR compliant with heavy metals concentration limit": hm.answer,
        "PPWR SoC content": soc.answer,
        "PPWR PFAS content": pfas_answer,
        "PPWR SVHC content": svhc.answer,
        "Concentration": concentration,
        "Heavy metals evidence": hm.evidence,
        "Heavy metals source document": hm.source_document,
        "SoC evidence": soc.evidence,
        "SoC source document": soc.source_document,
        "PFAS evidence": pfas.evidence,
        "PFAS source document": pfas.source_document,
        "SVHC evidence": svhc.evidence,
        "SVHC source document": svhc.source_document,
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
            if needle in s[0].lower()
            or needle in s[1].lower()
            or needle in s[2].lower()
            or needle in s[3].lower()
        ]

    rows: List[Dict[str, str]] = []
    for doc_list, supplier_folder, supplier_no, supplier_name, pdf_paths in suppliers:
        list_label = f" [{doc_list}]" if doc_list else ""
        print(f"\n{'=' * 60}\nSupplier: {supplier_name} ({supplier_no}){list_label}")
        if not pdf_paths:
            print("  No PDFs found — row will be N/A.")
            rows.append(
                {
                    "Supplier No.": supplier_no,
                    "Doc list": doc_list,
                    "Supplier folder": supplier_folder,
                    "Supplier": supplier_name,
                    "PPWR compliant with heavy metals concentration limit": "N/A",
                    "PPWR SoC content": "N/A",
                    "PPWR PFAS content": "N/A",
                    "PPWR SVHC content": "N/A",
                    "Concentration": "N/A",
                    "Heavy metals evidence": "NONE",
                    "Heavy metals source document": "",
                    "SoC evidence": "NONE",
                    "SoC source document": "",
                    "PFAS evidence": "NONE",
                    "PFAS source document": "",
                    "SVHC evidence": "NONE",
                    "SVHC source document": "",
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
            doc_list,
            supplier_folder,
            supplier_no,
            supplier_name,
            pdf_findings,
            with_citations_in_concentration=not include_evidence_columns,
        )
        rows.append(row)
        print(
            f"  → Heavy metals: {row['PPWR compliant with heavy metals concentration limit']} | "
            f"SoC: {row['PPWR SoC content']} | PFAS: {row['PPWR PFAS content']} | "
            f"SVHC: {row['PPWR SVHC content']}"
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
        for doc_list, supplier_folder, supplier_no, supplier_name, pdfs in discover_suppliers(args.docs_dir):
            label = f"{doc_list} | " if doc_list else ""
            print(f"{label}{supplier_folder} | {supplier_no} | {supplier_name} | {len(pdfs)} PDF(s)")
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
