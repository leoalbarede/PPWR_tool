"""
Streamlit dashboard: PPWR supplier compliance from ppwr_audit_results.csv

Run: streamlit run dashboard_ppwr.py

Regenerate data: python ppwr_audit.py
Optional evidence columns: python ppwr_audit.py --with-evidence-columns
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from evidence_validator import (
    evidence_matches_check,
    is_regulatory_boilerplate_only,
    soc_evidence_duplicates_heavy_metals,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "ppwr_audit_results.csv")
IS_STREAMLIT_CLOUD = bool(os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS"))

CHECK_ORDER = ["Heavy metals", "SoC"]

CHECK_DISPLAY: Dict[str, str] = {
    "Heavy metals": "Heavy metals compliance",
    "SoC": "SoC compliance",
}

COL_HEAVY_METALS = "PPWR compliant with heavy metals concentration limit"
COL_SOC = "PPWR SoC content"
COL_CONCENTRATION = "Concentration"

CHECK_TO_COLUMN: Dict[str, str] = {
    "Heavy metals": COL_HEAVY_METALS,
    "SoC": COL_SOC,
}

_EVIDENCE_IN_CONC_RE = re.compile(
    r'(?P<topic>Heavy metals|SoC)\s*(?:'
    r':\s*(?P<conc>.*?)\s*)?'
    r'\[(?P<doc>[^:]+):\s*"(?P<quote>[^"]*)"\]',
    re.IGNORECASE,
)


def evidence_ok(evidence: str) -> bool:
    return (evidence or "").strip() not in ("", "—", "NONE", "nan")


def enforce_evidence(raw_answer: str, evidence: str, check: str | None = None) -> str:
    """Yes/No without valid evidence for this check → N/A."""
    norm = normalize_answer(raw_answer)
    if norm not in ("yes", "no"):
        return norm
    if not evidence_ok(evidence):
        return "N/A"
    if is_regulatory_boilerplate_only(evidence):
        return "N/A"
    if check == "Heavy metals" and not evidence_matches_check(evidence, "heavy_metals"):
        return "N/A"
    if check == "SoC" and not evidence_matches_check(evidence, "soc"):
        return "N/A"
    return norm


def normalize_answer(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    s = str(value).strip().lower()
    if s in ("yes", "y", "oui"):
        return "yes"
    if s in ("no", "n", "non"):
        return "no"
    return "N/A"


def display_answer(value) -> str:
    norm = normalize_answer(value)
    if norm == "yes":
        return "Yes"
    if norm == "no":
        return "No"
    return "N/A"


def display_answer_for_check(check: str, value) -> str:
    """Map raw CSV answers to compliance display (SoC is inverted: yes in CSV = present)."""
    norm = normalize_answer(value)
    if check == "SoC":
        if norm == "no":
            return "Yes"  # absent / not detected → compliant
        if norm == "yes":
            return "No"  # present → non-compliant
        return "N/A"
    return display_answer(value)


def is_non_compliant(check: str, raw_answer: str) -> bool:
    """Non-compliant from raw CSV: Heavy metals No, SoC Yes (SoC present)."""
    norm = normalize_answer(raw_answer)
    if check == "Heavy metals":
        return norm == "no"
    if check == "SoC":
        return norm == "yes"
    return False


def format_concentration(concentration: str, raw_answer: str, check: str) -> str:
    if not is_non_compliant(check, raw_answer):
        return "—"
    conc = (concentration or "").strip()
    if not conc or conc in ("N/A", "—"):
        return "—"
    return conc


def matrix_cell_value(
    answer_display: str, raw_answer: str, check: str, concentration: str
) -> str:
    if is_non_compliant(check, raw_answer):
        conc = format_concentration(concentration, raw_answer, check)
        if conc != "—":
            return f"{answer_display}\n{conc}"
    return answer_display


def summary_count(displays: pd.Series) -> Tuple[int, int]:
    """Return (compliant Yes count, total)."""
    total = len(displays)
    return int((displays == "Yes").sum()), total


def _parse_concentration_field(text: str, check: str | None = None) -> Dict[str, Dict[str, str]]:
    """Extract evidence from Concentration column when dedicated columns are absent."""
    out: Dict[str, Dict[str, str]] = {}
    if not text or pd.isna(text):
        return out
    for m in _EVIDENCE_IN_CONC_RE.finditer(str(text)):
        topic = m.group("topic")
        key = "Heavy metals" if topic.lower().startswith("heavy") else "SoC"
        if check is not None and key != check:
            continue
        out[key] = {
            "concentration": (m.group("conc") or "").strip() or "N/A",
            "evidence": m.group("quote").strip(),
            "source_document": m.group("doc").strip(),
        }
    return out


def _row_evidence(row: pd.Series, check: str) -> Tuple[str, str, str]:
    """Return (concentration, evidence, source_document) for a check."""
    parsed = _parse_concentration_field(row.get(COL_CONCENTRATION, ""), check=check)

    if check == "Heavy metals":
        ev_col, doc_col = "Heavy metals evidence", "Heavy metals source document"
    else:
        ev_col, doc_col = "SoC evidence", "SoC source document"

    evidence = ""
    source = ""
    concentration = "N/A"

    if ev_col in row.index and str(row.get(ev_col, "")).strip() not in ("", "NONE", "nan"):
        evidence = str(row[ev_col]).strip()
        source = str(row.get(doc_col, "") or "").strip()
    elif check in parsed:
        evidence = parsed[check].get("evidence", "")
        source = parsed[check].get("source_document", "")
        concentration = parsed[check].get("concentration", "N/A")

    conc_raw = str(row.get(COL_CONCENTRATION, "") or "")
    if concentration == "N/A" and conc_raw and not pd.isna(conc_raw):
        prefix = "Heavy metals:" if check == "Heavy metals" else "SoC:"
        for part in conc_raw.split(";"):
            part = part.strip()
            if part.lower().startswith(prefix.lower()):
                concentration = re.sub(r"\s*\[.*\]\s*$", "", part[len(prefix) :]).strip() or "N/A"
                break

    return concentration, evidence or "—", source or "—"


@dataclass
class CheckResult:
    answer: str
    answer_display: str
    concentration: str
    concentration_display: str
    evidence: str
    source_document: str
    matrix_cell: str


def resolve_check(row: pd.Series, check: str) -> CheckResult:
    """Single source of truth for matrix, detail, and summary (always consistent)."""
    col = CHECK_TO_COLUMN[check]
    concentration, evidence, source = _row_evidence(row, check)
    answer = enforce_evidence(row.get(col), evidence, check=check)
    if (
        check == "SoC"
        and answer in ("yes", "no")
        and evidence_ok(evidence)
        and "Heavy metals evidence" in row.index
    ):
        hm_evidence = str(row.get("Heavy metals evidence", "") or "").strip()
        if soc_evidence_duplicates_heavy_metals(hm_evidence, evidence):
            answer = "N/A"
            evidence = ""
    answer_display = display_answer_for_check(check, answer)
    conc_display = format_concentration(concentration, answer, check)
    matrix_cell = matrix_cell_value(answer_display, answer, check, concentration)
    ev_out = evidence if evidence_ok(evidence) else "—"
    src_out = source if evidence_ok(evidence) else "—"
    return CheckResult(
        answer=answer,
        answer_display=answer_display,
        concentration=concentration,
        concentration_display=conc_display,
        evidence=ev_out,
        source_document=src_out,
        matrix_cell=matrix_cell,
    )


def build_detail_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in df.iterrows():
        supplier_no = str(row.get("Supplier No.", "")).strip()
        supplier = str(row.get("Supplier", "")).strip() or supplier_no
        for check in CHECK_ORDER:
            result = resolve_check(row, check)
            rows.append(
                {
                    "Supplier No.": supplier_no,
                    "Supplier": supplier,
                    "Check": CHECK_DISPLAY[check],
                    "Check_key": check,
                    "Answer": result.answer,
                    "Answer_display": result.answer_display,
                    "Matrix_cell": result.matrix_cell,
                    "Concentration": result.concentration,
                    "Concentration_display": result.concentration_display,
                    "Evidence": result.evidence,
                    "Source document": result.source_document,
                }
            )
    return pd.DataFrame(rows)


def _matrix_cell_style(check_key: str, display_value: str) -> str:
    v = str(display_value).strip().split("\n")[0]
    if v == "N/A":
        return "background-color: #e2e3e5; color: #383d41"
    if v == "Yes":
        return "background-color: #d4edda; color: #155724"
    return "background-color: #f8d7da; color: #721c24"


@st.cache_data
def load_results(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    st.set_page_config(
        page_title="PPWR — supplier compliance",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        "<style>[data-testid='stSidebar'],[data-testid='collapsedControl']{display:none;}</style>",
        unsafe_allow_html=True,
    )

    st.title("Supplier packaging declarations vs. PPWR requirements")

    if not os.path.isfile(RESULT_PATH):
        st.error("Results file not found: `ppwr_audit_results.csv`")
        if IS_STREAMLIT_CLOUD:
            st.info(
                "Commit **ppwr_audit_results.csv** to the repository after running "
                "`python ppwr_audit.py --with-evidence-columns` locally, then redeploy."
            )
        else:
            st.info(
                "Run **`python ppwr_audit.py --with-evidence-columns`** to generate "
                "**ppwr_audit_results.csv** from the PDFs in `docs/`."
            )
        return

    df = load_results(RESULT_PATH, os.path.getmtime(RESULT_PATH))
    if df.empty:
        st.warning("The CSV file has no rows.")
        return

    required = {"Supplier No.", "Supplier", COL_HEAVY_METALS, COL_SOC}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing columns in CSV: {', '.join(sorted(missing))}")
        return

    st.caption(f"Displayed: **{len(df)} supplier(s)** from `{os.path.basename(RESULT_PATH)}`.")

    st.info(
        "**PPWR checks** — Each supplier is assessed on two binary points from regulatory PDFs:\n\n"
        "**Heavy metals compliance**\n"
        "- `Yes` = sum of Pb, Cd, Hg, Cr6+ explicitly below 100 mg / ppm / mg/kg\n"
        "- `No` = non-compliant (above limit or explicitly not compliant)\n"
        "- `N/A` = not stated in documents\n\n"
        "**SoC compliance**\n"
        "- `Yes` = substances absent or not detected (compliant)\n"
        "- `No` = substances present (non-compliant)\n"
        "- `N/A` = not mentioned in documents"
    )

    detail = build_detail_rows(df)

    st.subheader("Summary — suppliers per check")
    st.caption("Heavy metals and SoC: count of compliant suppliers (Yes).")
    cols = st.columns(len(CHECK_ORDER))
    summary_labels = {
        "Heavy metals": "Compliant (Yes)",
        "SoC": "Compliant (Yes — absent or not detected)",
    }
    for col, check in zip(cols, CHECK_ORDER):
        sub = detail[detail["Check_key"] == check]
        n_ok, n_tot = summary_count(sub["Answer_display"])
        with col:
            st.metric(
                CHECK_DISPLAY[check],
                f"{n_ok} / {n_tot}",
                help=f"{summary_labels[check]} for **{CHECK_DISPLAY[check]}**",
            )

    st.subheader("Matrix — Supplier × check")
    matrix_rows: List[Dict] = []
    for (supplier_no, supplier), group in detail.groupby(["Supplier No.", "Supplier"], sort=False):
        row: Dict = {"Supplier No.": supplier_no, "Supplier": supplier}
        for check in CHECK_ORDER:
            rec = group[group["Check_key"] == check].iloc[0]
            row[CHECK_DISPLAY[check]] = rec["Matrix_cell"]
        matrix_rows.append(row)
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_display_cols = [CHECK_DISPLAY[c] for c in CHECK_ORDER]

    def _style_matrix(df_in: pd.DataFrame) -> pd.DataFrame:
        styled = df_in.style
        for check, col_name in zip(CHECK_ORDER, matrix_display_cols):
            styled = styled.apply(
                lambda col, c=check: [_matrix_cell_style(c, v) for v in col],
                subset=[col_name],
                axis=0,
            )
        return styled

    st.dataframe(_style_matrix(matrix_df), use_container_width=True, hide_index=True)

    st.subheader("Details by supplier")
    supplier_options = [
        f"{s} ({n})"
        for s, n in sorted(
            detail[["Supplier", "Supplier No."]].drop_duplicates().values.tolist()
        )
    ]
    pick_label = st.selectbox("Select supplier", options=supplier_options, index=0)
    pick_supplier = pick_label.rsplit(" (", 1)[0]
    pick_no = pick_label.rsplit(" (", 1)[1].rstrip(")")
    sub = detail[
        (detail["Supplier"] == pick_supplier) & (detail["Supplier No."] == pick_no)
    ].copy()

    show = sub[
        [
            "Check",
            "Answer_display",
            "Concentration_display",
            "Evidence",
            "Source document",
        ]
    ].rename(columns={"Answer_display": "Answer", "Concentration_display": "Concentration"})
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Concentration": st.column_config.TextColumn(
                "Concentration",
                help="Shown only when non-compliant (Heavy metals: No; SoC compliance: No).",
            ),
            "Evidence": st.column_config.TextColumn("Evidence", width="large"),
            "Source document": st.column_config.TextColumn("Source document", width="medium"),
        },
    )

    with st.expander("Show all supplier × check rows (long table)"):
        long_table = detail.sort_values(["Supplier", "Check"]).copy()
        long_table["Concentration"] = long_table["Concentration_display"]
        st.dataframe(
            long_table[
                [
                    "Supplier No.",
                    "Supplier",
                    "Check",
                    "Answer_display",
                    "Concentration",
                    "Evidence",
                    "Source document",
                ]
            ].rename(columns={"Answer_display": "Answer"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    if IS_STREAMLIT_CLOUD:
        st.caption(
            "Internal tool — PPWR supplier compliance dashboard. "
            "Data from `ppwr_audit_results.csv` in the repository."
        )
    else:
        st.caption(
            "Internal tool designed by Léo Albarede — PPWR supplier audit based on regulatory PDFs in `docs/`. "
            "Regenerate results with `python ppwr_audit.py --with-evidence-columns`."
        )


if __name__ == "__main__":
    main()
