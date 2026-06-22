"""
Streamlit dashboard: PPWR supplier compliance from ppwr_audit_results.csv

Run: streamlit run dashboard_ppwr.py

Regenerate data: python ppwr_audit.py
Optional evidence columns: python ppwr_audit.py --with-evidence-columns
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "ppwr_audit_results.csv")

CHECK_ORDER = ["Heavy metals", "SoC"]

COL_HEAVY_METALS = "PPWR compliant with heavy metals concentration limit"
COL_SOC = "PPWR SoC content"
COL_CONCENTRATION = "Concentration"

CHECK_TO_COLUMN: Dict[str, str] = {
    "Heavy metals": COL_HEAVY_METALS,
    "SoC": COL_SOC,
}

_EVIDENCE_IN_CONC_RE = re.compile(
    r'(?P<topic>Heavy metals|SoC)\s*(?::\s*(?P<conc>[^[]*))?\s*\[(?P<doc>[^:]+):\s*"(?P<quote>[^"]*)"\]',
    re.IGNORECASE,
)


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


def summary_count(check: str, answers: pd.Series) -> Tuple[int, int]:
    """Return (positive_count, total) for summary metrics."""
    norm = answers.map(normalize_answer)
    total = len(norm)
    if check == "Heavy metals":
        return int((norm == "yes").sum()), total
    # SoC: count suppliers with an explicit declaration (yes or no)
    return int(norm.isin(["yes", "no"]).sum()), total


def _parse_concentration_field(text: str) -> Dict[str, Dict[str, str]]:
    """Extract evidence from Concentration column when dedicated columns are absent."""
    out: Dict[str, Dict[str, str]] = {}
    if not text or pd.isna(text):
        return out
    for m in _EVIDENCE_IN_CONC_RE.finditer(str(text)):
        topic = m.group("topic")
        key = "Heavy metals" if topic.lower().startswith("heavy") else "SoC"
        out[key] = {
            "concentration": (m.group("conc") or "").strip() or "N/A",
            "evidence": m.group("quote").strip(),
            "source_document": m.group("doc").strip(),
        }
    return out


def _row_evidence(row: pd.Series, check: str) -> Tuple[str, str, str]:
    """Return (concentration, evidence, source_document) for a check."""
    parsed = _parse_concentration_field(row.get(COL_CONCENTRATION, ""))

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
                # Strip trailing [doc: "quote"] if present
                concentration = re.sub(r"\s*\[.*\]\s*$", "", part[len(prefix) :]).strip() or "N/A"
                break

    return concentration, evidence or "—", source or "—"


def build_detail_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in df.iterrows():
        supplier_no = str(row.get("Supplier No.", "")).strip()
        supplier = str(row.get("Supplier", "")).strip() or supplier_no
        for check in CHECK_ORDER:
            col = CHECK_TO_COLUMN[check]
            answer = normalize_answer(row.get(col))
            concentration, evidence, source = _row_evidence(row, check)
            rows.append(
                {
                    "Supplier No.": supplier_no,
                    "Supplier": supplier,
                    "Check": check,
                    "Answer": answer,
                    "Answer_display": display_answer(row.get(col)),
                    "Concentration": concentration,
                    "Evidence": evidence,
                    "Source document": source,
                }
            )
    return pd.DataFrame(rows)


def _matrix_cell_style(check: str, display_value: str) -> str:
    v = str(display_value).strip()
    if v == "N/A":
        return "background-color: #e2e3e5; color: #383d41"
    if check == "Heavy metals":
        if v == "Yes":
            return "background-color: #d4edda; color: #155724"
        return "background-color: #f8d7da; color: #721c24"
    # SoC: Yes = declared present, No = absent / not detected
    if v == "Yes":
        return "background-color: #fff3cd; color: #856404"
    if v == "No":
        return "background-color: #d4edda; color: #155724"
    return ""


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
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

    st.title("Supplier packaging declarations vs. PPWR requirements (regulatory PDFs)")

    if not os.path.isfile(RESULT_PATH):
        st.error(f"Results file not found:\n`{RESULT_PATH}`")
        st.info("Run **`python ppwr_audit.py`** to generate **ppwr_audit_results.csv** from the PDFs in `docs/`.")
        return

    df = load_results(RESULT_PATH)
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
        "- **Heavy metals** — `Yes` = sum of Pb, Cd, Hg, Cr6+ explicitly below 100 mg / ppm / mg/kg; "
        "`No` = non-compliant; `N/A` = not stated in documents.\n\n"
        "- **SoC** — `Yes` = Substances of Concern explicitly present; "
        "`No` = explicitly absent or not detected; `N/A` = not mentioned."
    )

    detail = build_detail_rows(df)

    st.subheader("Summary — suppliers per check")
    st.caption("Heavy metals: compliant suppliers. SoC: suppliers with an explicit yes/no declaration.")
    cols = st.columns(len(CHECK_ORDER))
    summary_labels = {
        "Heavy metals": "Compliant (Yes)",
        "SoC": "Explicit declaration (Yes or No)",
    }
    for col, check in zip(cols, CHECK_ORDER):
        sub = detail[detail["Check"] == check]
        n_ok, n_tot = summary_count(check, sub["Answer"])
        with col:
            st.metric(
                check,
                f"{n_ok} / {n_tot}",
                help=f"{summary_labels[check]} for **{check}**",
            )

    st.subheader("Matrix — Supplier × check")
    pivot = detail.pivot_table(
        index=["Supplier No.", "Supplier"],
        columns="Check",
        values="Answer_display",
        aggfunc="first",
    )
    for check in CHECK_ORDER:
        if check not in pivot.columns:
            pivot[check] = "N/A"
    pivot = pivot[CHECK_ORDER].fillna("N/A")

    matrix_df = pivot.reset_index()

    def _style_matrix(df_in: pd.DataFrame) -> pd.DataFrame:
        styled = df_in.style
        for check in CHECK_ORDER:
            styled = styled.apply(
                lambda col, c=check: [_matrix_cell_style(c, v) for v in col],
                subset=[check],
                axis=0,
            )
        return styled

    st.dataframe(_style_matrix(matrix_df), use_container_width=True, hide_index=True)

    st.subheader("Details by supplier")
    suppliers_sorted = sorted(detail["Supplier"].unique())
    pick_supplier = st.selectbox("Select supplier", options=suppliers_sorted, index=0)
    sub = detail[detail["Supplier"] == pick_supplier].copy()

    show = sub[
        [
            "Check",
            "Answer_display",
            "Concentration",
            "Evidence",
            "Source document",
        ]
    ].rename(columns={"Answer_display": "Answer"})
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Evidence": st.column_config.TextColumn("Evidence", width="large"),
            "Source document": st.column_config.TextColumn("Source document", width="medium"),
        },
    )

    with st.expander("Show all supplier × check rows (long table)"):
        st.dataframe(
            detail.sort_values(["Supplier", "Check"]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.caption(
        "Internal tool — PPWR supplier audit based on regulatory PDFs in `docs/`. "
        "Regenerate results with `python ppwr_audit.py`."
    )


if __name__ == "__main__":
    main()
