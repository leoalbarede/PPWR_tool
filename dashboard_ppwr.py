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
    correct_inverted_raw_answer,
    evidence_duplicates_prior_check,
    evidence_matches_check,
    is_regulatory_boilerplate_only,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "data", "ppwr_audit_results.csv")
IS_STREAMLIT_CLOUD = bool(os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS"))

CHECK_ORDER = ["Heavy metals", "SoC", "PFAS", "SVHC"]
INVERTED_CHECKS = {"SoC", "PFAS", "SVHC"}

CHECK_DISPLAY: Dict[str, str] = {
    "Heavy metals": "Heavy metals compliance",
    "SoC": "SoC compliance",
    "PFAS": "PFAS compliance",
    "SVHC": "SVHC conformity",
}

COL_HEAVY_METALS = "PPWR compliant with heavy metals concentration limit"
COL_SOC = "PPWR SoC content"
COL_PFAS = "PPWR PFAS content"
COL_SVHC = "PPWR SVHC content"
COL_CONCENTRATION = "Concentration"
COL_DOC_LIST = "Doc list"
COL_SUPPLIER_FOLDER = "Supplier folder"

CHECK_TO_COLUMN: Dict[str, str] = {
    "Heavy metals": COL_HEAVY_METALS,
    "SoC": COL_SOC,
    "PFAS": COL_PFAS,
    "SVHC": COL_SVHC,
}

CHECK_TO_VALIDATOR: Dict[str, str] = {
    "Heavy metals": "heavy_metals",
    "SoC": "soc",
    "PFAS": "pfas",
    "SVHC": "svhc",
}

EVIDENCE_COLUMNS: Dict[str, Tuple[str, str]] = {
    "Heavy metals": ("Heavy metals evidence", "Heavy metals source document"),
    "SoC": ("SoC evidence", "SoC source document"),
    "PFAS": ("PFAS evidence", "PFAS source document"),
    "SVHC": ("SVHC evidence", "SVHC source document"),
}


def _row_doc_list(row: pd.Series) -> str:
    if COL_DOC_LIST not in row.index:
        return ""
    val = row.get(COL_DOC_LIST, "")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _matrix_group_cols(df: pd.DataFrame) -> List[str]:
    """One matrix row per CSV row — include folder when present to disambiguate duplicates."""
    cols = ["Supplier No."]
    if COL_DOC_LIST in df.columns:
        cols.append(COL_DOC_LIST)
    if COL_SUPPLIER_FOLDER in df.columns:
        cols.append(COL_SUPPLIER_FOLDER)
    cols.append("Supplier")
    return cols


_EVIDENCE_IN_CONC_RE = re.compile(
    r'(?P<topic>Heavy metals|SoC|PFAS|SVHC)\s*(?:'
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
    if check and check in CHECK_TO_VALIDATOR:
        if not evidence_matches_check(evidence, CHECK_TO_VALIDATOR[check]):
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
    """Map raw CSV answers to compliance display (inverted checks: yes = present/non-compliant)."""
    norm = normalize_answer(value)
    if check in INVERTED_CHECKS:
        if norm == "no":
            return "Yes"
        if norm == "yes":
            return "No"
        return "N/A"
    return display_answer(value)


def is_non_compliant(check: str, raw_answer: str) -> bool:
    """Non-compliant from raw CSV."""
    norm = normalize_answer(raw_answer)
    if check in INVERTED_CHECKS:
        return norm == "yes"
    if check == "Heavy metals":
        return norm == "no"
    return False


def format_concentration(concentration: str, raw_answer: str, check: str) -> str:
    if not is_non_compliant(check, raw_answer):
        return "—"
    conc = (concentration or "").strip()
    if not conc or conc in ("N/A", "—"):
        return "—"
    return conc


def matrix_cell_value(answer_display: str) -> str:
    """Matrix shows Yes / No / N/A only (no concentration or extra text)."""
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
        key_map = {
            "heavy metals": "Heavy metals",
            "soc": "SoC",
            "pfas": "PFAS",
            "svhc": "SVHC",
        }
        key = key_map.get(topic.lower(), topic)
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

    ev_col, doc_col = EVIDENCE_COLUMNS[check]

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
        prefix = f"{check}:"
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
    if answer in ("yes", "no") and evidence_ok(evidence):
        answer = correct_inverted_raw_answer(CHECK_TO_VALIDATOR[check], answer, evidence)
        for prior_check in CHECK_ORDER:
            if prior_check == check:
                break
            prior_ev_col = EVIDENCE_COLUMNS[prior_check][0]
            if prior_ev_col not in row.index:
                continue
            prior_evidence = str(row.get(prior_ev_col, "") or "").strip()
            if evidence_duplicates_prior_check(
                prior_evidence,
                evidence,
                CHECK_TO_VALIDATOR[prior_check],
                CHECK_TO_VALIDATOR[check],
            ):
                answer = "N/A"
                evidence = ""
                break
    answer_display = display_answer_for_check(check, answer)
    conc_display = format_concentration(concentration, answer, check)
    matrix_cell = matrix_cell_value(answer_display)
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
    row_cols = _matrix_group_cols(df)
    for _, row in df.iterrows():
        base = {col: str(row.get(col, "")).strip() for col in row_cols}
        for check in CHECK_ORDER:
            result = resolve_check(row, check)
            rows.append(
                {
                    **base,
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


def _matrix_cell_style(display_value: str) -> str:
    v = str(display_value).strip()
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

    required = {"Supplier No.", "Supplier", COL_HEAVY_METALS, COL_SOC, COL_PFAS, COL_SVHC}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing columns in CSV: {', '.join(sorted(missing))}")
        return

    st.caption(f"Displayed: **{len(df)} supplier(s)** from `{os.path.basename(RESULT_PATH)}`.")

    st.info(
        "**PPWR checks** — Each supplier is assessed on four binary points from regulatory PDFs:\n\n"
        "**Heavy metals compliance**\n"
        "- `Yes` = sum of Pb, Cd, Hg, Cr6+ explicitly below 100 mg / ppm / mg/kg\n"
        "- `No` = non-compliant (above limit or explicitly not compliant)\n"
        "- `N/A` = not stated in documents\n\n"
        "**SoC compliance** (Article 3(2)(a), Regulation EU 2025/40)\n"
        "- `Yes` = no Substances of Concern in packaging (none of criteria i–vii)\n"
        "- `No` = one or more SoC detected\n"
        "- `N/A` = not mentioned in documents\n\n"
        "**PFAS compliance**\n"
        "- `Yes` = PFAS compliant: not detected, below limits, or supplier states the limit/requirement **is met**\n"
        "- `No` = PFAS detected above regulatory limits\n"
        "- `N/A` = not mentioned in documents\n\n"
        "**SVHC conformity**\n"
        "- `Yes` = no SVHC above 0.1% w/w (PPWR Art. 9)\n"
        "- `No` = SVHC detected above 0.1% w/w threshold\n"
        "- `N/A` = not mentioned in documents"
    )

    detail = build_detail_rows(df)

    st.subheader("Summary — suppliers per check")
    st.caption("Count of compliant suppliers (Yes) per check.")
    cols = st.columns(len(CHECK_ORDER))
    summary_labels = {
        "Heavy metals": "Compliant (Yes)",
        "SoC": "Compliant (Yes — no SoC per Art. 3(2)(a))",
        "PFAS": "Compliant (Yes — not detected / below limits)",
        "SVHC": "Conform (Yes — below 0.1% w/w)",
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
    group_cols = _matrix_group_cols(df)
    for _, row in df.iterrows():
        out = {col: str(row.get(col, "")).strip() for col in group_cols}
        for check in CHECK_ORDER:
            out[CHECK_DISPLAY[check]] = resolve_check(row, check).matrix_cell
        matrix_rows.append(out)
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_display_cols = group_cols + [CHECK_DISPLAY[c] for c in CHECK_ORDER]
    check_cols = [CHECK_DISPLAY[c] for c in CHECK_ORDER]

    def _style_matrix(df_in: pd.DataFrame) -> pd.DataFrame:
        styled = df_in.style
        for col_name in check_cols:
            styled = styled.apply(
                lambda col: [_matrix_cell_style(v) for v in col],
                subset=[col_name],
                axis=0,
            )
        return styled

    st.dataframe(_style_matrix(matrix_df), use_container_width=True, hide_index=True)

    st.subheader("Details by supplier")
    picker_cols = list(_matrix_group_cols(df))
    picker_df = df[picker_cols].drop_duplicates().sort_values(picker_cols)
    supplier_options: List[str] = []
    for vals in picker_df.values.tolist():
        parts = [str(v).strip() for v in vals if str(v).strip()]
        supplier_options.append(" — ".join(parts))
    pick_label = st.selectbox("Select supplier", options=supplier_options, index=0)
    pick_idx = supplier_options.index(pick_label)
    picked = picker_df.iloc[pick_idx]
    sub = detail.copy()
    for col in picker_cols:
        sub = sub[sub[col].astype(str) == str(picked[col])]

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
                help="Shown only when non-compliant (Heavy metals: No; SoC/PFAS/SVHC: No).",
            ),
            "Evidence": st.column_config.TextColumn("Evidence", width="large"),
            "Source document": st.column_config.TextColumn("Source document", width="medium"),
        },
    )

    with st.expander("Show all supplier × check rows (long table)"):
        long_table = detail.sort_values(group_cols + ["Check"]).copy()
        long_table["Concentration"] = long_table["Concentration_display"]
        long_cols = group_cols + [
            "Check",
            "Answer_display",
            "Concentration",
            "Evidence",
            "Source document",
        ]
        st.dataframe(
            long_table[long_cols].rename(columns={"Answer_display": "Answer"}),
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
