# PPWR tool

Document-audit engine for PPWR supplier compliance: PDF → RAG → binary compliance CSV,
with a Streamlit dashboard and helper scripts to export results.

## Project structure

```
PPWR_tool/
├── streamlit_app.py            # Streamlit Cloud entry point
├── dashboard_ppwr.py           # Dashboard UI + compliance resolution (source of truth)
├── evidence_validator.py       # Evidence checks, answer correction, PFAS/SoC/SVHC logic
├── analyzer_docling.py         # PDF → Markdown (cache) + adaptive RAG (small/medium/large)
├── ppwr_audit.py               # Batch audit: supplier folders → results CSV
├── tiktoken_cache_setup.py     # Persistent tiktoken cache setup
├── flashrank_cache_setup.py    # FlashRank cache (large documents only)
│
├── scripts/                    # One-off / helper tools
│   ├── fill_results_for_heiko.py   # Fill compliance columns in a Heiko workbook from a CSV
│   ├── patch_ppwr_csv.py           # Post-hoc corrections to an audit CSV (no LLM re-run)
│   └── file_manager.py
│
├── data/                       # Generated data (versioned)
│   ├── ppwr_audit_results.csv          # 1st wave audit (read by the dashboard)
│   ├── ppwr_audit_results_2nd_wave.csv # 2nd wave audit
│   └── results/                        # Filled Heiko workbooks
│       ├── 1st_wave_results_for_heiko.xlsx
│       └── 1st_2nd_wave_results_for_heiko.xlsx
│
├── docs/                       # Source supplier PDFs — LOCAL ONLY (git-ignored)
│   ├── 1st_wave/
│   └── 2st_wave/
│
├── requirements.txt            # Dashboard only (Streamlit Cloud)
├── requirements-audit.txt      # Full audit pipeline (PDF RAG)
└── .python-version
```

Runtime caches (`markdown_cache/`, `vector_cache/`, `tiktoken_cache/`, `flashrank_cache/`) are
created automatically and are git-ignored. Source PDFs under `docs/` are kept locally and are
**not** versioned; only the generated CSVs and result workbooks under `data/` are committed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

# Full audit pipeline (PDF RAG + export)
pip install -r requirements-audit.txt
pip install streamlit pandas openpyxl

# Dashboard only (no PDF audit)
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
# optional
RAG_EMBEDDING_MODEL=text-embedding-3-small
TIKTOKEN_CACHE_DIR=./tiktoken_cache
FLASHRANK_CACHE_DIR=./flashrank_cache
```

## PPWR supplier audit

Binary compliance check on regulatory PDFs per supplier folder under `docs/`.

Supplier folders are recognized in two layouts:

- `{Supplier No} - {Supplier name}` (e.g. `S00122_0100114617 - AMCOR FLEXIBLES SELESTAT`)
- `{numeric id} {Supplier name}` (e.g. `100001453 JOSE GRAELLS E HIJOS S A`)

Folders may be grouped under list folders (e.g. `docs/1st_wave/EU (PPWR list Metal Pack)/...`).
PDFs are collected recursively, so nested subfolders are supported.

```bash
# List suppliers and PDF counts for a docs subtree
python ppwr_audit.py --list-suppliers --docs-dir docs/2st_wave

# Run full audit -> data/ppwr_audit_results.csv
python ppwr_audit.py --docs-dir docs/1st_wave

# Write to a custom CSV, with separate evidence columns for review
python ppwr_audit.py --docs-dir docs/2st_wave \
  --output data/ppwr_audit_results_2nd_wave.csv --with-evidence-columns

# One supplier only (faster test)
python ppwr_audit.py --docs-dir docs/2st_wave --supplier "JOSE"
```

### Adaptive retrieval

| Profile | Chunks | Strategy |
|---------|--------|----------|
| `small` | ≤ 6 | BM25 only, no vector index, no rerank |
| `medium` | 7–20 | BM25 + dense MMR, no rerank |
| `large` | > 20 | Full hybrid + multi-query + FlashRank |

Profile is selected automatically; override with `retrieval_profile=` in `ask_pdf_multi_section`.

## Fill a Heiko workbook from an audit CSV

`scripts/fill_results_for_heiko.py` writes the compliance columns (heavy metals, PFAS, SoC, SVHC)
into the master workbook, matched by Supplier ID. Values: `1` = compliant, `0` = non-compliant,
`no response` = N/A. It edits only the matched cells (surgical XLSX edit) and makes a `.bak` first.

```bash
# Defaults: data/ppwr_audit_results.csv -> data/results/1st_2nd_wave_results_for_heiko.xlsx
python scripts/fill_results_for_heiko.py

# Explicit CSV + target workbook
python scripts/fill_results_for_heiko.py \
  --csv data/ppwr_audit_results_2nd_wave.csv \
  --xlsx data/results/1st_2nd_wave_results_for_heiko.xlsx

# Self-test (no real data needed)
python scripts/fill_results_for_heiko.py --self-test
```

## Streamlit Cloud deployment

The hosted app only needs `streamlit_app.py`, `dashboard_ppwr.py`, `evidence_validator.py`,
and `data/ppwr_audit_results.csv`. The heavy RAG stack (`requirements-audit.txt`) runs locally.

1. Generate/update results locally and commit `data/ppwr_audit_results.csv`.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from `leoalbarede/PPWR_tool`:
   - Main file: `streamlit_app.py`
   - Python: see `.python-version`
   - Requirements: `requirements.txt` (auto-detected)
3. Redeploy after each CSV update (push to `main`). No secrets are required (read-only CSV).

### Audit CSV columns

| Column | Meaning |
|--------|---------|
| Supplier No. | Supplier ID from the folder name |
| Doc list | Parent list folder (if any) |
| Supplier | Supplier name from the folder |
| PPWR compliant with heavy metals concentration limit | `yes` / `no` / `N/A` |
| PPWR SoC content | `yes` / `no` / `N/A` |
| PPWR PFAS content | `yes` / `no` / `N/A` (inverted semantics: see `evidence_validator.py`) |
| PPWR SVHC content | `yes` / `no` / `N/A` |
| Concentration | Stated values + verbatim quote and source PDF when found |

Answers are grounded in retrieved PDF text only; missing information is reported as `N/A` (no inference).

## Low-level RAG API

```python
import os
from dotenv import load_dotenv
from analyzer_docling import ask_pdf_multi_section

load_dotenv()

result = ask_pdf_multi_section(
    pdf_path="/path/to/report.pdf",
    instruction="Answer each numbered line: YES | NO | N/A with Evidence and Location.",
    sections=[("PPWR — packaging", "1. PPWR mentioned\n2. Recycled content disclosed")],
    api_key=os.getenv("OPENAI_API_KEY"),
    k=5,
)
print(result["retrieval_profile"], result["n_index_chunks"])
print(result["answer"])
```
