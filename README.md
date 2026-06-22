# PPWR tool — RAG core

Reusable document-audit engine extracted from the PlanetCare ESG pipeline.

## Modules

| File | Role |
|------|------|
| `analyzer_docling.py` | PDF → Markdown (cache), adaptive RAG by doc size (small / medium / large) |
| `ppwr_audit.py` | Batch PPWR audit: supplier folders → CSV (heavy metals + SoC) |
| `tiktoken_cache_setup.py` | Persistent local tiktoken cache (corporate proxy / macOS temp) |
| `flashrank_cache_setup.py` | FlashRank cache (large documents only, >20 chunks) |

Caches are created automatically under `markdown_cache/`, `vector_cache/`, `tiktoken_cache/`, and `flashrank_cache/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
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

On first run, tiktoken may download an encoding file unless cached. FlashRank is only loaded for large documents (>20 chunks).

### Adaptive retrieval

| Profile | Chunks | Strategy |
|---------|--------|----------|
| `small` | ≤ 6 (~≤5 pp) | BM25 only, no vector index, no rerank |
| `medium` | 7–20 (~6–15 pp) | BM25 + dense MMR, no rerank |
| `large` | > 20 | Full hybrid + multi-query + FlashRank |

Profile is selected automatically; override with `retrieval_profile=` in `ask_pdf_multi_section`.

## PPWR supplier audit

Binary compliance check on regulatory PDFs per CMO supplier folder under `docs/`.

Each subfolder must be named: `{Supplier No} - {Supplier name}` (e.g. `S00122_0100114617 - AMCOR FLEXIBLES SELESTAT`).

```bash
# List suppliers and PDF counts
python ppwr_audit.py --list-suppliers

# Run full audit → ppwr_audit_results.csv
python ppwr_audit.py

# One supplier only (faster test)
python ppwr_audit.py --supplier "CONSTANTIA"

# Separate evidence columns for manual review
python ppwr_audit.py --with-evidence-columns

# Dashboard (after audit CSV exists)
streamlit run dashboard_ppwr.py
```

**CSV columns (default):**

| Column | Meaning |
|--------|---------|
| Supplier No. | Folder prefix before ` - ` |
| Supplier | Supplier name from folder |
| PPWR compliant with heavy metals concentration limit | `yes` / `no` / `N/A` — sum of Pb, Cd, Hg, Cr6+ &lt; 100 mg |
| PPWR SoC content | `yes` / `no` / `N/A` — presence of Substances of Concern |
| Concentration | Stated values + verbatim quote and source PDF when found |

Answers are grounded in retrieved PDF text only; missing information is reported as `N/A` (no inference).

## Low-level RAG API

```python
import os
from dotenv import load_dotenv
from analyzer_docling import ask_pdf_multi_section

load_dotenv()

sections = [
    (
        "PPWR — packaging",
        """
1. PPWR or Directive 94/62/EC mentioned
2. Recycled content disclosed
""".strip(),
    ),
]

result = ask_pdf_multi_section(
    pdf_path="/path/to/report.pdf",
    instruction="Answer each numbered line: YES | NO | N/A with Evidence and Location.",
    sections=sections,
    api_key=os.getenv("OPENAI_API_KEY"),
    k=5,
)

print(result["retrieval_profile"], result["n_index_chunks"])
print(result["answer"])
```

Add your own checklist, output parser, and UI on top of this core.
