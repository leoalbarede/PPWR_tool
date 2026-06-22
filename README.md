# PPWR tool — RAG core

Reusable document-audit engine extracted from the PlanetCare ESG pipeline.

## Modules

| File | Role |
|------|------|
| `analyzer_docling.py` | PDF → Markdown (cache), vector index, multi-section hybrid RAG (BM25 + dense + FlashRank) |
| `file_manager.py` | Fuzzy PDF lookup by entity name and year |
| `tiktoken_cache_setup.py` | Persistent local tiktoken cache (corporate proxy / macOS temp) |
| `flashrank_cache_setup.py` | Persistent FlashRank reranker model cache |

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

On first run, tiktoken and FlashRank may download artifacts unless caches are pre-populated (see setup modules).

## Usage

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
    k=10,
)

print(result["source_file"])
print(result["answer"])
```

Add your own checklist, output parser, and UI on top of this core.
