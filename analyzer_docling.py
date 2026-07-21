import hashlib
import json
import os
import pickle
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken_cache_setup  # noqa: F401 — stable TIKTOKEN_CACHE_DIR before OpenAI stack

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
try:  # langchain >= 1.0 moved legacy chains/retrievers to langchain_classic
    from langchain_classic.chains.question_answering import load_qa_chain
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ModuleNotFoundError:  # langchain 0.3.x
    from langchain.chains.question_answering import load_qa_chain
    from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from langfuse import observe

# --- Dossiers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKDOWN_CACHE_DIR = os.path.join(BASE_DIR, "markdown_cache")
VECTOR_CACHE_DIR = os.path.join(BASE_DIR, "vector_cache")
CHUNKS_PICKLE_NAME = "index_chunks.pkl"

# Bump when chunking, embedding model, index layout, or retrieval stack changes.
RAG_INDEX_VERSION = "5"
# `text-embedding-3-large` améliore le dense retrieval sur du texte ESG riche ; coût plus élevé.
# Surcharge possible : RAG_EMBEDDING_MODEL=text-embedding-3-small
_DEFAULT_EMBEDDING = "text-embedding-3-small"
EMBEDDING_MODEL = (os.getenv("RAG_EMBEDDING_MODEL") or _DEFAULT_EMBEDDING).strip() or _DEFAULT_EMBEDDING
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 400
# Recursive split only when a header section exceeds this (keeps short PPWR letters intact).
SECTION_SPLIT_THRESHOLD = CHUNK_SIZE + 200

# --- Adaptive retrieval profiles (auto-selected from chunk count) ---
# small  (≤6 chunks, ~≤5 pp): BM25 only, single query, no rerank, no vector index
# medium (7–20 chunks, ~6–15 pp): BM25 + dense MMR, single query, no rerank
# large  (>20 chunks): full hybrid + multi-query + FlashRank (legacy ESG mode)
PROFILE_SMALL_MAX_CHUNKS = 6
PROFILE_MEDIUM_MAX_CHUNKS = 20

# Hybrid weights (medium / large dense branch)
ENSEMBLE_WEIGHTS_BM25_DENSE: Tuple[float, float] = (0.35, 0.65)
MMR_LAMBDA_MULT = 0.55

# large-profile only (100-page ESG reports)
BRANCH_K_MIN = 28
BRANCH_K_FACTOR = 6
FETCH_K_FACTOR = 2
FETCH_K_MIN = 48
POOL_TOP_PER_SUBQUERY = 42
MAX_MERGED_BEFORE_RERANK = 78
RERANK_QUERY_MAX_CHARS = 1500
SECTION_BODY_MAX_CHARS = 2600
MAX_NUMBERED_SUBQUERIES = 8

# Focused BM25 queries for second pass when a section returns N/A.
NA_RETRY_QUERIES: Dict[str, List[str]] = {
    "heavy metals": [
        "lead cadmium mercury hexavalent chromium 100 ppm mg/kg packaging material",
        "Pb Cd Hg Cr6 sum concentration does not exceed compliant",
        "heavy metals packaging components limit declaration",
    ],
    "soc": [
        "substances of concern Article 3 packaging Regulation EU 2025/40",
        "SoC absent not present Annex XIV XVII REACH CMR STOT PBT recyclability",
        "substances of concern recycling streams packaging material declaration",
    ],
    "pfas": [
        "PFAS perfluoro polyfluoro packaging material µg/kg",
        "PFAS not detected absent fluorinated alkyl substances",
        "total PFAS polymeric 50 mg/kg 25 µg/kg",
    ],
    "svhc": [
        "SVHC substances of very high concern 0.1% w/w packaging",
        "SVHC not detected absent REACH candidate list Article 9",
        "substances of very high concern packaging material concentration",
    ],
}

if not os.path.exists(MARKDOWN_CACHE_DIR):
    os.makedirs(MARKDOWN_CACHE_DIR)
if not os.path.exists(VECTOR_CACHE_DIR):
    os.makedirs(VECTOR_CACHE_DIR)

# Limite d'affichage / sérialisation par chunk (le LLM voit le texte complet en interne).
DEBUG_CHUNK_CHARS = 12000


def _serialize_document_for_debug(doc: Document, max_chars: int = DEBUG_CHUNK_CHARS) -> Dict[str, Any]:
    text = doc.page_content or ""
    truncated = len(text) > max_chars
    meta: Dict[str, Any] = {}
    for k, v in (doc.metadata or {}).items():
        try:
            json.dumps(v)
            meta[k] = v
        except (TypeError, ValueError):
            meta[k] = str(v)
    return {
        "page_content": text[:max_chars] + ("… [truncated for UI]" if truncated else ""),
        "metadata": meta,
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_markdown_document(pdf_path: str) -> Document:
    pdf_filename = os.path.basename(pdf_path)
    md_filename = pdf_filename.replace(".pdf", ".md")
    md_path = os.path.join(MARKDOWN_CACHE_DIR, md_filename)

    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"📖 Chargement depuis le cache Markdown : {md_filename}")
        return Document(page_content=content, metadata={"source": pdf_path})

    print(
        f"🧠 Docling convertit le PDF en Markdown (première fois) : {pdf_filename}. "
        "Patientez quelques minutes ..."
    )
    loader = DoclingLoader(file_path=pdf_path, export_type=ExportType.MARKDOWN)
    data = loader.load()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(data[0].page_content)
    print(f"💾 Conversion sauvegardée : {md_filename}")
    return Document(page_content=data[0].page_content, metadata={"source": pdf_path})


def _split_documents(documents: List[Document]) -> List[Document]:
    """Découpage prioritaire par titres Markdown (sortie Docling), puis morcellement récursif si trop gros."""
    base = documents[0]
    text = base.page_content
    source = base.metadata.get("source", "")

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    try:
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        header_chunks = md_splitter.split_text(text)
    except Exception:
        header_chunks = []

    if not header_chunks or (len(header_chunks) == 1 and len(text) > CHUNK_SIZE * 2):
        return recursive.split_documents(
            [Document(page_content=text, metadata={"source": source})]
        )

    final: List[Document] = []
    for doc in header_chunks:
        doc.metadata.setdefault("source", source)
        if len(doc.page_content) > SECTION_SPLIT_THRESHOLD:
            final.extend(recursive.split_documents([doc]))
        else:
            final.append(doc)

    return final if final else recursive.split_documents(documents)


def _index_meta_path(persist_root: str) -> str:
    return os.path.join(persist_root, "rag_meta.json")


def _chunks_path(persist_root: str) -> str:
    return os.path.join(persist_root, CHUNKS_PICKLE_NAME)


def _load_index_meta(persist_root: str) -> Optional[dict]:
    path = _index_meta_path(persist_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_index_meta(persist_root: str, meta: dict) -> None:
    with open(_index_meta_path(persist_root), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _save_chunks(persist_root: str, chunks: List[Document]) -> None:
    with open(_chunks_path(persist_root), "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_chunks(persist_root: str) -> Optional[List[Document]]:
    path = _chunks_path(persist_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _expected_index_meta(pdf_hash: str, num_chunks: int) -> dict:
    return {
        "pdf_sha256": pdf_hash,
        "rag_index_version": RAG_INDEX_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_strategy": "markdown_headers_plus_recursive",
        "chunk_count": num_chunks,
        "retrieval": "adaptive_small_medium_large_v5",
    }


def _get_or_build_vectorstore(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings,
    pdf_hash: str,
) -> Tuple[Chroma, List[Document]]:
    """
    Chroma (dense) + pickle des chunks identiques pour BM25 à chaque session.
    """
    persist_root = os.path.join(VECTOR_CACHE_DIR, f"{pdf_hash}_{RAG_INDEX_VERSION}")
    chroma_dir = os.path.join(persist_root, "chroma")
    collection_name = "audit_doc"
    num_chunks = len(chunks)
    expected = _expected_index_meta(pdf_hash, num_chunks)
    existing = _load_index_meta(persist_root)

    if existing == expected and os.path.isdir(chroma_dir):
        loaded = _load_chunks(persist_root)
        if loaded is not None and len(loaded) == num_chunks:
            print(f"📇 Index vectoriel (cache) : {os.path.basename(persist_root)}")
            return (
                Chroma(
                    persist_directory=chroma_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name,
                ),
                loaded,
            )

    if os.path.isdir(persist_root):
        shutil.rmtree(persist_root, ignore_errors=True)
    os.makedirs(chroma_dir, exist_ok=True)

    print(f"🔨 Construction de l'index vectoriel : {os.path.basename(persist_root)}")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=chroma_dir,
    )
    _write_index_meta(persist_root, expected)
    _save_chunks(persist_root, chunks)
    return vs, chunks


def _dedupe_docs_preserve_order(docs: Iterable[Document]) -> List[Document]:
    seen: set[str] = set()
    out: List[Document] = []
    for d in docs:
        key = hashlib.sha256((d.page_content or "").encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _select_retrieval_profile(n_chunks: int) -> str:
    if n_chunks <= PROFILE_SMALL_MAX_CHUNKS:
        return "small"
    if n_chunks <= PROFILE_MEDIUM_MAX_CHUNKS:
        return "medium"
    return "large"


def _section_retrieval_query(section_title: str, points_body: str) -> str:
    title = (section_title or "").strip()
    body = (points_body or "").strip()
    return f"{title}\n{body[:SECTION_BODY_MAX_CHARS]}".strip()


def _retrieval_subqueries(section_title: str, points_body: str) -> List[str]:
    """Multi-query expansion — large documents only."""
    queries = [_section_retrieval_query(section_title, points_body)]
    numbered = 0
    for line in (points_body or "").splitlines():
        line = line.strip()
        if not line or not re.match(r"^\d+\.\s", line):
            continue
        queries.append(f"{(section_title or '').strip()}\n{line}")
        numbered += 1
        if numbered >= MAX_NUMBERED_SUBQUERIES:
            break
    return queries


def _rerank_query_short(section_title: str, points_body: str) -> str:
    body = (points_body or "").strip()
    return f"{(section_title or '').strip()}\n{body[:RERANK_QUERY_MAX_CHARS]}".strip()


def _k_eff(k: int, n: int) -> int:
    return min(max(k, 1), n)


def _build_bm25_retriever(chunks: List[Document], branch_k: int) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = branch_k
    return retriever


def _build_ensemble_retriever(
    vectorstore: Chroma,
    chunks: List[Document],
    k: int,
    *,
    large_mode: bool = False,
) -> Tuple[EnsembleRetriever, int, int, int]:
    n = len(chunks)
    if n == 0:
        raise ValueError("No text chunks to index.")

    k_eff = _k_eff(k, n)
    if large_mode:
        branch_k = min(max(k_eff * BRANCH_K_FACTOR, BRANCH_K_MIN), n)
        fetch_k = min(max(branch_k * FETCH_K_FACTOR, FETCH_K_MIN), n)
    else:
        branch_k = min(n, max(k_eff * 2, k_eff + 2))
        fetch_k = min(n, branch_k * 2)

    bm25_retriever = _build_bm25_retriever(chunks, branch_k)
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": branch_k,
            "fetch_k": fetch_k,
            "lambda_mult": MMR_LAMBDA_MULT,
        },
    )
    w_bm25, w_dense = ENSEMBLE_WEIGHTS_BM25_DENSE
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[w_bm25, w_dense],
    )
    return ensemble, k_eff, branch_k, fetch_k


def _truncate_to_k(docs: List[Document], k: int) -> List[Document]:
    return _dedupe_docs_preserve_order(docs)[:k]


def _retrieve_small(
    chunks: List[Document],
    k: int,
    section_title: str,
    points_body: str,
) -> Tuple[List[Document], List[str], str]:
    query = _section_retrieval_query(section_title, points_body)
    k_eff = _k_eff(k, len(chunks))
    if len(chunks) <= k_eff:
        return list(chunks), [query], query
    bm25 = _build_bm25_retriever(chunks, k_eff)
    return _truncate_to_k(bm25.invoke(query), k_eff), [query], query


def _retrieve_medium(
    vectorstore: Chroma,
    chunks: List[Document],
    k: int,
    section_title: str,
    points_body: str,
) -> Tuple[List[Document], List[str], str]:
    query = _section_retrieval_query(section_title, points_body)
    ensemble, k_eff, _, _ = _build_ensemble_retriever(
        vectorstore, chunks, k, large_mode=False
    )
    return _truncate_to_k(ensemble.invoke(query), k_eff), [query], query


def _retrieve_large(
    ensemble: EnsembleRetriever,
    reranker: FlashrankRerank,
    k: int,
    n_chunks: int,
    section_title: str,
    points_body: str,
) -> Tuple[List[Document], List[str], str]:
    subqs = _retrieval_subqueries(section_title, points_body)
    k_eff = _k_eff(k, n_chunks)
    merged: List[Document] = []
    for sq in subqs:
        ranked = ensemble.invoke(sq)
        for d in ranked[:POOL_TOP_PER_SUBQUERY]:
            merged.append(d)
        merged = _dedupe_docs_preserve_order(merged)
        if len(merged) >= MAX_MERGED_BEFORE_RERANK:
            merged = merged[:MAX_MERGED_BEFORE_RERANK]
            break
    rerank_q = _rerank_query_short(section_title, points_body)
    if not merged:
        return [], subqs, rerank_q
    reranker.top_n = k_eff
    final = list(reranker.compress_documents(merged, rerank_q))
    return final, subqs, rerank_q


def _retrieve_for_section(
    profile: str,
    chunks: List[Document],
    k: int,
    section_title: str,
    points_body: str,
    vectorstore: Optional[Chroma] = None,
    ensemble: Optional[EnsembleRetriever] = None,
    reranker: Optional[FlashrankRerank] = None,
) -> Tuple[List[Document], List[str], str]:
    if profile == "small":
        return _retrieve_small(chunks, k, section_title, points_body)
    if profile == "medium":
        if vectorstore is None:
            raise ValueError("vectorstore required for medium profile")
        return _retrieve_medium(vectorstore, chunks, k, section_title, points_body)
    if ensemble is None or reranker is None:
        raise ValueError("ensemble and reranker required for large profile")
    return _retrieve_large(
        ensemble, reranker, k, len(chunks), section_title, points_body
    )


def _prepare_index(
    pdf_path: str,
    api_key: str,
) -> Tuple[Optional[Chroma], str, List[Document], str]:
    pdf_hash = _sha256_file(pdf_path)
    doc = _load_markdown_document(pdf_path)
    chunks = _split_documents([doc])
    profile = _select_retrieval_profile(len(chunks))
    print(f"📊 Profil retrieval : {profile} ({len(chunks)} chunk(s))")

    vectorstore: Optional[Chroma] = None
    if profile in ("medium", "large"):
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
        vectorstore, chunks_out = _get_or_build_vectorstore(chunks, embeddings, pdf_hash)
        return vectorstore, pdf_hash, chunks_out, profile

    return None, pdf_hash, chunks, profile


def get_document_markdown(pdf_path: str) -> str:
    """Full markdown text (from cache or Docling) for evidence validation."""
    return _load_markdown_document(pdf_path).page_content


def _na_retry_queries_for(section_title: str) -> List[str]:
    title = (section_title or "").lower()
    if "heavy metal" in title or "pb" in title:
        return NA_RETRY_QUERIES["heavy metals"]
    if "soc" in title or "substances of concern" in title:
        return NA_RETRY_QUERIES["soc"]
    if "pfas" in title:
        return NA_RETRY_QUERIES["pfas"]
    if "svhc" in title or "article 9" in title:
        return NA_RETRY_QUERIES["svhc"]
    return []


def _section_answer_is_na(answer_text: str) -> bool:
    m = re.search(r"Answer:\s*(YES|NO|N/A)\b", answer_text or "", re.IGNORECASE)
    return bool(m and m.group(1).upper() == "N/A")


def _retrieve_by_queries(
    profile: str,
    chunks: List[Document],
    k: int,
    queries: List[str],
    vectorstore: Optional[Chroma] = None,
    ensemble: Optional[EnsembleRetriever] = None,
) -> List[Document]:
    """Merge BM25 / ensemble hits for multiple short retry queries."""
    if not queries or not chunks:
        return []
    k_pool = min(len(chunks), max(_k_eff(k, len(chunks)) * 2, _k_eff(k, len(chunks)) + 3))
    merged: List[Document] = []

    if profile == "small":
        bm25 = _build_bm25_retriever(chunks, min(len(chunks), k_pool * 2))
        for q in queries:
            for doc in bm25.invoke(q):
                merged.append(doc)
            merged = _dedupe_docs_preserve_order(merged)
    elif profile == "medium" and vectorstore is not None:
        ensemble, branch_k, _, _ = _build_ensemble_retriever(
            vectorstore, chunks, k, large_mode=False
        )
        for q in queries:
            for doc in ensemble.invoke(q)[:branch_k]:
                merged.append(doc)
            merged = _dedupe_docs_preserve_order(merged)
    elif profile == "large" and ensemble is not None:
        for q in queries:
            for doc in ensemble.invoke(q)[:POOL_TOP_PER_SUBQUERY]:
                merged.append(doc)
            merged = _dedupe_docs_preserve_order(merged)

    return merged[:k_pool]


@observe
def ask_pdf_multi_section(
    pdf_path: str,
    instruction: str,
    sections: List[Tuple[str, str]],
    api_key: str,
    k: int = 5,
    return_retrieval_debug: bool = False,
    retrieval_profile: Optional[str] = None,
    retry_na_sections: bool = False,
) -> dict:
    """
    Plusieurs passes retrieval (une par bloc thématique) sur le même index.

    Profil auto (selon le nombre de chunks) :
    - small  : BM25 seul, pas d'index vectoriel ni FlashRank
    - medium : BM25 + dense MMR, pas de FlashRank
    - large  : hybride + multi-requêtes + FlashRank (rapports longs)

    sections: liste de (titre_section, texte_points_numérotés).
    retry_na_sections: seconde passe retrieval ciblée si Answer: N/A.
    """
    if not sections:
        out: Dict[str, Any] = {"answer": "", "source_file": os.path.basename(pdf_path)}
        if return_retrieval_debug:
            out["retrieval_debug"] = []
        return out

    vectorstore, _pdf_hash, chunks, auto_profile = _prepare_index(pdf_path, api_key)
    profile = retrieval_profile or auto_profile
    if profile not in ("small", "medium", "large"):
        raise ValueError(f"Invalid retrieval_profile: {profile}")

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
        max_tokens=2500,
    )
    combine = load_qa_chain(llm, chain_type="stuff")

    ensemble: Optional[EnsembleRetriever] = None
    reranker: Optional[FlashrankRerank] = None
    k_eff = _k_eff(k, len(chunks))

    if profile == "large":
        if vectorstore is None:
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
            vectorstore, chunks = _get_or_build_vectorstore(
                chunks, embeddings, _sha256_file(pdf_path)
            )
        ensemble, k_eff, _, _ = _build_ensemble_retriever(
            vectorstore, chunks, k, large_mode=True
        )
        from flashrank_cache_setup import get_flashrank_ranker

        reranker = FlashrankRerank(top_n=k_eff, client=get_flashrank_ranker())

    blocks: List[str] = []
    retrieval_rows: List[Dict[str, Any]] = []
    for title, points in sections:
        points_stripped = points.strip()
        query_llm = (
            f"{instruction}\n\n"
            f"SECTION: {title}\n"
            f"Answer ONLY for the numbered points in this section. "
            f"Keep the same line format for each point.\n"
        )
        title_lower = title.lower()
        if "soc" in title_lower or "substances of concern" in title_lower:
            query_llm += (
                "IMPORTANT: Evidence for SoC must relate to Substances of Concern under PPWR Article 3(2)(a) "
                "(REACH Annex XIV/XVII, CMR, STOT/PBT/vPvB, recyclability). "
                "Do NOT reuse heavy metals or PFAS quotes. Use N/A if SoC is not discussed.\n\n"
            )
        elif "heavy metal" in title_lower or "pb, cd" in title_lower:
            query_llm += (
                "IMPORTANT: Evidence for heavy metals must mention Pb, Cd, Hg, Cr6+, or heavy metals. "
                "Do NOT reuse SoC/SVHC/PFAS quotes unless they explicitly state metal limits.\n\n"
            )
        elif "pfas" in title_lower:
            query_llm += (
                "IMPORTANT: Evidence for PFAS must mention PFAS, perfluoro, or polyfluoro substances. "
                "Do NOT reuse heavy metals or SVHC quotes. Use N/A if PFAS are not discussed.\n\n"
            )
        elif "svhc" in title_lower or "article 9" in title_lower:
            query_llm += (
                "IMPORTANT: Evidence for SVHC must mention SVHC, substances of very high concern, "
                "or Article 9 / 0.1% w/w. Do NOT reuse PFAS or heavy metals quotes.\n\n"
            )
        query_llm += f"POINTS TO VERIFY:\n{points_stripped}\n"
        source_docs, subqs, rerank_q = _retrieve_for_section(
            profile,
            chunks,
            k,
            title,
            points_stripped,
            vectorstore=vectorstore,
            ensemble=ensemble,
            reranker=reranker,
        )
        pool_empty = not source_docs
        if pool_empty:
            source_docs = [
                Document(
                    page_content=(
                        "(No matching passages were retrieved from the report for this section. "
                        "Answer each checklist line using N/A and Evidence: NONE.)"
                    ),
                    metadata={"source": "retrieval_empty"},
                )
            ]

        qa_out = combine.invoke(
            {"input_documents": source_docs, "question": query_llm}
        )
        answer = (qa_out.get("output_text") or "").strip()
        retried = False

        if retry_na_sections and _section_answer_is_na(answer):
            retry_qs = _na_retry_queries_for(title)
            if retry_qs:
                retry_docs = _retrieve_by_queries(
                    profile,
                    chunks,
                    k,
                    retry_qs,
                    vectorstore=vectorstore,
                    ensemble=ensemble,
                )
                if retry_docs:
                    merged = _dedupe_docs_preserve_order(source_docs + retry_docs)
                    if pool_empty:
                        merged = retry_docs
                    merged = merged[: max(k_eff * 2, k_eff)]
                    print(f"  ↻ Retry retrieval (section was N/A): {title}")
                    qa_out = combine.invoke(
                        {"input_documents": merged, "question": query_llm}
                    )
                    answer = (qa_out.get("output_text") or "").strip()
                    source_docs = merged
                    pool_empty = False
                    retried = True

        blocks.append(f"=== {title} ===\n{answer}")
        if return_retrieval_debug:
            debug_docs = [] if pool_empty else source_docs
            retrieval_rows.append(
                {
                    "section_title": title,
                    "retrieval_profile": profile,
                    "retrieval_retried": retried,
                    "query": query_llm,
                    "llm_query": query_llm,
                    "retrieval_subqueries": subqs,
                    "rerank_query": rerank_q,
                    "retrieval_pool_empty": pool_empty,
                    "n_chunks": len(debug_docs),
                    "chunks": [_serialize_document_for_debug(d) for d in debug_docs],
                }
            )

    result: Dict[str, Any] = {
        "answer": "\n\n".join(blocks),
        "source_file": os.path.basename(pdf_path),
        "retrieval_profile": profile,
        "n_index_chunks": len(chunks),
    }
    if return_retrieval_debug:
        result["retrieval_debug"] = retrieval_rows
    return result
