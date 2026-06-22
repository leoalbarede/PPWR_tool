import hashlib
import json
import os
import pickle
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken_cache_setup  # noqa: F401 — stable TIKTOKEN_CACHE_DIR before OpenAI stack
import flashrank_cache_setup
from flashrank_cache_setup import get_flashrank_ranker

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document

from langfuse import observe

# --- Dossiers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKDOWN_CACHE_DIR = os.path.join(BASE_DIR, "markdown_cache")
VECTOR_CACHE_DIR = os.path.join(BASE_DIR, "vector_cache")
CHUNKS_PICKLE_NAME = "index_chunks.pkl"

# Bump when chunking, embedding model, index layout, or retrieval stack changes.
RAG_INDEX_VERSION = "4"
# `text-embedding-3-large` améliore le dense retrieval sur du texte ESG riche ; coût plus élevé.
# Surcharge possible : RAG_EMBEDDING_MODEL=text-embedding-3-small
_DEFAULT_EMBEDDING = "text-embedding-3-small"
EMBEDDING_MODEL = (os.getenv("RAG_EMBEDDING_MODEL") or _DEFAULT_EMBEDDING).strip() or _DEFAULT_EMBEDDING
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 400

# --- Retrieval (hybride RRF + multi-requêtes + rerank court) ---
# Plus de poids sur le dense : thématique / reformulations dans les rapports ESG.
ENSEMBLE_WEIGHTS_BM25_DENSE: Tuple[float, float] = (0.35, 0.65)
MMR_LAMBDA_MULT = 0.55
BRANCH_K_MIN = 28
BRANCH_K_FACTOR = 6
FETCH_K_FACTOR = 2
FETCH_K_MIN = 48
# Par sous-requête : tronquer la sortie fusionnée pour limiter latence avant rerank global.
POOL_TOP_PER_SUBQUERY = 42
# Documents uniques max passés au cross-encodeur FlashRank.
MAX_MERGED_BEFORE_RERANK = 78
# Requête FlashRank : rester sous ~512 tokens utiles pour le CE.
RERANK_QUERY_MAX_CHARS = 1500
# Première sous-requête : titre + checklist (sans l’instruction LLM longue).
SECTION_BODY_MAX_CHARS = 2600
# Lignes numérotées supplémentaires (1 requête courte par ligne).
MAX_NUMBERED_SUBQUERIES = 8

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
        if len(doc.page_content) > CHUNK_SIZE + 200:
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
        "retrieval": "hybrid_rrf_mmr_flashrank_multiq_short_rerank_v4",
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


def _retrieval_subqueries(section_title: str, points_body: str) -> List[str]:
    """
    Requêtes courtes pour BM25 + dense (pas l’instruction LLM complète).
    - Une requête « checklist » tronquée.
    - Une requête par ligne numérotée (alignée au cross-encodeur / lexique).
    """
    title = (section_title or "").strip()
    body = (points_body or "").strip()
    queries: List[str] = [f"{title}\n{body[:SECTION_BODY_MAX_CHARS]}".strip()]
    numbered = 0
    for line in body.splitlines():
        line = line.strip()
        if not line or not re.match(r"^\d+\.\s", line):
            continue
        queries.append(f"{title}\n{line}")
        numbered += 1
        if numbered >= MAX_NUMBERED_SUBQUERIES:
            break
    return queries


def _rerank_query_short(section_title: str, points_body: str) -> str:
    body = (points_body or "").strip()
    return f"{(section_title or '').strip()}\n{body[:RERANK_QUERY_MAX_CHARS]}".strip()


def _build_ensemble_retriever(
    vectorstore: Chroma,
    chunks: List[Document],
    k: int,
) -> Tuple[EnsembleRetriever, int, int, int]:
    """
    BM25 + MMR dense, fusion RRF (sans FlashRank — le rerank est appliqué ensuite
    sur le pool fusionné multi-requêtes).
    Retourne (ensemble, k_eff, branch_k, fetch_k).
    """
    n = len(chunks)
    if n == 0:
        raise ValueError("No text chunks to index.")

    k_eff = min(max(k, 1), n)
    branch_k = min(max(k_eff * BRANCH_K_FACTOR, BRANCH_K_MIN), n)
    fetch_k = min(max(branch_k * FETCH_K_FACTOR, FETCH_K_MIN), n)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = branch_k

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


def _retrieve_merge_rerank(
    ensemble: EnsembleRetriever,
    reranker: FlashrankRerank,
    section_title: str,
    points_body: str,
) -> Tuple[List[Document], List[str], str]:
    subqs = _retrieval_subqueries(section_title, points_body)
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
    final = list(reranker.compress_documents(merged, rerank_q))
    return final, subqs, rerank_q


def _prepare_index(pdf_path: str, api_key: str) -> Tuple[Chroma, str, List[Document]]:
    pdf_hash = _sha256_file(pdf_path)
    doc = _load_markdown_document(pdf_path)
    chunks = _split_documents([doc])
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
    vectorstore, chunks_out = _get_or_build_vectorstore(chunks, embeddings, pdf_hash)
    return vectorstore, pdf_hash, chunks_out


@observe
def ask_pdf_multi_section(
    pdf_path: str,
    instruction: str,
    sections: List[Tuple[str, str]],
    api_key: str,
    k: int = 10,
    return_retrieval_debug: bool = False,
) -> dict:
    """
    Plusieurs passes retrieval (une par bloc thématique) sur le même index.
    sections: liste de (titre_section, texte_points_numérotés).

    Retrieval : sous-requêtes courtes (checklist + une par ligne numérotée), fusion
    des pools RRF, puis rerank FlashRank sur une requête courte. Le LLM reçoit la
    question complète (instruction + section + points) avec uniquement ces chunks.

    Si return_retrieval_debug=True, la clé retrieval_debug contient, par section,
    les documents finaux (post rerank) et les requêtes utilisées.
    """
    if not sections:
        out: Dict[str, Any] = {"answer": "", "source_file": os.path.basename(pdf_path)}
        if return_retrieval_debug:
            out["retrieval_debug"] = []
        return out

    vectorstore, _pdf_hash, chunks = _prepare_index(pdf_path, api_key)
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
        max_tokens=2500,
    )
    ensemble, k_eff, _branch_k, _fetch_k = _build_ensemble_retriever(vectorstore, chunks, k)
    reranker = FlashrankRerank(top_n=k_eff, client=get_flashrank_ranker())
    combine = load_qa_chain(llm, chain_type="stuff")

    blocks: List[str] = []
    retrieval_rows: List[Dict[str, Any]] = []
    for title, points in sections:
        points_stripped = points.strip()
        query_llm = (
            f"{instruction}\n\n"
            f"SECTION: {title}\n"
            f"Answer ONLY for the numbered points in this section. "
            f"Keep the same line format for each point.\n\n"
            f"POINTS TO VERIFY:\n{points_stripped}\n"
        )
        source_docs, subqs, rerank_q = _retrieve_merge_rerank(
            ensemble, reranker, title, points_stripped
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
        blocks.append(f"=== {title} ===\n{answer}")
        if return_retrieval_debug:
            debug_docs = [] if pool_empty else source_docs
            retrieval_rows.append(
                {
                    "section_title": title,
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
    }
    if return_retrieval_debug:
        result["retrieval_debug"] = retrieval_rows
    return result
