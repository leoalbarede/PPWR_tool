"""Post-RAG validation: evidence must appear in source markdown; reject legal boilerplate."""

from __future__ import annotations

import html
import re
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

MIN_EVIDENCE_CHARS = 15

CHECK_HEAVY_METALS = "heavy_metals"
CHECK_SOC = "soc"
CHECK_PFAS = "pfas"
CHECK_SVHC = "svhc"

CHECK_ORDER = [CHECK_HEAVY_METALS, CHECK_SOC, CHECK_PFAS, CHECK_SVHC]

__all__ = [
    "CHECK_HEAVY_METALS",
    "CHECK_SOC",
    "CHECK_PFAS",
    "CHECK_SVHC",
    "CHECK_ORDER",
    "dedupe_findings",
    "evidence_duplicates_prior_check",
    "evidence_in_source",
    "evidence_matches_check",
    "is_regulatory_boilerplate_only",
    "normalize_text_for_match",
    "reject_shared_evidence",
    "soc_evidence_duplicates_heavy_metals",
    "correct_inverted_check_finding",
    "correct_inverted_raw_answer",
    "recover_pfas_from_markdown",
    "evidence_indicates_compliant",
    "evidence_indicates_noncompliant",
    "INVERTED_CHECKS",
    "evidence_declarant_mismatch",
    "validate_point_finding",
]

# Topic signals — evidence must match the check it supports.
_HM_TOPIC_RE = re.compile(
    r"(?:"
    r"heavy\s+metals?|"
    r"\blead\b|\bpb\b|"
    r"cadmium|\bcd\b|"
    r"mercury|\bhg\b|"
    r"chromium|cr\s*vi|cr6|hexavalent\s+chromium|"
    r"100\s*(?:ppm|mg/kg|mg)"
    r")",
    re.IGNORECASE,
)

_SOC_TOPIC_RE = re.compile(
    r"(?:"
    r"substances?\s+of\s+concern|"
    r"\bsoc\b|"
    r"article\s*3\s*[\(\[]?\s*2\s*[\)\]]?\s*[\(\[]?\s*a|"
    r"regulation\s*\(eu\)\s*2025\s*/\s*40|"
    r"annex\s*xiv|"
    r"annex\s*xvii|"
    r"\bcmr\b|"
    r"\bstot\b|"
    r"\bpbt\b|"
    r"\bvpvb\b|"
    r"recyclab|"
    r"recycling\s+stream|"
    r"concern\s+substances?"
    r")",
    re.IGNORECASE,
)

_PFAS_TOPIC_RE = re.compile(
    r"(?:"
    r"\bpfas\b|"
    r"per-?\s*and\s*polyfluoro|"
    r"perfluoro|"
    r"polyfluoro|"
    r"fluorinated\s+alkyl|"
    r"total\s+fluorine|"
    r"article\s*5[\.\s]*5?|"
    r"\btf\b|\btof\b|"
    r"25\s*(?:µg|ug)/kg|"
    r"250\s*(?:µg|ug)/kg|"
    r"50\s*(?:mg/kg|ppm).*(?:pfas|fluorine)|"
    r"total\s+pfas|"
    r"polymeric\s+pfas"
    r")",
    re.IGNORECASE,
)

# PFAS / SoC / SVHC: CSV "yes" = substance detected or above limit; "no" = compliant / absent.
INVERTED_CHECKS = frozenset({CHECK_SOC, CHECK_PFAS, CHECK_SVHC})

_PFAS_COMPLIANT_EVIDENCE_RE = re.compile(
    r"(?:"
    r"limit(?:s)?\s+(?:for\s+)?pfas.{0,120}?(?:is|are)\s+met|"
    r"specific\s+limit\s+for\s+pfas.{0,120}?(?:is|are)\s+met|"
    r"requirements?\s+for\s+pfas.{0,150}?(?:is|are)\s+met|"
    r"requirements?\s+as\s+set\s+out\s+in\s+article\s*5\.?5?.{0,80}?(?:is|are)\s+met|"
    r"pfas.{0,120}?(?:is|are)\s+met|"
    r"pfas\s+are\s+not\s+used|"
    r"no\s+pfas|"
    r"(?:pfas|fluorine).{0,80}?not\s+detected|"
    r"(?:pfas|fluorine).{0,80}?not\s+present|"
    r"(?:pfas|fluorine).{0,80}?below\s+(?:the\s+)?limit|"
    r"total\s+fluorine.{0,80}?(?:met|within)|"
    r"analytical\s+investigations?\s+showing\s+that.{0,120}?(?:is|are)\s+met"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_PFAS_NONCOMPLIANT_EVIDENCE_RE = re.compile(
    r"(?:"
    r"above\s+(?:the\s+)?limit|"
    r"exceeds?\s+(?:the\s+)?limit|"
    r"non-?compliant|"
    r"detected\s+above|"
    r"present\s+above"
    r")",
    re.IGNORECASE,
)

_SVHC_TOPIC_RE = re.compile(
    r"(?:"
    r"\bsvhc\b|"
    r"substances?\s+of\s+very\s+high\s+concern|"
    r"reach\s+candidate|"
    r"authorisation\s+list|"
    r"article\s*9|"
    r"0\.1\s*%\s*(?:w/w|by\s+weight)?"
    r")",
    re.IGNORECASE,
)

_TOPIC_RES: Dict[str, re.Pattern] = {
    CHECK_HEAVY_METALS: _HM_TOPIC_RE,
    CHECK_SOC: _SOC_TOPIC_RE,
    CHECK_PFAS: _PFAS_TOPIC_RE,
    CHECK_SVHC: _SVHC_TOPIC_RE,
}

# Legal / regulatory wording without a supplier material declaration.
_LEGAL_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"article\s*5\s*[\(\[]?\d|"
    r"ppwr\s+requires|"
    r"sets?\s+a\s+limit|"
    r"shall\s+be\s+minim|"
    r"regulation\s+\(?eu\)?|"
    r"directive\s+\(?eu\)?|"
    r"legal\s+requirement|"
    r"packaging\s+and\s+packaging\s+waste\s+regulation"
    r")",
    re.IGNORECASE,
)

_MATERIAL_DECLARATION_RE = re.compile(
    r"(?:"
    r"our\s+(?:packaging|material|product|article)|"
    r"supplied\s+(?:packaging|material|food\s+packaging)|"
    r"we\s+(?:confirm|declare|certify|state)|"
    r"(?:does|do|shall)\s+not\s+(?:contain|exceed)|"
    r"not\s+detected|"
    r"not\s+present|"
    r"in\s+(?:the|our)\s+(?:packaging|material|article|product)|"
    r"above-mentioned\s+material|"
    r"packaging\s+materials?\s+(?:do|shall|does)|"
    r"neither\s+in\s+our\s+production|"
    r"combined\s+total\s+amount|"
    r"has\s+conducted|"
    r"analytical\s+investigations|"
    r"limit(?:s)?\s+(?:for\s+)?pfas.*(?:is|are)\s+met|"
    r"requirements?.*(?:is|are)\s+met"
    r")",
    re.IGNORECASE,
)


def normalize_text_for_match(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def evidence_in_source(evidence: str, source_markdown: str) -> bool:
    """True when the quote (normalized) is a substring of the source markdown."""
    if not evidence or evidence.strip().upper() == "NONE":
        return False
    ev = normalize_text_for_match(evidence)
    src = normalize_text_for_match(source_markdown)
    if len(ev) < MIN_EVIDENCE_CHARS:
        return False
    if ev in src:
        return True
    ev_trim = ev.rstrip(".,;:")
    if ev_trim in src:
        return True
    if len(ev) > 40 and ev[: max(40, len(ev) - 20)] in src:
        return True
    return False


def evidence_matches_check(evidence: str, check: str) -> bool:
    """True when the quote is on-topic for the requested PPWR check."""
    if not evidence or evidence.strip().upper() == "NONE":
        return False
    pattern = _TOPIC_RES.get(check)
    if pattern is None:
        return True
    if not pattern.search(evidence):
        return False
    hm = bool(_HM_TOPIC_RE.search(evidence))
    if check == CHECK_SOC and hm and not _SOC_TOPIC_RE.search(evidence):
        return False
    if check == CHECK_SVHC and _PFAS_TOPIC_RE.search(evidence) and not _SVHC_TOPIC_RE.search(evidence):
        return False
    if check == CHECK_PFAS and _SVHC_TOPIC_RE.search(evidence) and not _PFAS_TOPIC_RE.search(evidence):
        return False
    return True


def _evidence_same_quote(a: str, b: str) -> bool:
    na = normalize_text_for_match(a)
    nb = normalize_text_for_match(b)
    if not na or not nb or na == "none" or nb == "none":
        return False
    if na == nb:
        return True
    return na in nb or nb in na


def _downgrade_finding(finding, notes: str):
    finding.answer = "N/A"
    finding.evidence = "NONE"
    finding.concentration = "N/A"
    finding.notes = notes


def dedupe_findings(findings: Dict[str, object]) -> Dict[str, object]:
    """Downgrade checks that reuse another check's quote without a matching topic."""
    for i, check_i in enumerate(CHECK_ORDER):
        fi = findings.get(check_i)
        if fi is None or fi.answer not in ("yes", "no"):
            continue
        for check_j in CHECK_ORDER[:i]:
            fj = findings.get(check_j)
            if fj is None or fj.answer not in ("yes", "no"):
                continue
            if not _evidence_same_quote(fi.evidence, fj.evidence):
                continue
            match_i = evidence_matches_check(fi.evidence, check_i)
            match_j = evidence_matches_check(fj.evidence, check_j)
            if match_j and not match_i:
                _downgrade_finding(
                    fi,
                    f"Evidence relates to {check_j.replace('_', ' ')} only.",
                )
                break
            if not match_i and _evidence_same_quote(fi.evidence, fj.evidence):
                _downgrade_finding(
                    fi,
                    f"Same evidence as {check_j.replace('_', ' ')} check; not valid here.",
                )
                break
    return findings


def soc_evidence_duplicates_heavy_metals(hm_evidence: str, soc_evidence: str) -> bool:
    """True when SoC reuses the same heavy-metals quote."""
    if not _evidence_same_quote(hm_evidence, soc_evidence):
        return False
    hm_match = evidence_matches_check(hm_evidence, CHECK_HEAVY_METALS)
    soc_match = evidence_matches_check(soc_evidence, CHECK_SOC)
    if hm_match and not soc_match:
        return True
    if hm_match and soc_match:
        return True
    return False


def evidence_duplicates_prior_check(
    prior_evidence: str, check_evidence: str, prior_check: str, check: str
) -> bool:
    if not _evidence_same_quote(prior_evidence, check_evidence):
        return False
    prior_match = evidence_matches_check(prior_evidence, prior_check)
    check_match = evidence_matches_check(check_evidence, check)
    if prior_match and not check_match:
        return True
    return False


def evidence_indicates_compliant(check: str, evidence: str) -> bool:
    if check == CHECK_PFAS:
        if not _PFAS_TOPIC_RE.search(evidence):
            return False
        return bool(_PFAS_COMPLIANT_EVIDENCE_RE.search(evidence))
    return False


def evidence_indicates_noncompliant(check: str, evidence: str) -> bool:
    if check == CHECK_PFAS:
        return bool(_PFAS_NONCOMPLIANT_EVIDENCE_RE.search(evidence))
    return False


def correct_inverted_check_finding(finding, check: str):
    """
    Fix misclassified inverted checks (PFAS/SoC/SVHC).

    CSV yes = detected / above limit; CSV no = compliant / absent.
    """
    if check not in INVERTED_CHECKS:
        return finding
    if finding.answer not in ("yes", "no") or finding.evidence in ("", "NONE"):
        return finding
    compliant = evidence_indicates_compliant(check, finding.evidence)
    noncompliant = evidence_indicates_noncompliant(check, finding.evidence)
    if finding.answer == "yes" and compliant and not noncompliant:
        finding.answer = "no"
        note = "Corrected: evidence indicates compliance (limit met / not detected)."
        finding.notes = note if finding.notes in ("", "NONE") else f"{finding.notes} {note}"
    elif finding.answer == "no" and noncompliant and not compliant:
        finding.answer = "yes"
        note = "Corrected: evidence indicates non-compliance."
        finding.notes = note if finding.notes in ("", "NONE") else f"{finding.notes} {note}"
    return finding


def correct_inverted_raw_answer(check: str, raw_answer: str, evidence: str) -> str:
    """Dashboard/CSV post-correction for inverted checks."""
    if check not in INVERTED_CHECKS:
        return raw_answer
    if raw_answer not in ("yes", "no") or not evidence or evidence in ("", "NONE", "—"):
        return raw_answer
    compliant = evidence_indicates_compliant(check, evidence)
    noncompliant = evidence_indicates_noncompliant(check, evidence)
    if raw_answer == "yes" and compliant and not noncompliant:
        return "no"
    if raw_answer == "no" and noncompliant and not compliant:
        return "yes"
    return raw_answer


def recover_pfas_from_markdown(markdown: str, source_file: str = ""):
    """Extract a PFAS finding when the LLM returned N/A but source text supports it."""
    if not markdown:
        return None

    def _quote_around(match: re.Match) -> str:
        start = max(0, match.start() - 120)
        end = min(len(markdown), match.end() + 180)
        return re.sub(r"\s+", " ", markdown[start:end]).strip()[:300]

    compliant_hits: List[Tuple[int, str]] = []
    noncompliant_hits: List[str] = []

    def _score_compliant(quote: str) -> int:
        q = quote.lower()
        score = 0
        if "requirements for pfas" in q and "are met" in q:
            score += 20
        if "limit for pfas" in q and "is met" in q:
            score += 20
        if "confirm" in q and "pfas" in q:
            score += 10
        if "article 5" in q:
            score += 5
        score -= max(0, len(quote) - 200)
        return score

    for match in _PFAS_COMPLIANT_EVIDENCE_RE.finditer(markdown):
        quote = _quote_around(match)
        if (
            len(quote) >= MIN_EVIDENCE_CHARS
            and _PFAS_TOPIC_RE.search(quote)
            and evidence_in_source(quote, markdown)
        ):
            compliant_hits.append((_score_compliant(quote), quote))

    for match in _PFAS_NONCOMPLIANT_EVIDENCE_RE.finditer(markdown):
        quote = _quote_around(match)
        if (
            len(quote) >= MIN_EVIDENCE_CHARS
            and _PFAS_TOPIC_RE.search(quote)
            and evidence_in_source(quote, markdown)
        ):
            noncompliant_hits.append(quote)

    if compliant_hits:
        compliant_hits.sort(key=lambda x: x[0], reverse=True)
        best = compliant_hits[0][1]
        return SimpleNamespace(
            answer="no",
            evidence=best,
            concentration="N/A",
            notes="Recovered from source text (PFAS compliance statement).",
            source_document=source_file,
        )
    if noncompliant_hits:
        return SimpleNamespace(
            answer="yes",
            evidence=noncompliant_hits[0],
            concentration="N/A",
            notes="Recovered from source text.",
            source_document=source_file,
        )

    # Fallback: sentence chunks (legacy path for cleanly formatted text).
    candidates: List[Tuple[str, str]] = []
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", markdown):
        text = chunk.strip()
        if len(text) < MIN_EVIDENCE_CHARS or not _PFAS_TOPIC_RE.search(text):
            continue
        if not evidence_in_source(text[:300], markdown):
            continue
        if _PFAS_NONCOMPLIANT_EVIDENCE_RE.search(text):
            candidates.append(("yes", text[:300]))
        elif _PFAS_COMPLIANT_EVIDENCE_RE.search(text):
            candidates.append(("no", text[:300]))
    if not candidates:
        return None
    for answer, quote in candidates:
        if answer == "no":
            return SimpleNamespace(
                answer="no",
                evidence=quote,
                concentration="N/A",
                notes="Recovered from source text (PFAS compliance statement).",
                source_document=source_file,
            )
    answer, quote = candidates[0]
    return SimpleNamespace(
        answer=answer,
        evidence=quote,
        concentration="N/A",
        notes="Recovered from source text.",
        source_document=source_file,
    )


_DECLARANT_ACTION_RE = re.compile(
    r"(?P<subject>[A-Z][A-Za-z0-9&.'\-]+(?:\s+[A-Za-z0-9&.'\-]+){0,6}?)\s+has\s+conducted",
    re.IGNORECASE,
)

_SUPPLIER_SUFFIX_RE = re.compile(
    r"\b(?:spa|s\.p\.a|srl|s\.r\.l|gmbh|ltd|inc|co\.?\s*kg|a\.s\.?|"
    r"ve\s+tic(?:\.?\s*a\.s\.?)?|ambalaj|sanayi|san\.?|pharma|packaging)\b",
    re.IGNORECASE,
)


def _supplier_name_tokens(name: str) -> set:
    """Distinctive tokens from a supplier legal name."""
    text = (name or "").lower()
    text = _SUPPLIER_SUFFIX_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9]", " ", text)
    return {w for w in text.split() if len(w) >= 4}


def _subject_matches_supplier(subject: str, supplier_name: str) -> bool:
    subject_tokens = _supplier_name_tokens(subject)
    supplier_tokens = _supplier_name_tokens(supplier_name)
    if not subject_tokens or not supplier_tokens:
        return True
    for st in subject_tokens:
        for sp in supplier_tokens:
            if st == sp or st in sp or sp in st:
                return True
    return False


_PASS_THROUGH_DECLARATION_RE = re.compile(
    r"present\s+declaration\s+is\s+valid\s+for\s+material",
    re.IGNORECASE,
)


def _supplier_named_in_text(text: str, supplier_name: str) -> bool:
    tokens = _supplier_name_tokens(supplier_name)
    if not tokens:
        return False
    haystack = (text or "").lower()
    return any(token in haystack for token in tokens)


def evidence_declarant_mismatch(
    evidence: str,
    supplier_name: str,
    source_markdown: Optional[str] = None,
) -> bool:
    """
    True when evidence attributes analysis to another company without scoping to the audited supplier.

    Pass-through manufacturer declarations are allowed when the same document names the audited
    supplier (e.g. Belkofleks header + "Present declaration is valid for material…").
    """
    if not evidence or not supplier_name:
        return False
    match = _DECLARANT_ACTION_RE.search(evidence)
    if not match:
        return False
    if _subject_matches_supplier(match.group("subject"), supplier_name):
        return False

    context_parts = [evidence]
    if source_markdown:
        context_parts.append(source_markdown[:2500])
    context = "\n".join(context_parts)
    if (
        _supplier_named_in_text(context, supplier_name)
        and _PASS_THROUGH_DECLARATION_RE.search(context)
    ):
        return False
    return True


def enrich_pass_through_pfas_evidence(
    evidence: str,
    source_markdown: str,
    supplier_name: str,
) -> str:
    """Build a PFAS quote for manufacturer pass-through declarations scoped to a supplier."""
    if not source_markdown or not supplier_name:
        return evidence
    if not _PASS_THROUGH_DECLARATION_RE.search(source_markdown):
        return evidence
    if (
        evidence
        and evidence not in ("NONE", "")
        and evidence_indicates_compliant(CHECK_PFAS, evidence)
        and _supplier_named_in_text(evidence, supplier_name)
    ):
        return evidence

    pfas_match = _PFAS_COMPLIANT_EVIDENCE_RE.search(source_markdown)
    if not pfas_match:
        return evidence

    header_match = re.search(
        rf"({re.escape(supplier_name.split()[0])}[^\n]{{0,120}})",
        source_markdown,
        re.IGNORECASE,
    )
    if not header_match:
        tokens = sorted(_supplier_name_tokens(supplier_name), key=len, reverse=True)
        for token in tokens:
            header_match = re.search(
                rf"(\b{re.escape(token)}\b[^\n]{{0,160}})", source_markdown, re.I
            )
            if header_match:
                break
    if not header_match:
        return evidence

    pfas_end = min(len(source_markdown), pfas_match.end() + 80)
    ranked: List[Tuple[int, str]] = []

    for window in range(300, 199, -1):
        start = pfas_end - window
        if start < 0:
            continue
        candidate = re.sub(r"\s+", " ", source_markdown[start:pfas_end]).strip()
        if (
            _PFAS_COMPLIANT_EVIDENCE_RE.search(candidate)
            and evidence_in_source(candidate, source_markdown)
        ):
            score = 0
            if _supplier_named_in_text(candidate[:140], supplier_name):
                score += 20
            if _supplier_named_in_text(candidate, supplier_name):
                score += 10
            if _PASS_THROUGH_DECLARATION_RE.search(candidate):
                score += 8
            ranked.append((score, candidate))

    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_quote = ranked[0][1]
        if len(best_quote) >= MIN_EVIDENCE_CHARS:
            return best_quote[:300]

    pfas_start = max(0, pfas_match.start() - 60)
    fallback = re.sub(r"\s+", " ", source_markdown[pfas_start:pfas_end]).strip()
    if len(fallback) >= MIN_EVIDENCE_CHARS and evidence_in_source(fallback, source_markdown):
        return fallback[:300]
    return evidence


def is_regulatory_boilerplate_only(evidence: str) -> bool:
    """True when the quote looks like PPWR/legal text only, not a supplier material statement."""
    if not evidence or evidence.strip().upper() == "NONE":
        return False
    has_legal = bool(_LEGAL_BOILERPLATE_RE.search(evidence))
    has_material = bool(_MATERIAL_DECLARATION_RE.search(evidence))
    return has_legal and not has_material


def validate_point_finding(
    finding,
    source_markdown: str,
    check: Optional[str] = None,
    supplier_name: Optional[str] = None,
):
    """Downgrade to N/A when evidence is missing, off-topic, not in source, or boilerplate-only."""
    if finding.answer not in ("yes", "no"):
        return finding
    if finding.evidence in ("", "NONE"):
        finding.answer = "N/A"
        finding.concentration = "N/A"
        return finding
    if supplier_name and evidence_declarant_mismatch(
        finding.evidence, supplier_name, source_markdown
    ):
        _downgrade_finding(
            finding,
            "Evidence describes another company's analysis, not the audited supplier.",
        )
        return finding
    if check and not evidence_matches_check(finding.evidence, check):
        _downgrade_finding(
            finding,
            "Evidence does not address this specific PPWR check.",
        )
        return finding
    if not evidence_in_source(finding.evidence, source_markdown):
        _downgrade_finding(
            finding,
            "Evidence not found verbatim in source document.",
        )
        return finding
    if is_regulatory_boilerplate_only(finding.evidence):
        _downgrade_finding(
            finding,
            "Quote describes legal requirements only, not supplier material.",
        )
        return finding
    if check == CHECK_PFAS and supplier_name:
        finding.evidence = enrich_pass_through_pfas_evidence(
            finding.evidence, source_markdown, supplier_name
        )
    return correct_inverted_check_finding(finding, check) if check else finding


# Backward compatibility
def reject_shared_evidence(hm_finding, soc_finding):
    findings = {CHECK_HEAVY_METALS: hm_finding, CHECK_SOC: soc_finding}
    dedupe_findings(findings)
    return hm_finding, soc_finding
