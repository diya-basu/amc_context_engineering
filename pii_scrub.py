"""
pii_scrub.py
============
Regex-based DPDP-oriented scrubber. Runs on page text BEFORE chunking so
no PII ever reaches embeddings, the LLM relation extractor, or Neo4j.
Deliberately simple (rapid prototype) — pattern-based, no ML NER for PII.
"""
from __future__ import annotations
import re

_PATTERNS = {
    "PAN":       re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "AADHAAR":   re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "MOBILE":    re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"),
    "EMAIL":     re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "BANK_ACC":  re.compile(r"\b\d{9,18}\b(?=.{0,15}(A/c|Account|IFSC))"),
    "IFSC":      re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
}

# ISIN (INF...) must survive — it's a legitimate fund identifier, not PII.
_ISIN_GUARD = re.compile(r"\bIN[EF][A-Z0-9]{9}\b")


def scrub_text(text: str) -> str:
    if not text:
        return text

    # protect ISINs from the generic BANK_ACC digit-run pattern by temporarily
    # tokenizing them out
    isins = _ISIN_GUARD.findall(text)
    placeholder_map = {}
    for i, isin in enumerate(isins):
        ph = f"__ISIN_{i}__"
        placeholder_map[ph] = isin
        text = text.replace(isin, ph)

    for label, pattern in _PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{label}]", text)

    for ph, isin in placeholder_map.items():
        text = text.replace(ph, isin)

    return text


def scrub_pages(pages_data: list[dict]) -> list[dict]:
    for p in pages_data:
        p["text"] = scrub_text(p["text"])
    return pages_data