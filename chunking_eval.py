"""
chunking_eval.py
==================
Proves parent/child + sentence-boundary chunking beats naive fixed-size
("semantic") chunking, on two axes:

  1. Boundary violations — how often a chunk cuts mid-sentence / mid-table-row.
  2. Retrieval quality — hit rate @k on a small hand-written query set,
     using the SAME embedder for both chunking strategies so it's apples-to-apples.

Run: python chunking_eval.py --pdf path/to/sample.pdf
"""
from __future__ import annotations
import argparse
import re
from typing import List

import numpy as np

import faiss_store


# ── naive baseline: fixed-size char splitter, no sentence awareness ──────
def naive_semantic_chunks(text: str, size: int = 250, overlap: int = 50) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return [c for c in chunks if c.strip()]


def count_boundary_violations(chunks: List[str]) -> int:
    """A violation = chunk does NOT end on sentence-ending punctuation,
    a newline, or a table-row pipe — i.e. it was cut mid-word/mid-sentence."""
    violations = 0
    for c in chunks:
        tail = c.rstrip()
        if not tail:
            continue
        if not re.search(r'[.!?\n]$|\|\s*$', tail):
            violations += 1
    return violations


def eval_retrieval(chunks: List[str], queries: List[str], relevant_substrings: List[str],
                    top_k: int = 3) -> float:
    """
    Very rough proxy: embed all chunks, for each query check whether the
    top_k retrieved chunks contain the expected substring. Returns hit rate.
    """
    if not chunks:
        return 0.0
    vecs = faiss_store._embed_texts(chunks)
    model = faiss_store._get_embedder()

    hits = 0
    for q, expected_sub in zip(queries, relevant_substrings):
        q_vec = model.encode([q], normalize_embeddings=True).astype("float32")
        sims = vecs @ q_vec[0]
        top_idx = np.argsort(-sims)[:top_k]
        found = any(expected_sub.lower() in chunks[i].lower() for i in top_idx)
        hits += int(found)
    return hits / len(queries) if queries else 0.0


def run_eval(pdf_path: str, queries: List[str], relevant_substrings: List[str]):
    print(f"Extracting: {pdf_path}", flush=True)
    pages_data = faiss_store.extract_pdf_text_full(pdf_path, verbose=False)
    full_text = "\n\n".join(p["text"] for p in pages_data)

    print("\n== Parent/Child sentence-boundary chunker (current) ==")
    parents, children = faiss_store.build_parent_child_chunks(pages_data, "eval")
    child_texts = [c["text"] for c in children]
    v1 = count_boundary_violations(child_texts)
    r1 = eval_retrieval(child_texts, queries, relevant_substrings)
    print(f"  chunks={len(child_texts)}  boundary_violations={v1}  "
          f"violation_rate={v1/max(1,len(child_texts)):.2%}  retrieval_hit_rate={r1:.2%}")

    print("\n== Naive fixed-size chunker (baseline) ==")
    naive_chunks = naive_semantic_chunks(full_text)
    v2 = count_boundary_violations(naive_chunks)
    r2 = eval_retrieval(naive_chunks, queries, relevant_substrings)
    print(f"  chunks={len(naive_chunks)}  boundary_violations={v2}  "
          f"violation_rate={v2/max(1,len(naive_chunks)):.2%}  retrieval_hit_rate={r2:.2%}")

    print("\n== Verdict ==")
    print(f"  Boundary violation rate: {'PASS' if v1/max(1,len(child_texts)) < v2/max(1,len(naive_chunks)) else 'FAIL'} "
          f"(parent/child lower is better)")
    print(f"  Retrieval hit rate:      {'PASS' if r1 >= r2 else 'FAIL'} (parent/child higher is better)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    args = ap.parse_args()

    # EDIT ME: a handful of queries + a substring you know exists in the
    # relevant page, for this specific PDF.
    demo_queries = [
        "What is the exit load for this scheme?",
        "Who is the fund manager?",
        "What is the benchmark index?",
    ]
    demo_substrings = ["exit load", "fund manager", "benchmark"]

    run_eval(args.pdf, demo_queries, demo_substrings)