"""
taxonomy_coverage_eval.py
============================
Proves the NER pipeline (Layer A+B) actually captures the taxonomies built
from AMFI/ and Sub Classification/. Runs NER over a sample doc, checks what
fraction of taxonomy entries mentioned in the text were actually detected.

Run: python taxonomy_coverage_eval.py --pdf path/to/sample.pdf
"""
from __future__ import annotations
import argparse
import re

import faiss_store
import ner_pipeline
import taxonomy as taxonomy_mod


def find_taxonomy_mentions(full_text: str, taxonomy: dict) -> dict:
    """Ground truth: which taxonomy entries literally appear in the text."""
    text_lower = full_text.lower()
    mentions = {}
    for key, values in taxonomy.items():
        found = [v for v in values if v and v.lower() in text_lower]
        mentions[key] = found
    return mentions


def run_coverage_eval(pdf_path: str):
    taxonomy = taxonomy_mod.load_taxonomy()
    pages_data = faiss_store.extract_pdf_text_full(pdf_path, verbose=False)
    full_text = "\n\n".join(p["text"] for p in pages_data)

    ground_truth = find_taxonomy_mentions(full_text, taxonomy)
    total_expected = sum(len(v) for v in ground_truth.values())
    print(f"Ground truth: {total_expected} taxonomy entries appear literally in the text.")
    for k, v in ground_truth.items():
        print(f"  {k}: {len(v)} mentions")

    parents, children = faiss_store.build_parent_child_chunks(pages_data, "eval")
    detected_texts = set()
    for child in children:
        ents = ner_pipeline.run_layers_ab(child["text"])
        detected_texts.update(e["text"].strip().lower() for e in ents)

    print("\nNER detection results:")
    total_hits = 0
    for key, expected_list in ground_truth.items():
        hits = [e for e in expected_list if e.lower() in detected_texts]
        total_hits += len(hits)
        rate = len(hits) / len(expected_list) if expected_list else float("nan")
        print(f"  {key}: {len(hits)}/{len(expected_list)} detected "
              f"({rate:.1%})" if expected_list else f"  {key}: no ground truth to compare")

    overall = total_hits / total_expected if total_expected else float("nan")
    print(f"\nOverall taxonomy coverage: {overall:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    args = ap.parse_args()
    run_coverage_eval(args.pdf)