"""
build_index.py
================
Walks data/AMC/ (nested folders), and for every supported file:
  1. extract  (document_extractors / faiss_store)
  2. scrub PII
  3. parent/child chunk  (faiss_store.build_parent_child_chunks — NOT semantic)
  4. run NER layers A+B+C
  5. write entities+relations to Neo4j
  6. embed children + add to a single combined FAISS index

Run: python build_index.py [--rebuild] [--no-gemini]
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import config
import document_extractors
import graph_store
import ner_pipeline
import pii_scrub
import taxonomy as taxonomy_mod

import faiss_store  # reused: chunking + embedding + faiss index logic


def walk_amc_folder(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTS:
            yield path


def build(rebuild: bool = False, use_gemini: bool = True):
    import faiss

    print("== Step 0: taxonomy ==", flush=True)
    taxonomy_mod.build_taxonomy()

    print("== Step 1: Neo4j schema ==", flush=True)
    graph_store.init_schema()

    index_dir  = config.FAISS_DIR / "amc_master"
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = index_dir / "index.faiss"
    pkl_path   = index_dir / "index.pkl"

    all_children, all_parents = [], {}
    if not rebuild and faiss_path.exists() and pkl_path.exists():
        with open(pkl_path, "rb") as f:
            existing = pickle.load(f)
        all_children, all_parents = existing["children"], existing["parents"]
        print(f"  ↩ resuming — {len(all_children)} children already indexed", flush=True)

    already_done_sources = {c["source"] for c in all_children}
    index = faiss.read_index(str(faiss_path)) if faiss_path.exists() and not rebuild else None

    parent_offset = len(all_parents)
    child_offset  = len(all_children)
    new_vectors_batches = []

    for file_path in walk_amc_folder(config.DATA_DIR):
        if file_path.name in already_done_sources and not rebuild:
            print(f"  ↩ skip (already indexed): {file_path.name}", flush=True)
            continue

        print(f"\n== {file_path.relative_to(config.DATA_DIR)} ==", flush=True)
        product_name = file_path.stem

        try:
            pages_data = document_extractors.extract_any(str(file_path), use_gemini=use_gemini)
        except Exception as e:
            print(f"  !! extraction failed, skipping: {e}", flush=True)
            continue

        if config.PII_SCRUB_ENABLED:
            pages_data = pii_scrub.scrub_pages(pages_data)

        # non-semantic, sentence-boundary-aware chunking — reused as-is
        parents, children = faiss_store.build_parent_child_chunks(pages_data, product_name)

        # remap IDs into the master index's ID space
        id_map = {}
        for p in parents:
            new_pid = f"P{parent_offset:05d}"
            id_map[p["parent_id"]] = new_pid
            p["parent_id"] = new_pid
            all_parents[new_pid] = p
            parent_offset += 1
        for c in children:
            c["child_id"] = f"C{child_offset:05d}"
            c["parent_id"] = id_map[c["parent_id"]]
            all_children.append(c)
            child_offset += 1

        # NER — 3 layers
        parents_map = {p["parent_id"]: p for p in parents}
        ner_out = ner_pipeline.run_full_ner_for_chunk_set(children, parents_map)
        print(f"  [ner] {len(ner_out['entities'])} entities, "
              f"{len(ner_out['relations'])} relations", flush=True)

        graph_store.upsert_entities(ner_out["entities"], product_name, file_path.name)
        graph_store.upsert_relations(ner_out["relations"], product_name, file_path.name)

        # embed this file's children now (streaming, avoids OOM on big folders)
        vecs = faiss_store._embed_texts([c["text"] for c in children])
        if index is None:
            index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

    print("\n== Step N: resolving unresolved entities ==", flush=True)
    graph_store.resolve_unresolved_entities()

    if index is not None:
        faiss.write_index(index, str(faiss_path))
        with open(pkl_path, "wb") as f:
            pickle.dump({"children": all_children, "parents": all_parents}, f)
        (index_dir / "meta.json").write_text(json.dumps({
            "slug": "amc_master",
            "num_parents": len(all_parents),
            "num_children": len(all_children),
        }, indent=2))
        print(f"\nDone. {len(all_children)} children indexed -> {index_dir}", flush=True)
    else:
        print("\nNo new files found under data/AMC/.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="ignore cache, rebuild everything")
    ap.add_argument("--no-gemini", action="store_true", help="force local extraction fallback")
    args = ap.parse_args()
    build(rebuild=args.rebuild, use_gemini=not args.no_gemini)