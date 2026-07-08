"""
faiss_store.py  — v3.0
======================
Extraction pipeline (PRIMARY: Claude vision via Anthropic API, FALLBACK: local):

SMART EXTRACTION (default on):
  For each page, run the local 4-pass pipeline FIRST (fast, free). Only
  escalate to Claude vision if the page's local text is sparse, or the page
  has embedded images, or PyMuPDF detects a table — i.e. any signal that the
  local pass might have missed something. This is what keeps a large corpus
  rebuild from blowing through Claude's rate limits: most clean prose pages
  never touch the API at all.

  Escalation ladder per page:
    1. Local 4-pass pipeline (always runs first, free)
    2. If flagged -> Claude Haiku vision (cheap, high-throughput)
    3. If Haiku output is missing/too-short -> retry once on Claude Sonnet
    4. If both fail -> local pipeline output is used as-is (already have it)

  Set SMART_EXTRACTION=false in .env to force vision on every page instead
  (closer to the old "always call the vision model" behavior, higher cost).

FALLBACK (local 4-pass pipeline, unchanged from earlier versions):
  1. PyMuPDF  text blocks  →  prose paragraphs (fast, lossless)
  2. PyMuPDF  table finder →  markdown-formatted tables
  3. PyMuPDF  image list   →  EasyOCR on each embedded image
  4. Full-page OCR fallback when total chars < OCR_THRESHOLD

Chunking — SENTENCE-BOUNDARY AWARE:
  Parent : ~1200 chars  (what the LLM reads as context)
  Child  : ~250  chars  (what gets embedded and searched)
  Both splits respect sentence boundaries — no abrupt mid-sentence cuts.

Per-page extraction is cached to disk (logs/extraction_cache/) keyed on
file content hash, so an interrupted or re-run build never re-bills pages
that were already successfully extracted.

workers=0 in EasyOCR disables the DataLoader worker *processes* that
cause the macOS ARM (M1/M2/M3) segfault. OCR quality unaffected.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import config
from claude_vision_client import extract_from_image
from extraction_prompts import PDF_PAGE_PROMPT

_easyocr_reader = None
_embedder       = None

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEXES_DIR  = PROJECT_ROOT / "faiss_indexes"
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

USER_UPLOAD_SLUG = "user_uploads"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE    = 1200   # slightly larger to avoid cutting mid-table
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE     = 250
CHILD_CHUNK_OVERLAP  = 50

OCR_THRESHOLD = 150

# Claude vision extraction settings (see claude_vision_client.py for the
# actual API calls, retry/backoff, and shared rate limiter)
CLAUDE_PAGE_DPI = 150   # render resolution for vision calls (balance quality/tokens)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    name = Path(name).stem
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[\s-]+", "_", name)


def _file_hash(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:10]


# ─────────────────────────────────────────────────────────────────────────────
# LAZY LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedder():
    global _embedder
    if _embedder is None:
        print("  [embed] Loading sentence-transformer model…", flush=True)
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu",
        )
        print("  [embed] Model ready.", flush=True)
    return _embedder


def _get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        print("  [ocr] Loading EasyOCR…", flush=True)
        import easyocr
        _easyocr_reader = easyocr.Reader(["en", "hi"], gpu=False)
        print("  [ocr] EasyOCR ready.", flush=True)
    return _easyocr_reader


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY EXTRACTION — Claude vision per page (smart-gated)
# ─────────────────────────────────────────────────────────────────────────────

def _page_to_png_bytes(page, dpi: int = CLAUDE_PAGE_DPI) -> bytes:
    """Render a PyMuPDF page to PNG bytes at given DPI."""
    import fitz
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("png")


def _page_needs_vision(page, local_text: str) -> bool:
    """
    Decides whether a page needs Claude vision on top of the local pass.
    Deliberately conservative — any signal that local extraction might be
    incomplete triggers escalation, since a missed chart/table is worse
    than one extra API call.

    Escalates if:
      - local text is sparse (likely scanned or image-heavy page)
      - the page has ANY embedded image (could be a chart/benefit table)
      - PyMuPDF detects ANY table (verify it was captured cleanly)
    """
    if len(local_text.strip()) < config.LOCAL_TEXT_SUFFICIENCY_THRESHOLD:
        return True
    try:
        if len(page.get_images(full=True)) >= 1:
            return True
    except Exception:
        return True  # can't verify -> don't risk it, escalate
    try:
        tabs = page.find_tables()
        if tabs and tabs.tables:
            return True
    except Exception:
        pass
    return False


def extract_page_with_claude(page, page_num: int, verbose: bool = True) -> Optional[str]:
    """
    Send one page image to Claude. Tries Haiku first (cheap, fast); if the
    result is missing or suspiciously short, retries once on Sonnet before
    giving up — catches pages Haiku genuinely struggles with (dense tables,
    messy layouts) without paying Sonnet's cost on every page.
    """
    img_bytes = _page_to_png_bytes(page)

    text = extract_from_image(img_bytes, PDF_PAGE_PROMPT, model=config.CLAUDE_VISION_MODEL)
    if text and len(text.strip()) > 30:
        return text.strip()

    if verbose:
        print(f"    [claude-vision] page {page_num}: Haiku output too short, "
              f"retrying on {config.CLAUDE_MODEL_RELATIONS}…", flush=True)
    text = extract_from_image(img_bytes, PDF_PAGE_PROMPT, model=config.CLAUDE_MODEL_RELATIONS)
    if text and len(text.strip()) > 30:
        return text.strip()

    if verbose:
        print(f"    [claude-vision] page {page_num}: both models failed/too short.", flush=True)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK EXTRACTION — local 4-pass pipeline (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_prose(page) -> str:
    return page.get_text("text") or ""


def _extract_tables(page) -> str:
    try:
        tabs = page.find_tables()
    except AttributeError:
        return ""
    if not tabs or not tabs.tables:
        return ""
    md_blocks = []
    for table in tabs.tables:
        try:
            rows = table.extract()
            if not rows:
                continue
            cleaned = []
            for row in rows:
                cleaned.append([
                    str(cell).strip().replace("\n", " ") if cell else ""
                    for cell in row
                ])
            header    = cleaned[0]
            separator = ["---"] * len(header)
            body      = cleaned[1:]
            def _row(cells):
                return "| " + " | ".join(cells) + " |"
            lines = [_row(header), _row(separator)]
            lines += [_row(r) for r in body if any(c for c in r)]
            md_blocks.append("\n".join(lines))
        except Exception:
            continue
    return "\n\n".join(md_blocks)


def _extract_embedded_images_ocr(page) -> str:
    from PIL import Image
    import io
    doc        = page.parent
    image_list = page.get_images(full=True)
    if not image_list:
        return ""
    reader    = _get_ocr_reader()
    ocr_parts = []
    for img_info in image_list:
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            img_bytes  = base_image["image"]
            img        = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h       = img.size
            if w < 100 or h < 50:
                continue
            img_arr = np.array(img)
            results = reader.readtext(img_arr, detail=0, paragraph=True, workers=0)
            text    = "\n".join(results).strip()
            if text:
                ocr_parts.append(f"[Image OCR]\n{text}")
        except Exception:
            continue
    return "\n\n".join(ocr_parts)


def _full_page_ocr(page, dpi: int = 150) -> str:
    import fitz
    from PIL import Image
    import io
    mat      = fitz.Matrix(dpi / 72, dpi / 72)
    pix      = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("png")
    img      = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_arr  = np.array(img)
    reader   = _get_ocr_reader()
    results  = reader.readtext(img_arr, detail=0, paragraph=True, workers=0)
    return "\n".join(results)


def _extract_page_local(page, page_num: int, verbose: bool = True,
                         ocr_threshold: int = OCR_THRESHOLD) -> str:
    """Local 4-pass fallback/primary-first-pass extraction for one page."""
    parts:   List[str] = []
    methods: List[str] = []

    prose = _extract_prose(page).strip()
    if prose:
        parts.append(prose)
        methods.append("pymupdf")

    table_md = _extract_tables(page).strip()
    if table_md:
        parts.append(table_md)
        methods.append("table")

    try:
        img_text = _extract_embedded_images_ocr(page).strip()
        if img_text:
            parts.append(img_text)
            methods.append("image_ocr")
    except Exception as e:
        if verbose:
            print(f"    img-ocr page {page_num} skipped: {e}", flush=True)

    combined = "\n\n".join(parts)
    if len(combined) < ocr_threshold:
        if verbose:
            print(f"    full-page OCR page {page_num} ({len(combined)} chars)", flush=True)
        try:
            fp = _full_page_ocr(page).strip()
            if fp:
                parts.append(fp)
                methods.append("full_page_ocr")
                combined = "\n\n".join(parts)
        except Exception as e:
            if verbose:
                print(f"    full-page OCR page {page_num} failed: {e}", flush=True)

    return re.sub(r"\n{3,}", "\n\n", combined).strip()


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — local-first, Claude vision on demand, disk-cached per page
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text_full(
    pdf_path: str,
    ocr_threshold: int = OCR_THRESHOLD,
    verbose: bool = True,
    use_gemini: bool = None,   # deprecated alias — old callers still work
    use_vision: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract all text from a PDF.

    Smart mode (config.SMART_EXTRACTION, default True):
      local pass runs first on every page; Claude vision is only called on
      pages _page_needs_vision() flags as risky. Big reduction in API calls
      on text-heavy corpora, without unconditionally skipping vision.

    Non-smart mode (config.SMART_EXTRACTION=false):
      every page goes to Claude vision first, local pipeline as fallback —
      matches the old "always call the vision model" behavior.

    Returns list of page dicts:
      {page_num, text, source, extraction_method}
    """
    import fitz

    if use_gemini is not None:
        use_vision = use_gemini  # backward-compat: old call sites pass use_gemini=...

    pages_data  = []
    doc         = fitz.open(pdf_path)
    source_name = Path(pdf_path).name
    total_pages = doc.page_count

    # ── per-page disk cache, keyed on file content hash ─────────────────────
    pdf_hash   = _file_hash(pdf_path)
    cache_path = config.EXTRACTION_CACHE_DIR / f"{_slugify(pdf_path)}_{pdf_hash}.json"
    cache: Dict[str, Any] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def _save_cache():
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))

    if verbose:
        mode = "smart (local-first, vision on demand)" if config.SMART_EXTRACTION else "vision-first (every page)"
        print(f"  [extract] '{source_name}' ({total_pages} pages) — {mode}", flush=True)

    for page_num, page in enumerate(doc, start=1):
        cache_key = str(page_num)
        if cache_key in cache:
            pages_data.append(cache[cache_key])
            continue

        text   = ""
        method = "unknown"

        if config.SMART_EXTRACTION:
            # ── local pass first ─────────────────────────────────────────
            local_text = _extract_page_local(page, page_num, verbose=False,
                                              ocr_threshold=ocr_threshold)
            text, method = local_text, "local_fast"

            escalate = use_vision and _page_needs_vision(page, local_text)
            if escalate:
                claude_text = extract_page_with_claude(page, page_num, verbose=verbose)
                if claude_text and len(claude_text) > 50:
                    text, method = claude_text, "claude_vision"
                elif not text:
                    # local pass produced nothing AND vision failed — last resort
                    text = _extract_page_local(page, page_num, verbose=verbose,
                                                ocr_threshold=ocr_threshold)
                    method = "local_fallback"

        else:
            # ── old behavior: vision first, unconditionally ────────────────
            if use_vision:
                claude_text = extract_page_with_claude(page, page_num, verbose=verbose)
                if claude_text and len(claude_text) > 50:
                    text, method = claude_text, "claude_vision"
                else:
                    if verbose:
                        print(f"    [claude-vision] page {page_num}: falling back to local…", flush=True)
            if not text:
                text   = _extract_page_local(page, page_num, verbose=verbose,
                                              ocr_threshold=ocr_threshold)
                method = "local_fallback"

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            row = {
                "page_num":          page_num,
                "text":              text,
                "source":            source_name,
                "extraction_method": method,
            }
            pages_data.append(row)
            cache[cache_key] = row
            _save_cache()

        if verbose and page_num % 5 == 0:
            print(f"    … {page_num}/{total_pages} pages processed", flush=True)

    doc.close()
    return pages_data


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING — SENTENCE-BOUNDARY AWARE
# Fixes abrupt mid-sentence cuts seen in the UI generation prompts.
# ─────────────────────────────────────────────────────────────────────────────

def _split_at_sentence_boundary(text: str, target_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of approximately `target_size` characters.
    Splits ONLY at sentence boundaries (., !, ?, newline, table row end).
    Never cuts mid-sentence. Overlap is applied by repeating the last
    `overlap` chars of the previous chunk at the start of the next.

    Strategy:
      1. Split into sentences using regex.
      2. Greedily fill chunks up to target_size.
      3. When a chunk would exceed target_size, close it and start new one.
      4. Apply character-level overlap from the previous chunk's tail.
    """
    if len(text) <= target_size:
        return [text]

    # Sentence splitter: split after . ! ? \n  but keep the delimiter
    # Also treat markdown table rows (lines starting with |) as atomic units
    sentence_pattern = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F"\'(\[])'
        r'|\n(?=\n)'       # double newline = paragraph break
        r'|\n(?=\|)'       # line before a table row
        r'(?<=\|)\n',      # line after a table row
        re.UNICODE,
    )

    # Split into atomic units (sentences / table rows)
    units = sentence_pattern.split(text)
    # Remove empty units
    units = [u.strip() for u in units if u.strip()]

    chunks:   List[str] = []
    current:  List[str] = []
    cur_len = 0

    for unit in units:
        unit_len = len(unit)

        if cur_len + unit_len + 1 > target_size and current:
            # Close current chunk
            chunk_text = " ".join(current)
            chunks.append(chunk_text)

            # Overlap: take tail chars of closed chunk
            if overlap > 0:
                tail = chunk_text[-overlap:]
                # find first sentence boundary in tail
                m = re.search(r'(?<=[.!?\n])\s+', tail)
                overlap_text = tail[m.start():] if m else tail
                current  = [overlap_text.strip(), unit]
                cur_len  = len(overlap_text) + unit_len + 1
            else:
                current  = [unit]
                cur_len  = unit_len
        else:
            current.append(unit)
            cur_len += unit_len + 1

    if current:
        chunks.append(" ".join(current))

    # Safety: if any chunk is still way over target (e.g., a single huge table),
    # hard-split it without breaking table rows mid-line.
    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= target_size * 1.5:
            final_chunks.append(chunk)
        else:
            # Hard split on newlines to preserve table rows
            lines   = chunk.split("\n")
            partial: List[str] = []
            p_len   = 0
            for line in lines:
                if p_len + len(line) > target_size and partial:
                    final_chunks.append("\n".join(partial))
                    partial = [line]
                    p_len   = len(line)
                else:
                    partial.append(line)
                    p_len += len(line) + 1
            if partial:
                final_chunks.append("\n".join(partial))

    return [c for c in final_chunks if c.strip()]


def build_parent_child_chunks(
    pages_data: List[Dict[str, Any]],
    product_name: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build parent (context) and child (search) chunks from page data.
    Uses sentence-boundary-aware splitting to avoid abrupt cuts.
    """
    parents:  List[Dict] = []
    children: List[Dict] = []
    parent_id = child_id = 0

    for page in pages_data:
        page_text = page["text"]

        parent_texts = _split_at_sentence_boundary(
            page_text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)

        for pc_text in parent_texts:
            if not pc_text.strip():
                continue
            pid = f"P{parent_id:05d}"
            parents.append({
                "parent_id":    pid,
                "text":         pc_text,
                "page_num":     page["page_num"],
                "source":       page["source"],
                "product_name": product_name,
                "method":       page.get("extraction_method", ""),
            })

            child_texts = _split_at_sentence_boundary(
                pc_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)

            for cc_text in child_texts:
                if not cc_text.strip():
                    continue
                children.append({
                    "child_id":     f"C{child_id:05d}",
                    "parent_id":    pid,
                    "text":         cc_text,
                    "page_num":     page["page_num"],
                    "source":       page["source"],
                    "product_name": product_name,
                })
                child_id += 1
            parent_id += 1

    return parents, children


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def _embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model     = _get_embedder()
    all_vecs  = []
    total     = len(texts)
    n_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch     = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"    embed batch {batch_num}/{n_batches} …", end=" ", flush=True)
        vecs = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        all_vecs.append(vecs)
        gc.collect()
        print("ok", flush=True)

    return np.vstack(all_vecs).astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# FAISS INDEX BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index_for_pdf(
    pdf_path: str,
    force_rebuild: bool = False,
    verbose: bool = True,
    use_gemini: bool = None,   # deprecated alias — old callers still work
    use_vision: bool = True,
) -> str:
    import faiss

    if use_gemini is not None:
        use_vision = use_gemini

    slug      = _slugify(pdf_path)
    index_dir = INDEXES_DIR / slug
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / "index.faiss"
    pkl_path   = index_dir / "index.pkl"
    meta_path  = index_dir / "meta.json"

    if faiss_path.exists() and pkl_path.exists() and not force_rebuild:
        if verbose:
            print(f"  ↩ Cached — skipping '{slug}'.", flush=True)
        return slug

    product_name = Path(pdf_path).stem

    if verbose:
        print(f"  Extracting text (use_vision={use_vision})…", flush=True)

    pages_data  = extract_pdf_text_full(
        pdf_path, verbose=verbose, use_vision=use_vision)
    total_chars = sum(len(p["text"]) for p in pages_data)
    methods_used = set(p["extraction_method"] for p in pages_data)
    vision_pages = sum(1 for p in pages_data if p["extraction_method"] == "claude_vision")
    local_pages  = len(pages_data) - vision_pages

    if verbose:
        print(f"  {len(pages_data)} pages · {total_chars:,} chars", flush=True)
        print(f"  Methods: claude_vision={vision_pages} local={local_pages}", flush=True)

    parents, children = build_parent_child_chunks(pages_data, product_name)
    if verbose:
        print(f"  {len(parents)} parents · {len(children)} children", flush=True)

    child_texts = [c["text"] for c in children]
    if verbose:
        print(f"  Embedding {len(child_texts)} child chunks…", flush=True)
    vectors = _embed_texts(child_texts)

    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(faiss_path))
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "children": children,
            "parents":  {p["parent_id"]: p for p in parents},
        }, f)

    meta = {
        "slug":          slug,
        "product_name":  product_name,
        "source_file":   Path(pdf_path).name,
        "num_pages":     len(pages_data),
        "num_parents":   len(parents),
        "num_children":  len(children),
        "embed_dim":     dim,
        "methods_used":  list(methods_used),
        "claude_vision_pages": vision_pages,
        "local_pages":   local_pages,
        "model":         "paraphrase-multilingual-MiniLM-L12-v2",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if verbose:
        print(f"  Saved -> {index_dir}", flush=True)

    del vectors, parents, children, pages_data
    gc.collect()
    return slug


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

class BrochureFAISSStore:
    def __init__(self, slug: str):
        import faiss

        self.slug  = slug
        index_dir  = INDEXES_DIR / slug
        faiss_path = index_dir / "index.faiss"
        pkl_path   = index_dir / "index.pkl"
        meta_path  = index_dir / "meta.json"

        if not faiss_path.exists():
            raise FileNotFoundError(
                f"No FAISS index for '{slug}'. "
                f"Run scripts/build_all_indexes.py.")

        self.index = faiss.read_index(str(faiss_path))
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.children    = data["children"]
        self.parents_map = data["parents"]
        self.meta        = json.loads(meta_path.read_text())

    def retrieve(
        self,
        query: str,
        top_k_children: int = 6,
        dedupe_parents: bool = True,
    ) -> List[Dict]:
        model  = _get_embedder()
        q_vec  = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_vec, top_k_children)

        seen:    set        = set()
        results: List[Dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.children):
                continue
            child     = self.children[idx]
            parent_id = child["parent_id"]
            if dedupe_parents and parent_id in seen:
                continue
            seen.add(parent_id)
            parent = self.parents_map.get(parent_id, {})
            results.append({
                "parent_id":    parent_id,
                "parent_text":  parent.get("text", child["text"]),
                "child_text":   child["text"],
                "score":        float(score),
                "page_num":     child["page_num"],
                "product_name": child["product_name"],
                "source":       child["source"],
            })
        return results

    def get_context_string(self, query: str, top_k: int = 5) -> str:
        hits  = self.retrieve(query, top_k_children=top_k)
        parts = [
            f"[{h['product_name']} | Page {h['page_num']}]\n{h['parent_text']}"
            for h in hits
        ]
        return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_store_cache: Dict[str, BrochureFAISSStore] = {}


def list_available_products() -> List[Dict]:
    results = []
    for d in sorted(INDEXES_DIR.iterdir()):
        if d.is_dir() and d.name != USER_UPLOAD_SLUG:
            mp = d / "meta.json"
            if mp.exists():
                results.append(json.loads(mp.read_text()))
    return results


def get_store(slug: str) -> BrochureFAISSStore:
    if slug not in _store_cache:
        _store_cache[slug] = BrochureFAISSStore(slug)
    return _store_cache[slug]


def get_store_for_product(product_name: str) -> Optional[BrochureFAISSStore]:
    for meta in list_available_products():
        if meta["product_name"].lower() == product_name.lower():
            return get_store(meta["slug"])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# USER UPLOAD INDEX
# ─────────────────────────────────────────────────────────────────────────────

def build_user_upload_index(
    pdf_paths: List[str],
    append: bool = False,
    use_gemini: bool = None,   # deprecated alias — old callers still work
    use_vision: bool = True,
) -> BrochureFAISSStore:
    import faiss

    if use_gemini is not None:
        use_vision = use_gemini

    slug      = USER_UPLOAD_SLUG
    index_dir = INDEXES_DIR / slug
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / "index.faiss"
    pkl_path   = index_dir / "index.pkl"

    all_children: List[Dict] = []
    all_parents:  Dict       = {}

    if append and faiss_path.exists():
        existing_index = faiss.read_index(str(faiss_path))
        with open(pkl_path, "rb") as f:
            existing   = pickle.load(f)
        all_children   = existing["children"]
        all_parents    = existing["parents"]

    parent_offset = len(all_parents)
    child_offset  = len(all_children)
    new_children: List[Dict] = []

    for pdf_path in pdf_paths:
        product_name      = Path(pdf_path).stem
        pages_data        = extract_pdf_text_full(
            pdf_path, verbose=False, use_vision=use_vision)
        parents, children = build_parent_child_chunks(pages_data, product_name)
        id_map: Dict[str, str] = {}

        for p in parents:
            new_pid                = f"P{parent_offset:05d}"
            id_map[p["parent_id"]] = new_pid
            p["parent_id"]         = new_pid
            all_parents[new_pid]   = p
            parent_offset         += 1

        for c in children:
            c["child_id"]  = f"C{child_offset:05d}"
            c["parent_id"] = id_map[c["parent_id"]]
            all_children.append(c)
            new_children.append(c)
            child_offset  += 1

    if new_children:
        new_vecs = _embed_texts([c["text"] for c in new_children])
        dim      = new_vecs.shape[1]
        index    = (existing_index
                    if (append and faiss_path.exists())
                    else faiss.IndexFlatIP(dim))
        index.add(new_vecs)
        faiss.write_index(index, str(faiss_path))

    with open(pkl_path, "wb") as f:
        pickle.dump({"children": all_children, "parents": all_parents}, f)

    meta = {
        "slug":         slug,
        "product_name": "User Uploads",
        "source_file":  ", ".join(Path(p).name for p in pdf_paths),
        "num_parents":  len(all_parents),
        "num_children": len(all_children),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    _store_cache.pop(slug, None)
    return BrochureFAISSStore(slug)