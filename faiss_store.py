from __future__ import annotations

import base64
import gc
import json
import os
import pickle
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

# Gemini extraction settings
GEMINI_PAGE_DPI      = 150   # render resolution for Gemini (balance quality/tokens)
GEMINI_RETRY_DELAY   = 5     # seconds between retries on rate-limit
GEMINI_MAX_RETRIES   = 3


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI EXTRACTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────
_GEMINI_PAGE_PROMPT = """You are extracting ALL content from a page of an SBI Life insurance product brochure.

Your output will be used to build a semantic search index. Extract EVERYTHING on the page.

EXTRACTION RULES:
1. PROSE TEXT: Extract all body text in natural reading order (top-to-bottom, left-to-right).
   Preserve paragraph breaks with a blank line. Do NOT omit any sentence.

2. TABLES: Convert every table to GitHub-flavoured markdown (| col1 | col2 | format).
   Include ALL rows and ALL columns. Never truncate a table.
   If a table spans the full page, that's fine — output the complete table.

3. IMAGES / CHARTS / ILLUSTRATIONS:
   - For benefit illustration tables embedded as images: extract every number, label, and column.
   - For graphs: describe axes, all data points, and any legends.
   - For diagrams (e.g., policy flow diagrams): describe each step/box and arrows.
   - For icons with labels: output "Icon: <label>".

4. HEADERS / FOOTERS: Include page headers/footers verbatim (they often contain product name,
   UIN number, disclaimer text). Mark them as [HEADER] and [FOOTER].

5. BOLD / HIGHLIGHTED TEXT: Preserve emphasis by wrapping in **double asterisks**.

6. NUMBERS AND PERCENTAGES: Never paraphrase or round. Output exact values as shown.

7. DISCLAIMERS / FINE PRINT: Extract completely, even if very small font.

OUTPUT FORMAT:
- Plain text with markdown tables.
- Section the output by content type if helpful (e.g., start tables with a blank line).
- Do NOT add commentary, summaries, or your own words.
- Do NOT say "This page contains..." or "I can see...".
- Output ONLY the extracted content.

Extract everything from the insurance brochure page shown:"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    name = Path(name).stem
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[\s-]+", "_", name)


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
# GEMINI CLIENT — tries Vertex AI first, then Gemini Direct API
# ─────────────────────────────────────────────────────────────────────────────

def _get_gemini_client():
    """
    Returns a callable: fn(image_bytes: bytes) -> str
    Tries Vertex AI (google-cloud-aiplatform) first.
    Falls back to google-generativeai (Gemini Direct API).
    Returns None if neither is available.
    """
    # ── Try Vertex AI ─────────────────────────────────────────────────────────
    _vertex_key_path = PROJECT_ROOT / "vertex_key.json"
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and _vertex_key_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_vertex_key_path)

    # Auto-extract project_id from key file if env var not set
    if not os.environ.get("GOOGLE_CLOUD_PROJECT") and _vertex_key_path.exists():
        try:
            import json as _json
            _key_data = _json.loads(_vertex_key_path.read_text())
            os.environ["GOOGLE_CLOUD_PROJECT"] = _key_data.get("project_id", "")
        except Exception:
            pass

    vertex_project  = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    vertex_location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    vertex_creds    = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    vertex_model    = os.environ.get("VERTEX_MODEL_ID", "gemini-2.5-pro")

    if vertex_project and (vertex_creds or os.path.exists("./vertex_key.json")):
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, Part, Image as VImage

            if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and \
               os.path.exists("./vertex_key.json"):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./vertex_key.json"

            vertexai.init(project=vertex_project, location=vertex_location)
            model = GenerativeModel(vertex_model)

            def vertex_fn(image_bytes: bytes, prompt: str = _GEMINI_PAGE_PROMPT) -> str:
                img_part = Part.from_data(data=image_bytes, mime_type="image/png")
                response = model.generate_content(
                    [img_part, prompt],
                    generation_config={"temperature": 0.1, "max_output_tokens": 8192},
                )
                return response.text or ""

            print(f"  [gemini] Using Vertex AI ({vertex_project} / {vertex_model})", flush=True)
            return vertex_fn

        except Exception as e:
            print(f"  [gemini] Vertex AI init failed: {e} — trying Gemini Direct…", flush=True)

    # ── Try Gemini Direct API ─────────────────────────────────────────────────
    gemini_key   = os.environ.get("GEMINI_API_KEY", "")
    gemini_model = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-pro")

    if gemini_key:
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(gemini_model)

            def gemini_direct_fn(image_bytes: bytes, prompt: str = _GEMINI_PAGE_PROMPT) -> str:
                import PIL.Image
                import io
                img = PIL.Image.open(io.BytesIO(image_bytes))
                response = model.generate_content(
                    [prompt, img],
                    generation_config={"temperature": 0.1, "max_output_tokens": 8192},
                )
                return response.text or ""

            print(f"  [gemini] Using Gemini Direct API ({gemini_model})", flush=True)
            return gemini_direct_fn

        except Exception as e:
            print(f"  [gemini] Gemini Direct init failed: {e}", flush=True)

    print("  [gemini] No Gemini client available — will use local fallback.", flush=True)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY EXTRACTION — Gemini multimodal per page
# ─────────────────────────────────────────────────────────────────────────────

def _page_to_png_bytes(page, dpi: int = GEMINI_PAGE_DPI) -> bytes:
    """Render a PyMuPDF page to PNG bytes at given DPI."""
    import fitz
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("png")


def extract_page_with_gemini(
    page,
    gemini_fn,
    page_num: int,
    verbose: bool = True,
) -> Optional[str]:
    """
    Send one page image to Gemini. Returns extracted text or None on failure.
    Includes retry logic for rate-limit errors (429).
    """
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            img_bytes = _page_to_png_bytes(page)
            text      = gemini_fn(img_bytes, _GEMINI_PAGE_PROMPT)
            text      = text.strip()
            if len(text) > 30:   # sanity: at least some content
                return text
            if verbose:
                print(f"    [gemini] page {page_num}: suspiciously short ({len(text)} chars), "
                      f"retrying…", flush=True)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource exhausted" in err_str:
                wait = GEMINI_RETRY_DELAY * (2 ** attempt)
                if verbose:
                    print(f"    [gemini] page {page_num}: rate-limited, "
                          f"waiting {wait}s…", flush=True)
                time.sleep(wait)
            else:
                if verbose:
                    print(f"    [gemini] page {page_num} attempt {attempt+1} failed: {e}",
                          flush=True)
                if attempt == GEMINI_MAX_RETRIES - 1:
                    return None
        time.sleep(1.0)   # polite inter-call gap
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK EXTRACTION — local 4-pass pipeline (unchanged from v1.3)
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
    """Local 4-pass fallback extraction for one page."""
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
# ORCHESTRATOR — Gemini primary, local fallback
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text_full(
    pdf_path: str,
    ocr_threshold: int = OCR_THRESHOLD,
    verbose: bool = True,
    use_gemini: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract all text from a PDF.
    Primary: Gemini-2.5-pro multimodal (if configured).
    Fallback: local 4-pass pipeline.

    Returns list of page dicts:
      {page_num, text, source, extraction_method}
    """
    import fitz

    pages_data  = []
    doc         = fitz.open(pdf_path)
    source_name = Path(pdf_path).name
    total_pages = doc.page_count

    # Try to get Gemini client
    gemini_fn = _get_gemini_client() if use_gemini else None

    if gemini_fn:
        print(f"  [extract] Using Gemini multimodal for '{source_name}' "
              f"({total_pages} pages)…", flush=True)
    else:
        print(f"  [extract] Using local pipeline for '{source_name}' "
              f"({total_pages} pages)…", flush=True)

    for page_num, page in enumerate(doc, start=1):
        text   = ""
        method = "unknown"

        # ── Primary: Gemini ───────────────────────────────────────────────────
        if gemini_fn:
            gemini_text = extract_page_with_gemini(
                page, gemini_fn, page_num, verbose=verbose)
            if gemini_text and len(gemini_text) > 50:
                text   = gemini_text
                method = "gemini"
            else:
                if verbose:
                    print(f"    [gemini] page {page_num}: falling back to local…", flush=True)

        # ── Fallback: local pipeline ──────────────────────────────────────────
        if not text:
            text   = _extract_page_local(page, page_num, verbose=verbose,
                                          ocr_threshold=ocr_threshold)
            method = "local_fallback"

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages_data.append({
                "page_num":          page_num,
                "text":              text,
                "source":            source_name,
                "extraction_method": method,
            })

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
    use_gemini: bool = True,
) -> str:
    import faiss

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
        print(f"  Extracting text (use_gemini={use_gemini})…", flush=True)

    pages_data  = extract_pdf_text_full(
        pdf_path, verbose=verbose, use_gemini=use_gemini)
    total_chars = sum(len(p["text"]) for p in pages_data)
    methods_used = set(p["extraction_method"] for p in pages_data)
    gemini_pages = sum(1 for p in pages_data if p["extraction_method"] == "gemini")
    local_pages  = len(pages_data) - gemini_pages

    if verbose:
        print(f"  {len(pages_data)} pages · {total_chars:,} chars", flush=True)
        print(f"  Methods: gemini={gemini_pages} local_fallback={local_pages}", flush=True)

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
        "gemini_pages":  gemini_pages,
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
    use_gemini: bool = True,
) -> BrochureFAISSStore:
    import faiss

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
            pdf_path, verbose=False, use_gemini=use_gemini)
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