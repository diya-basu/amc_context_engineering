"""
config.py — single source of truth for all env vars, paths, and constants.
Reuses the same Vertex/Gemini credential pattern as faiss_store.py.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PATHS ────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
DATA_DIR       = PROJECT_ROOT / "data" / "AMC"
AMFI_DIR       = PROJECT_ROOT / "data" / "AMFI"
SUBCLASS_DIR   = PROJECT_ROOT / "data" / "Sub Classification"
TAXONOMY_PATH  = PROJECT_ROOT / "taxonomy.json"
FAISS_DIR      = PROJECT_ROOT / "faiss_indexes"
LOG_DIR        = PROJECT_ROOT / "logs"
for d in (FAISS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── CLAUDE ────────────────────────────────────────────────────────────
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL_RELATIONS = os.environ.get("CLAUDE_MODEL_RELATIONS", "claude-sonnet-5")
CLAUDE_MODEL_LIGHT = os.environ.get("CLAUDE_MODEL_LIGHT", "claude-haiku-4-5-20251001")

# ── VERTEX / GEMINI (same pattern as faiss_store._get_gemini_client) ─────
VERTEX_KEY_PATH  = PROJECT_ROOT / "vertex_key.json"
GOOGLE_CLOUD_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_REGION   = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
VERTEX_MODEL_RELATIONS = os.environ.get("VERTEX_MODEL_RELATIONS", "gemini-2.5-pro")
VERTEX_MODEL_LIGHT     = os.environ.get("VERTEX_MODEL_LIGHT", "gemini-2.5-flash")
GEMINI_API_KEY          = os.environ.get("GEMINI_API_KEY", "")

# ── CHUNKING (reused verbatim from faiss_store.py so both indexes align) ─
PARENT_CHUNK_SIZE    = 1200
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE     = 250
CHILD_CHUNK_OVERLAP  = 50

# ── NER ────────────────────────────────────────────────────────────────
GLINER_MODEL_ID   = os.environ.get("GLINER_MODEL_ID", "urchade/gliner_medium-v2.1")
GLINER_LABELS     = [
    "mutual fund scheme name", "fund house", "benchmark index",
    "fund manager", "asset class", "sector",
]
GLINER_THRESHOLD  = 0.4

# ── NEO4J ─────────────────────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# ── PII / DPDP ─────────────────────────────────────────────────────────
PII_SCRUB_ENABLED = os.environ.get("PII_SCRUB_ENABLED", "true").lower() == "true"

# ── MISC ──────────────────────────────────────────────────────────────
SUPPORTED_EXTS = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".pptx", ".txt"}
VERBOSE = True