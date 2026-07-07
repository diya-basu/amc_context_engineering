"""
llm_text_client.py
===================
Claude-only text client. Same call_llm()/call_llm_json() interface as
before, so retrieval.py, ner_pipeline.py etc. don't need to change how
they call this module — only this file's internals changed.
"""
from __future__ import annotations
import json
import re
import time
from typing import Optional

import anthropic
import config

_client: Optional[anthropic.Anthropic] = None

CLAUDE_MAX_RETRIES = 3
CLAUDE_RETRY_DELAY = 5


def _get_client() -> Optional[anthropic.Anthropic]:
    global _client
    if _client is None:
        if not config.CLAUDE_API_KEY:
            print("  [llm_text] No CLAUDE_API_KEY set in .env.", flush=True)
            return None
        _client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        print("  [llm_text] Claude client ready.", flush=True)
    return _client


def call_llm(prompt: str, model_id: str = None) -> str:
    client = _get_client()
    if client is None:
        return ""
    model = model_id or config.CLAUDE_MODEL_RELATIONS

    for attempt in range(CLAUDE_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except anthropic.RateLimitError:
            wait = CLAUDE_RETRY_DELAY * (2 ** attempt)
            print(f"  [llm_text] rate-limited, waiting {wait}s…", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  [llm_text] call failed (attempt {attempt+1}): {e}", flush=True)
            if attempt == CLAUDE_MAX_RETRIES - 1:
                return ""
            time.sleep(1.0)
    return ""


def call_llm_json(prompt: str, model_id: str = None) -> Optional[dict | list]:
    raw = call_llm(
        prompt + "\n\nRespond with ONLY valid JSON, no commentary, no markdown fences.",
        model_id=model_id)
    if not raw:
        return None
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None