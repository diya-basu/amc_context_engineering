"""
claude_vision_client.py
=========================
Shared Claude vision helper. Uses Haiku by default (config.CLAUDE_VISION_MODEL)
since extraction is transcription, not reasoning — Haiku is cheaper and has
materially higher throughput than Sonnet, which matters once you're running
this across hundreds of pages/images in a large corpus.

Every call routes through rate_limiter.vision_limiter.
"""
from __future__ import annotations
import base64
import time
from typing import Optional

import anthropic
import config
from rate_limiter import vision_limiter

_client: Optional[anthropic.Anthropic] = None
VISION_MAX_RETRIES = 4
VISION_RETRY_BASE_DELAY = 5


def _get_client() -> Optional[anthropic.Anthropic]:
    global _client
    if _client is None:
        if not config.CLAUDE_API_KEY:
            print("  [claude-vision] No CLAUDE_API_KEY set.", flush=True)
            return None
        _client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
    return _client


def extract_from_image(image_bytes: bytes, prompt: str,
                        model: str = None, max_tokens: int = 4096) -> str:
    client = _get_client()
    if client is None:
        return ""
    model = model or config.CLAUDE_VISION_MODEL
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    for attempt in range(VISION_MAX_RETRIES):
        with vision_limiter:
            try:
                resp = client.messages.create(
                    model=model, max_tokens=max_tokens, temperature=0.1,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/png", "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            except anthropic.RateLimitError:
                wait = VISION_RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    [claude-vision] rate-limited, waiting {wait}s…", flush=True)
                time.sleep(wait)
            except Exception as e:
                print(f"    [claude-vision] attempt {attempt+1} failed: {e}", flush=True)
                if attempt == VISION_MAX_RETRIES - 1:
                    return ""
                time.sleep(2.0)
    return ""