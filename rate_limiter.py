"""
rate_limiter.py
================
Process-wide throttle for all Claude vision calls. Two layers:
  1. Semaphore caps concurrent in-flight requests.
  2. Sliding-window timer caps requests/minute.
Shared by faiss_store.py and document_extractors.py so PDF pages and
docx/pptx embedded images don't independently exceed your rate limit.
"""
from __future__ import annotations
import threading
import time
from collections import deque

import config


class RateLimiter:
    def __init__(self, max_concurrent: int, requests_per_minute: int):
        self._sem = threading.Semaphore(max_concurrent)
        self._rpm = requests_per_minute
        self._timestamps = deque()
        self._lock = threading.Lock()

    def acquire(self):
        self._sem.acquire()
        with self._lock:
            now = time.time()
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._rpm:
                wait = 60 - (now - self._timestamps[0]) + 0.1
                if wait > 0:
                    time.sleep(wait)
            self._timestamps.append(time.time())

    def release(self):
        self._sem.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()


vision_limiter = RateLimiter(
    max_concurrent=config.CLAUDE_MAX_CONCURRENT_VISION,
    requests_per_minute=config.CLAUDE_VISION_RPM,
)