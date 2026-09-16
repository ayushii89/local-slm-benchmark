"""Phase 3: memory usage snapshots via Ollama's /api/ps (ollama.ps())."""

import threading

import ollama


def _base_name(name: str | None) -> str | None:
    """Strips the ":tag" suffix, e.g. "phi4-mini:latest" -> "phi4-mini"."""
    return name.split(":")[0] if name else name


def snapshot_model_memory(tag: str) -> dict:
    """Returns {"size_bytes": int, "size_vram_bytes": int} for the currently
    resident model matching `tag`. Call this right after a generate() call,
    before the model might get unloaded. Returns zeros if not found running.

    Matches by exact name/model first, then by base name (before ":") --
    Ollama's ps() reports models with an explicit tag (e.g. "phi4-mini:latest")
    even when ours has no tag (e.g. "phi4-mini"), so an exact-only match
    silently misses those and reports 0 memory.
    """
    try:
        running = ollama.ps()
    except Exception:
        return {"size_bytes": 0, "size_vram_bytes": 0}

    models = running.get("models", []) if isinstance(running, dict) else getattr(running, "models", [])
    tag_base = _base_name(tag)
    for m in models:
        m_name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
        m_model = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
        candidates = (m_name, m_model, _base_name(m_name), _base_name(m_model))
        if tag in candidates or tag_base in candidates:
            size = m.get("size") if isinstance(m, dict) else getattr(m, "size", 0)
            size_vram = m.get("size_vram") if isinstance(m, dict) else getattr(m, "size_vram", 0)
            return {"size_bytes": size or 0, "size_vram_bytes": size_vram or 0}

    return {"size_bytes": 0, "size_vram_bytes": 0}


class PeakMemoryTracker:
    """Polls snapshot_model_memory(tag) on a background thread while a `with`
    block runs, tracking the maximum size_bytes/size_vram_bytes seen. A single
    post-call snapshot can miss a mid-generation spike that drops back down
    before the call returns; polling catches that.
    """

    def __init__(self, tag: str, interval: float = 0.2):
        self.tag = tag
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = {"size_bytes": 0, "size_vram_bytes": 0}

    def _poll(self) -> None:
        while not self._stop.is_set():
            snap = snapshot_model_memory(self.tag)
            self._peak["size_bytes"] = max(self._peak["size_bytes"], snap["size_bytes"])
            self._peak["size_vram_bytes"] = max(self._peak["size_vram_bytes"], snap["size_vram_bytes"])
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakMemoryTracker":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        # One last snapshot in case the call finished faster than `interval`.
        snap = snapshot_model_memory(self.tag)
        self._peak["size_bytes"] = max(self._peak["size_bytes"], snap["size_bytes"])
        self._peak["size_vram_bytes"] = max(self._peak["size_vram_bytes"], snap["size_vram_bytes"])

    @property
    def peak(self) -> dict:
        return dict(self._peak)
