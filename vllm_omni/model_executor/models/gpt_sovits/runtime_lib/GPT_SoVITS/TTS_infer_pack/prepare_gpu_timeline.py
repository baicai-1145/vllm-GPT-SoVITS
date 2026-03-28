import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict


_TRACE_PATH = os.environ.get("GPTSOVITS_PREPARE_GPU_TIMELINE_PATH", "").strip()
_TRACE_MAX_EVENTS = max(0, int(os.environ.get("GPTSOVITS_PREPARE_GPU_TIMELINE_MAX_EVENTS", "0") or 0))
_TRACE_ENABLED = bool(_TRACE_PATH)
_TRACE_LOCK = threading.Lock()
_TRACE_FILE = None
_TRACE_START = time.perf_counter()
_TRACE_EVENT_COUNT = 0


def _relative_ms(ts: float) -> float:
    return max(0.0, (float(ts) - float(_TRACE_START)) * 1000.0)


def _ensure_trace_file():
    global _TRACE_FILE
    if not _TRACE_ENABLED:
        return None
    if _TRACE_FILE is not None:
        return _TRACE_FILE
    path = Path(_TRACE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_FILE = path.open("a", encoding="utf-8")
    return _TRACE_FILE


def is_gpu_timeline_enabled() -> bool:
    return bool(_TRACE_ENABLED)


def sync_timeline_cuda(device: Any) -> None:
    if not _TRACE_ENABLED:
        return
    try:
        device_str = str(device)
        if device_str.startswith("cuda"):
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
    except Exception:
        pass


def _close_trace_file() -> None:
    global _TRACE_FILE
    with _TRACE_LOCK:
        if _TRACE_FILE is not None:
            try:
                _TRACE_FILE.flush()
                _TRACE_FILE.close()
            except Exception:
                pass
            _TRACE_FILE = None


atexit.register(_close_trace_file)


def trace_gpu_batch(kind: str, **payload: Any) -> None:
    global _TRACE_EVENT_COUNT
    if not _TRACE_ENABLED:
        return
    with _TRACE_LOCK:
        if _TRACE_MAX_EVENTS > 0 and _TRACE_EVENT_COUNT >= _TRACE_MAX_EVENTS:
            return
        trace_file = _ensure_trace_file()
        if trace_file is None:
            return
        event: Dict[str, Any] = {
            "kind": str(kind),
            "pid": int(os.getpid()),
            "thread": threading.current_thread().name,
            "event_index": int(_TRACE_EVENT_COUNT),
        }
        for key, value in payload.items():
            if key.endswith("_ts"):
                event[key[:-3] + "_ms"] = _relative_ms(float(value))
            else:
                event[key] = value
        trace_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        trace_file.flush()
        _TRACE_EVENT_COUNT += 1
