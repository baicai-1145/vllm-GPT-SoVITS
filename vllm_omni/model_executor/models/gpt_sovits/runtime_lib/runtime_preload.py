import importlib
import os
from pathlib import Path

_RUNTIME_ENV_STATE: dict | None = None


def configure_model_runtime_environment() -> dict:
    global _RUNTIME_ENV_STATE
    if _RUNTIME_ENV_STATE is not None:
        return dict(_RUNTIME_ENV_STATE)

    cache_root = Path(os.environ.get("GPTSOVITS_RUNTIME_CACHE_ROOT", "/home/waas/gptsovits_cache")).expanduser()
    torchinductor_cache_dir = Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))
    ).expanduser()
    triton_cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", str(cache_root / "triton"))).expanduser()

    torchinductor_cache_dir.mkdir(parents=True, exist_ok=True)
    triton_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GPTSOVITS_RUNTIME_CACHE_ROOT", str(cache_root))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_cache_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE", "1")

    _RUNTIME_ENV_STATE = {
        "cache_root": str(cache_root),
        "torchinductor_cache_dir": str(torchinductor_cache_dir),
        "triton_cache_dir": str(triton_cache_dir),
        "fx_graph_cache": os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE", ""),
        "autotune_local_cache": os.environ.get("TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE", ""),
    }
    return dict(_RUNTIME_ENV_STATE)


def preload_text_runtime_deps() -> None:
    preload_modules = [
        "sqlite3",
        "nltk",
        "g2p_en",
    ]
    optional_modules = [
        "text.english",
        "text.japanese",
        "text.korean",
    ]
    for module_name in preload_modules:
        importlib.import_module(module_name)
    for module_name in optional_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
