import importlib.util
import os
import sysconfig
import threading
from pathlib import Path
from typing import Any, Sequence


_ROOT_DIR = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve().with_name("split_fastpath.cpp")
_BUILD_DIR = _ROOT_DIR / "outputs" / "split_fastpath_native"
_MODULE_NAME = "gptsovits_split_fastpath"
_EXT_SUFFIX = str(sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if os.name == "nt" else ".so"))
_MODULE_PATH = _BUILD_DIR / f"{_MODULE_NAME}{_EXT_SUFFIX}"
_LOAD_LOCK = threading.Lock()
_LOAD_ATTEMPTED = False
_LOAD_ERROR: Exception | None = None
_MODULE: Any | None = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _build_module() -> Path:
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    extra_compile_args = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    extension = Extension(
        _MODULE_NAME,
        sources=[str(_SOURCE_PATH)],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
    distribution = Distribution({"name": _MODULE_NAME, "ext_modules": [extension]})
    cmd = build_ext(distribution)
    cmd.build_lib = str(_BUILD_DIR)
    cmd.build_temp = str(_BUILD_DIR / "temp")
    cmd.ensure_finalized()
    cmd.run()
    if not _MODULE_PATH.exists():
        raise FileNotFoundError(f"Missing built split fast path module: {_MODULE_PATH}")
    return _MODULE_PATH


def _ensure_built() -> Path:
    needs_build = not _MODULE_PATH.exists()
    if not needs_build:
        needs_build = _MODULE_PATH.stat().st_mtime < _SOURCE_PATH.stat().st_mtime
    if needs_build:
        return _build_module()
    return _MODULE_PATH


def _load_module():
    module_path = _ensure_built()
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load split fast path module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_module():
    global _LOAD_ATTEMPTED, _LOAD_ERROR, _MODULE
    if not _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_SPLIT_NATIVE", True):
        return None
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ATTEMPTED:
        return None
    with _LOAD_LOCK:
        if _MODULE is not None:
            return _MODULE
        if _LOAD_ATTEMPTED:
            return None
        _LOAD_ATTEMPTED = True
        try:
            _MODULE = _load_module()
        except Exception as exc:  # noqa: PERF203
            _LOAD_ERROR = exc
            _MODULE = None
        return _MODULE


def scan_selective_direct_runs(texts: Sequence[str]):
    module = _get_module()
    if module is None:
        return None
    try:
        return module.scan_selective_direct_runs(list(texts))
    except Exception as exc:  # noqa: PERF203
        global _LOAD_ERROR
        _LOAD_ERROR = exc
        return None


def get_last_error() -> Exception | None:
    return _LOAD_ERROR
