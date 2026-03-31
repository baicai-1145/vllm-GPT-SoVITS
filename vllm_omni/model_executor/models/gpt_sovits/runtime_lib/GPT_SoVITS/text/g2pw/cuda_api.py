import ctypes
import fcntl
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

from TTS_infer_pack.prepare_gpu_timeline import trace_gpu_batch

from .backend_common import _G2PWBaseOnnxConverter


class G2PWCudaError(RuntimeError):
    pass


@dataclass
class G2PWBatchTask:
    model_input: Dict[str, np.ndarray]
    created_at: float = field(default_factory=time.perf_counter)
    enqueued_at: float = 0.0
    done_event: threading.Event = field(default_factory=threading.Event)
    output: np.ndarray | None = None
    profile: Dict[str, float] = field(default_factory=dict)
    error: Exception | None = None


_ROOT_DIR = Path(__file__).resolve().parents[3]
_PACKAGE_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _ROOT_DIR / "outputs" / "g2pw_cuda_bridge"
_WRAPPER_SOURCE = _PACKAGE_DIR / "g2pw_cuda_bridge.cpp"
_LOCK_PATH = _OUTPUT_DIR / "build.lock"


def _env_flag(name: str, default: bool) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return 1 if default else 0
    return 0 if raw.strip().lower() in {"0", "false", "no", "off"} else 1


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)


def _resolve_cuda_root() -> Path:
    env_root = os.environ.get("GPTSOVITS_G2PW_CUDA_ROOT", "").strip()
    candidates = [
        env_root,
        _ROOT_DIR / "third_party" / "g2pw-cu",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    checked = [
        str(Path(candidate).expanduser().resolve())
        for candidate in candidates
        if str(candidate).strip() != ""
    ]
    raise G2PWCudaError(
        "Cannot locate g2pw-cu root. "
        "Expected one of: "
        f"{checked}. "
        "Recommended: clone https://github.com/baicai-1145/g2pw-cu.git into "
        f"{(_ROOT_DIR / 'third_party' / 'g2pw-cu').as_posix()} "
        "or set GPTSOVITS_G2PW_CUDA_ROOT explicitly."
    )


def _resolve_runtime_paths() -> tuple[Path, Path, Path]:
    cuda_root = _resolve_cuda_root()
    runtime_lib = Path(
        os.environ.get("GPTSOVITS_G2PW_CUDA_RUNTIME_LIB", str(cuda_root / "build" / "libg2pw_runtime.so"))
    ).expanduser()
    manifest_path = Path(
        os.environ.get("GPTSOVITS_G2PW_CUDA_MANIFEST", str(cuda_root / "artifacts" / "model" / "manifest.txt"))
    ).expanduser()
    weights_path = Path(
        os.environ.get("GPTSOVITS_G2PW_CUDA_WEIGHTS", str(cuda_root / "artifacts" / "model" / "weights.bin"))
    ).expanduser()
    for path in (runtime_lib, manifest_path, weights_path):
        if not path.exists():
            raise G2PWCudaError(f"Missing g2pw-cu artifact: {path}")
    return runtime_lib.resolve(), manifest_path.resolve(), weights_path.resolve()


def _build_bridge(wrapper_output: Path, runtime_lib: Path) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compile_cmd = [
        os.environ.get("CXX", "g++"),
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(_WRAPPER_SOURCE),
        "-I",
        str(runtime_lib.parent.parent / "include"),
        "-L",
        str(runtime_lib.parent),
        "-lg2pw_runtime",
        f"-Wl,-rpath,{runtime_lib.parent}",
        "-o",
        str(wrapper_output),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise G2PWCudaError(
            "Failed to build g2pw-cu bridge:\n"
            f"cmd={' '.join(compile_cmd)}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )


def _ensure_bridge_built(runtime_lib: Path) -> Path:
    wrapper_output = _OUTPUT_DIR / "g2pw_cuda_bridge.so"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK_PATH.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        needs_build = not wrapper_output.exists()
        if not needs_build:
            so_mtime = wrapper_output.stat().st_mtime
            needs_build = so_mtime < _WRAPPER_SOURCE.stat().st_mtime or so_mtime < runtime_lib.stat().st_mtime
        if needs_build:
            tmp_output = wrapper_output.with_suffix(".tmp.so")
            if tmp_output.exists():
                tmp_output.unlink()
            _build_bridge(tmp_output, runtime_lib)
            tmp_output.replace(wrapper_output)
    return wrapper_output


def _load_bridge():
    runtime_lib, manifest_path, weights_path = _resolve_runtime_paths()
    bridge_path = _ensure_bridge_built(runtime_lib)
    global_mode = getattr(ctypes, "RTLD_GLOBAL", getattr(os, "RTLD_GLOBAL", 0))
    ctypes.CDLL(str(runtime_lib), mode=global_mode)
    lib = ctypes.CDLL(str(bridge_path))
    lib.g2pw_runtime_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.g2pw_runtime_create.restype = ctypes.c_void_p
    lib.g2pw_runtime_destroy.argtypes = [ctypes.c_void_p]
    lib.g2pw_runtime_destroy.restype = None
    lib.g2pw_runtime_last_error.argtypes = [ctypes.c_void_p]
    lib.g2pw_runtime_last_error.restype = ctypes.c_char_p
    lib.g2pw_runtime_num_labels.argtypes = [ctypes.c_void_p]
    lib.g2pw_runtime_num_labels.restype = ctypes.c_int
    lib.g2pw_runtime_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.g2pw_runtime_run.restype = ctypes.c_int
    return lib, manifest_path, weights_path, runtime_lib


def _gemm_precision_value() -> int:
    precision = os.environ.get("GPTSOVITS_G2PW_CUDA_GEMM_PRECISION", "fp32").strip().lower()
    if precision == "fp16":
        return 1
    if precision == "bf16":
        return 2
    return 0


class G2PWRuntimeWrapper:
    def __init__(self, shard_index: int = 0) -> None:
        self.lib, self.manifest_path, self.weights_path, self.runtime_lib = _load_bridge()
        self.shard_index = int(shard_index)
        self.device_ordinal = _env_int("GPTSOVITS_G2PW_CUDA_DEVICE", 0)
        self.allow_tensor_cores = _env_flag("GPTSOVITS_G2PW_CUDA_ALLOW_TENSOR_CORES", False)
        self.use_cublaslt_bias_epilogue = _env_flag("GPTSOVITS_G2PW_CUDA_USE_CUBLASLT_BIAS_EPILOGUE", False)
        self.enable_profiling = _env_flag("GPTSOVITS_G2PW_CUDA_ENABLE_PROFILE", False)
        # Default off: prepare-only at high concurrency is not stable with CUDA graph capture.
        self.enable_cuda_graph = _env_flag("GPTSOVITS_G2PW_CUDA_ENABLE_GRAPH", False)
        self.dump_graph_cache_stats = _env_flag("GPTSOVITS_G2PW_CUDA_DUMP_GRAPH_CACHE_STATS", False)
        self.full_graph_cache_limit = _env_int("GPTSOVITS_G2PW_CUDA_FULL_GRAPH_CACHE_LIMIT", 0)
        self.tail_graph_cache_limit = _env_int("GPTSOVITS_G2PW_CUDA_TAIL_GRAPH_CACHE_LIMIT", 0)
        self.gemm_precision = _gemm_precision_value()
        self.lock = threading.Lock()
        self.handle = None
        self.max_batch_size = 0
        self.max_seq_len = 0
        self.num_labels = 0
        self.batch_enabled = _env_flag("GPTSOVITS_G2PW_CUDA_BATCHING", True) != 0
        self.batch_window_s = max(0.0, float(_env_int("GPTSOVITS_G2PW_CUDA_BATCH_WINDOW_MS", 1)) / 1000.0)
        self.batch_max_requests = max(1, _env_int("GPTSOVITS_G2PW_CUDA_BATCH_MAX_REQUESTS", 64))
        self.batch_max_rows = max(1, _env_int("GPTSOVITS_G2PW_CUDA_BATCH_MAX_ROWS", 96))
        self.batch_max_tokens = max(1, _env_int("GPTSOVITS_G2PW_CUDA_BATCH_MAX_TOKENS", 4096))
        self.batch_condition = threading.Condition()
        self.pending_tasks: Deque[G2PWBatchTask] = deque()
        self.batch_total_tasks = 0
        self.batch_total_batches = 0
        self.batch_total_rows = 0
        self.batch_total_queue_wait_ms = 0.0
        self.batch_queue_wait_peak_ms = 0.0
        self.batch_total_collect_wait_ms = 0.0
        self.batch_collect_wait_peak_ms = 0.0
        self.batch_total_run_ms = 0.0
        self.batch_run_peak_ms = 0.0
        self.batch_rows_peak = 0
        self.batch_requests_peak = 0
        self.batch_pending_peak = 0
        self.closed = False
        self._ensure_capacity(
            batch_size=max(1, _env_int("GPTSOVITS_G2PW_CUDA_MAX_BATCH_SIZE", 256)),
            seq_len=max(1, _env_int("GPTSOVITS_G2PW_CUDA_MAX_SEQ_LEN", 128)),
        )
        self.direct_max_rows = max(
            1,
            _env_int("GPTSOVITS_G2PW_CUDA_DIRECT_MAX_ROWS", min(int(self.max_batch_size), int(self.batch_max_rows))),
        )
        self.direct_max_sequences = max(
            1,
            _env_int(
                "GPTSOVITS_G2PW_CUDA_DIRECT_MAX_SEQUENCES",
                min(int(self.max_batch_size), int(self.batch_max_rows)),
            ),
        )
        self.direct_max_tokens = max(
            1,
            _env_int("GPTSOVITS_G2PW_CUDA_DIRECT_MAX_TOKENS", max(int(self.batch_max_tokens), int(self.max_seq_len))),
        )
        self.batch_worker = None
        if self.batch_enabled:
            self.batch_worker = threading.Thread(
                target=self._batch_loop,
                name=f"g2pw-cuda-batch-worker-{self.shard_index}",
                daemon=True,
            )
            self.batch_worker.start()

    def _sync_runtime_env_overrides(self) -> None:
        os.environ["G2PW_ENABLE_CUDA_GRAPH"] = "1" if self.enable_cuda_graph else "0"
        os.environ["G2PW_ENABLE_PROFILE"] = "1" if self.enable_profiling else "0"
        os.environ["G2PW_DUMP_GRAPH_CACHE_STATS"] = "1" if self.dump_graph_cache_stats else "0"
        os.environ["G2PW_FULL_GRAPH_CACHE_LIMIT"] = str(int(self.full_graph_cache_limit))
        os.environ["G2PW_TAIL_GRAPH_CACHE_LIMIT"] = str(int(self.tail_graph_cache_limit))
        os.environ["G2PW_ALLOW_TENSOR_CORES"] = "1" if self.allow_tensor_cores else "0"
        os.environ["G2PW_USE_CUBLASLT_BIAS_EPILOGUE"] = "1" if self.use_cublaslt_bias_epilogue else "0"
        os.environ["G2PW_GEMM_PRECISION"] = {0: "fp32", 1: "fp16", 2: "bf16"}.get(int(self.gemm_precision), "fp32")

    def _destroy_handle(self) -> None:
        if self.handle:
            self.lib.g2pw_runtime_destroy(self.handle)
            self.handle = None

    def close(self) -> None:
        with self.batch_condition:
            self.closed = True
            self.batch_condition.notify_all()
        self._destroy_handle()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _last_error(self) -> str:
        if not self.handle:
            return "uninitialized runtime"
        message = self.lib.g2pw_runtime_last_error(self.handle)
        return "" if not message else message.decode("utf-8", errors="replace")

    def _create_handle(self, batch_size: int, seq_len: int) -> None:
        self._sync_runtime_env_overrides()
        new_handle = self.lib.g2pw_runtime_create(
            str(self.manifest_path).encode("utf-8"),
            str(self.weights_path).encode("utf-8"),
            int(self.device_ordinal),
            int(batch_size),
            int(seq_len),
            int(self.full_graph_cache_limit),
            int(self.tail_graph_cache_limit),
            int(self.allow_tensor_cores),
            int(self.use_cublaslt_bias_epilogue),
            int(self.enable_profiling),
            int(self.enable_cuda_graph),
            int(self.dump_graph_cache_stats),
            int(self.gemm_precision),
        )
        if not new_handle:
            raise G2PWCudaError("g2pw-cu returned null runtime handle")
        self.handle = new_handle
        self.max_batch_size = int(batch_size)
        self.max_seq_len = int(seq_len)
        self.num_labels = int(self.lib.g2pw_runtime_num_labels(self.handle))
        last_error = self._last_error()
        if self.num_labels <= 0 or last_error:
            self.close()
            raise G2PWCudaError(f"Failed to initialize g2pw-cu runtime: {last_error or 'num_labels <= 0'}")

    def _ensure_capacity(self, batch_size: int, seq_len: int) -> None:
        target_batch = max(1, int(batch_size))
        target_seq = max(1, int(seq_len))
        if self.handle and target_batch <= self.max_batch_size and target_seq <= self.max_seq_len:
            return
        next_batch = max(target_batch, self.max_batch_size * 2 if self.max_batch_size else 0)
        next_seq = max(target_seq, self.max_seq_len * 2 if self.max_seq_len else 0)
        self._destroy_handle()
        self._create_handle(batch_size=next_batch, seq_len=next_seq)

    @staticmethod
    def _normalize_model_input(model_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        input_ids = np.ascontiguousarray(model_input["input_ids"], dtype=np.int64)
        token_type_ids = np.ascontiguousarray(model_input["token_type_ids"], dtype=np.int64)
        attention_masks = np.ascontiguousarray(model_input["attention_masks"], dtype=np.int64)
        phoneme_masks = np.ascontiguousarray(model_input["phoneme_masks"], dtype=np.float32)
        char_ids = np.ascontiguousarray(model_input["char_ids"], dtype=np.int64)
        position_ids = np.ascontiguousarray(model_input["position_ids"], dtype=np.int64)
        batch_size = int(char_ids.shape[0])
        sequence_batch_size = int(input_ids.shape[0])
        sequence_ids_raw = model_input.get("sequence_ids")
        if sequence_ids_raw is None:
            if sequence_batch_size == batch_size:
                sequence_ids = np.arange(batch_size, dtype=np.int64)
            elif sequence_batch_size == 1 and batch_size >= 1:
                sequence_ids = np.zeros((batch_size,), dtype=np.int64)
            else:
                raise G2PWCudaError(
                    "model_input 缺少 sequence_ids，且 input/query 批大小不匹配: "
                    f"sequence_batch_size={sequence_batch_size} query_batch_size={batch_size}"
                )
        else:
            sequence_ids = np.ascontiguousarray(sequence_ids_raw, dtype=np.int64)
            if sequence_ids.ndim != 1 or int(sequence_ids.shape[0]) != batch_size:
                raise G2PWCudaError(
                    "sequence_ids 形状无效: "
                    f"expected=({batch_size},) actual={tuple(sequence_ids.shape)}"
                )
        if sequence_batch_size <= 0 or batch_size <= 0:
            raise G2PWCudaError(
                "model_input 批大小无效: "
                f"sequence_batch_size={sequence_batch_size} query_batch_size={batch_size}"
            )
        min_sequence_id = int(sequence_ids.min()) if batch_size > 0 else 0
        max_sequence_id = int(sequence_ids.max()) if batch_size > 0 else 0
        if min_sequence_id < 0 or max_sequence_id >= sequence_batch_size:
            raise G2PWCudaError(
                "sequence_ids 超出 sequence batch 范围: "
                f"min_sequence_id={min_sequence_id} max_sequence_id={max_sequence_id} "
                f"sequence_batch_size={sequence_batch_size}"
            )
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_masks": attention_masks,
            "phoneme_masks": phoneme_masks,
            "char_ids": char_ids,
            "position_ids": position_ids,
            "sequence_ids": sequence_ids,
        }

    def _run_direct(self, model_input: Dict[str, np.ndarray]) -> np.ndarray:
        normalized = self._normalize_model_input(model_input)
        input_ids = normalized["input_ids"]
        token_type_ids = normalized["token_type_ids"]
        attention_masks = normalized["attention_masks"]
        phoneme_masks = normalized["phoneme_masks"]
        char_ids = normalized["char_ids"]
        position_ids = normalized["position_ids"]
        sequence_ids = normalized["sequence_ids"]
        batch_size = int(char_ids.shape[0])
        sequence_batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])
        probs = np.empty((batch_size, self.num_labels), dtype=np.float32)
        with self.lock:
            self._ensure_capacity(batch_size=max(batch_size, sequence_batch_size), seq_len=seq_len)
            status = self.lib.g2pw_runtime_run(
                self.handle,
                input_ids.ctypes.data_as(ctypes.c_void_p),
                token_type_ids.ctypes.data_as(ctypes.c_void_p),
                attention_masks.ctypes.data_as(ctypes.c_void_p),
                phoneme_masks.ctypes.data_as(ctypes.c_void_p),
                char_ids.ctypes.data_as(ctypes.c_void_p),
                position_ids.ctypes.data_as(ctypes.c_void_p),
                sequence_ids.ctypes.data_as(ctypes.c_void_p),
                batch_size,
                sequence_batch_size,
                seq_len,
                probs.ctypes.data_as(ctypes.c_void_p),
            )
            if int(status) != 0:
                raise G2PWCudaError(f"g2pw-cu inference failed: {self._last_error()}")
        return probs

    def _split_model_input_for_direct_run(self, model_input: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        normalized = self._normalize_model_input(model_input)
        query_rows = int(normalized["char_ids"].shape[0])
        sequence_rows = int(normalized["input_ids"].shape[0])
        seq_len = int(normalized["input_ids"].shape[1])
        max_rows = max(1, int(self.direct_max_rows))
        max_sequences = max(1, int(self.direct_max_sequences))
        max_tokens = max(max(1, int(self.direct_max_tokens)), seq_len)
        if (
            query_rows <= max_rows
            and sequence_rows <= max_sequences
            and (sequence_rows * seq_len) <= max_tokens
        ):
            return [normalized]

        sequence_ids = normalized["sequence_ids"]
        chunks: List[Dict[str, np.ndarray]] = []
        query_start = 0
        while query_start < query_rows:
            query_end = query_start
            chunk_sequence_ids: List[int] = []
            sequence_map: Dict[int, int] = {}
            while query_end < query_rows:
                source_sequence_id = int(sequence_ids[query_end])
                if source_sequence_id not in sequence_map:
                    proposed_sequence_count = len(chunk_sequence_ids) + 1
                    proposed_tokens = proposed_sequence_count * seq_len
                    if chunk_sequence_ids and (
                        proposed_sequence_count > max_sequences or proposed_tokens > max_tokens
                    ):
                        break
                    sequence_map[source_sequence_id] = len(chunk_sequence_ids)
                    chunk_sequence_ids.append(source_sequence_id)
                if (query_end - query_start + 1) > max_rows and query_end > query_start:
                    break
                query_end += 1
                if (query_end - query_start) >= max_rows:
                    break

            if query_end <= query_start:
                query_end = query_start + 1
                source_sequence_id = int(sequence_ids[query_start])
                sequence_map = {source_sequence_id: 0}
                chunk_sequence_ids = [source_sequence_id]

            local_query_slice = slice(query_start, query_end)
            local_sequence_indices = np.asarray(chunk_sequence_ids, dtype=np.int64)
            local_sequence_ids = np.asarray(
                [sequence_map[int(value)] for value in sequence_ids[local_query_slice]],
                dtype=np.int64,
            )
            chunks.append(
                {
                    "input_ids": np.ascontiguousarray(normalized["input_ids"][local_sequence_indices]),
                    "token_type_ids": np.ascontiguousarray(normalized["token_type_ids"][local_sequence_indices]),
                    "attention_masks": np.ascontiguousarray(normalized["attention_masks"][local_sequence_indices]),
                    "phoneme_masks": np.ascontiguousarray(normalized["phoneme_masks"][local_query_slice]),
                    "char_ids": np.ascontiguousarray(normalized["char_ids"][local_query_slice]),
                    "position_ids": np.ascontiguousarray(normalized["position_ids"][local_query_slice]),
                    "sequence_ids": np.ascontiguousarray(local_sequence_ids),
                }
            )
            query_start = query_end
        return chunks

    def _run_chunked_direct_with_profile(
        self,
        model_inputs: List[Dict[str, np.ndarray]],
    ) -> tuple[np.ndarray, Dict[str, float]]:
        outputs: List[np.ndarray] = []
        total_run_ms = 0.0
        max_chunk_rows = 0
        max_chunk_sequences = 0
        max_chunk_seq_len = 0
        for chunk in model_inputs:
            started = time.perf_counter()
            outputs.append(self._run_direct(chunk))
            total_run_ms += float((time.perf_counter() - started) * 1000.0)
            max_chunk_rows = max(max_chunk_rows, int(chunk["char_ids"].shape[0]))
            max_chunk_sequences = max(max_chunk_sequences, int(chunk["input_ids"].shape[0]))
            max_chunk_seq_len = max(max_chunk_seq_len, int(chunk["input_ids"].shape[1]))
        merged_output = np.ascontiguousarray(np.concatenate(outputs, axis=0)) if outputs else np.zeros((0, 0), dtype=np.float32)
        total_rows = sum(int(chunk["char_ids"].shape[0]) for chunk in model_inputs)
        return merged_output, {
            "g2pw_runtime_queue_wait_ms": 0.0,
            "g2pw_runtime_collect_wait_ms": 0.0,
            "g2pw_runtime_run_ms": float(total_run_ms),
            "g2pw_runtime_batch_rows": float(total_rows),
            "g2pw_runtime_batch_requests": float(len(model_inputs)),
            "g2pw_runtime_task_rows": float(total_rows),
            "g2pw_runtime_task_seq_len": float(max_chunk_seq_len),
            "g2pw_runtime_chunk_count": float(len(model_inputs)),
            "g2pw_runtime_chunk_rows_peak": float(max_chunk_rows),
            "g2pw_runtime_chunk_sequences_peak": float(max_chunk_sequences),
            "g2pw_runtime_shard_index": float(self.shard_index),
        }

    def _can_append_task(self, tasks: List[G2PWBatchTask], candidate: G2PWBatchTask) -> bool:
        request_count = len(tasks) + 1
        if request_count > self.batch_max_requests:
            return False
        total_rows = sum(int(item.model_input["char_ids"].shape[0]) for item in tasks) + int(
            candidate.model_input["char_ids"].shape[0]
        )
        if total_rows > self.batch_max_rows:
            return False
        total_tokens = sum(
            int(item.model_input["input_ids"].shape[0]) * int(item.model_input["input_ids"].shape[1]) for item in tasks
        ) + int(candidate.model_input["input_ids"].shape[0]) * int(candidate.model_input["input_ids"].shape[1])
        return total_tokens <= self.batch_max_tokens

    def _merge_batch_inputs(self, tasks: List[G2PWBatchTask]) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int]]]:
        normalized_inputs = [self._normalize_model_input(task.model_input) for task in tasks]
        total_query_rows = sum(int(item["char_ids"].shape[0]) for item in normalized_inputs)
        total_sequence_rows = sum(int(item["input_ids"].shape[0]) for item in normalized_inputs)
        max_seq_len = max(int(item["input_ids"].shape[1]) for item in normalized_inputs)
        input_ids = np.zeros((total_sequence_rows, max_seq_len), dtype=np.int64)
        token_type_ids = np.zeros((total_sequence_rows, max_seq_len), dtype=np.int64)
        attention_masks = np.zeros((total_sequence_rows, max_seq_len), dtype=np.int64)
        phoneme_masks = np.zeros((total_query_rows, normalized_inputs[0]["phoneme_masks"].shape[1]), dtype=np.float32)
        char_ids = np.zeros((total_query_rows,), dtype=np.int64)
        position_ids = np.zeros((total_query_rows,), dtype=np.int64)
        sequence_ids = np.zeros((total_query_rows,), dtype=np.int64)
        slices: List[Tuple[int, int]] = []
        query_cursor = 0
        sequence_cursor = 0
        for item in normalized_inputs:
            sequence_rows = int(item["input_ids"].shape[0])
            rows = int(item["char_ids"].shape[0])
            seq_len = int(item["input_ids"].shape[1])
            next_sequence_cursor = sequence_cursor + sequence_rows
            next_query_cursor = query_cursor + rows
            input_ids[sequence_cursor:next_sequence_cursor, :seq_len] = item["input_ids"]
            token_type_ids[sequence_cursor:next_sequence_cursor, :seq_len] = item["token_type_ids"]
            attention_masks[sequence_cursor:next_sequence_cursor, :seq_len] = item["attention_masks"]
            phoneme_masks[query_cursor:next_query_cursor] = item["phoneme_masks"]
            char_ids[query_cursor:next_query_cursor] = item["char_ids"]
            position_ids[query_cursor:next_query_cursor] = item["position_ids"]
            sequence_ids[query_cursor:next_query_cursor] = item["sequence_ids"] + sequence_cursor
            slices.append((query_cursor, next_query_cursor))
            query_cursor = next_query_cursor
            sequence_cursor = next_sequence_cursor
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_masks": attention_masks,
            "phoneme_masks": phoneme_masks,
            "char_ids": char_ids,
            "position_ids": position_ids,
            "sequence_ids": sequence_ids,
        }, slices

    def _finish_task(
        self,
        task: G2PWBatchTask,
        output: np.ndarray | None = None,
        profile: Dict[str, float] | None = None,
        error: Exception | None = None,
    ) -> None:
        task.output = output
        task.profile = dict(profile or {})
        task.error = error
        task.done_event.set()

    def _batch_loop(self) -> None:
        while True:
            with self.batch_condition:
                while not self.pending_tasks and not self.closed:
                    self.batch_condition.wait()
                if self.closed and not self.pending_tasks:
                    return
                first_task = self.pending_tasks.popleft()
                batch_tasks = [first_task]
                collect_started = time.perf_counter()
                deadline = collect_started + self.batch_window_s
                while True:
                    if len(batch_tasks) >= self.batch_max_requests:
                        break
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    if not self.pending_tasks:
                        self.batch_condition.wait(timeout=remaining)
                        continue
                    candidate = self.pending_tasks[0]
                    if not self._can_append_task(batch_tasks, candidate):
                        break
                    batch_tasks.append(self.pending_tasks.popleft())
                collect_wait_ms = max(0.0, (time.perf_counter() - collect_started) * 1000.0)

            now = time.perf_counter()
            queue_wait_values = [max(0.0, (now - task.enqueued_at) * 1000.0) for task in batch_tasks]
            batch_started = now
            batch_rows = 0
            batch_sequence_rows = 0
            batch_max_seq_len = 0
            run_started = 0.0
            run_finished = 0.0
            trace_error = 0
            try:
                merged_input, row_slices = self._merge_batch_inputs(batch_tasks)
                batch_rows = int(merged_input["char_ids"].shape[0])
                batch_sequence_rows = int(merged_input["input_ids"].shape[0])
                batch_max_seq_len = int(merged_input["input_ids"].shape[1])
                run_started = time.perf_counter()
                merged_chunks = self._split_model_input_for_direct_run(merged_input)
                if len(merged_chunks) > 1:
                    merged_output, merged_profile = self._run_chunked_direct_with_profile(merged_chunks)
                else:
                    merged_output = self._run_direct(merged_input)
                    merged_profile = {}
                run_finished = time.perf_counter()
                run_ms = float(
                    merged_profile.get("g2pw_runtime_run_ms", max(0.0, (run_finished - run_started) * 1000.0))
                )
                for task, (start, end) in zip(batch_tasks, row_slices):
                    task_rows = int(task.model_input["char_ids"].shape[0])
                    task_seq_len = int(task.model_input["input_ids"].shape[1])
                    task_profile = {
                        "g2pw_runtime_queue_wait_ms": float(max(0.0, (run_started - task.enqueued_at) * 1000.0)),
                        "g2pw_runtime_collect_wait_ms": float(collect_wait_ms),
                        "g2pw_runtime_run_ms": float(run_ms),
                        "g2pw_runtime_batch_rows": float(sum(int(item.model_input["char_ids"].shape[0]) for item in batch_tasks)),
                        "g2pw_runtime_batch_requests": float(len(batch_tasks)),
                        "g2pw_runtime_task_rows": float(task_rows),
                        "g2pw_runtime_task_seq_len": float(task_seq_len),
                        "g2pw_runtime_shard_index": float(self.shard_index),
                    }
                    for key in (
                        "g2pw_runtime_chunk_count",
                        "g2pw_runtime_chunk_rows_peak",
                        "g2pw_runtime_chunk_sequences_peak",
                    ):
                        if key in merged_profile:
                            task_profile[key] = float(merged_profile[key])
                    self._finish_task(
                        task,
                        output=np.ascontiguousarray(merged_output[start:end]),
                        profile=task_profile,
                    )
            except Exception as exc:
                run_ms = 0.0
                trace_error = 1
                run_finished = time.perf_counter()
                for task in batch_tasks:
                    self._finish_task(task, error=exc)
            finally:
                batch_finished = time.perf_counter()
                if run_started <= 0.0:
                    run_started = batch_finished
                if run_finished <= 0.0:
                    run_finished = batch_finished
                trace_gpu_batch(
                    "g2pw_runtime_batch",
                    stage="g2pw_runtime",
                    shard_index=int(self.shard_index),
                    batch_requests=int(len(batch_tasks)),
                    batch_rows=int(batch_rows),
                    batch_sequence_rows=int(batch_sequence_rows),
                    batch_max_seq_len=int(batch_max_seq_len),
                    queue_wait_ms=float(sum(queue_wait_values) / max(1, len(queue_wait_values))),
                    queue_wait_peak_ms=float(max(queue_wait_values or [0.0])),
                    collect_wait_ms=float(collect_wait_ms),
                    run_ms=float(run_ms),
                    error=int(trace_error),
                    batch_started_ts=float(batch_started),
                    run_started_ts=float(run_started),
                    run_finished_ts=float(run_finished),
                    batch_finished_ts=float(batch_finished),
                )
                with self.batch_condition:
                    self.batch_total_batches += 1
                    self.batch_total_tasks += len(batch_tasks)
                    self.batch_total_rows += sum(int(task.model_input["char_ids"].shape[0]) for task in batch_tasks)
                    self.batch_total_queue_wait_ms += float(sum(queue_wait_values))
                    self.batch_queue_wait_peak_ms = max(self.batch_queue_wait_peak_ms, max(queue_wait_values or [0.0]))
                    self.batch_total_collect_wait_ms += float(collect_wait_ms) * float(len(batch_tasks))
                    self.batch_collect_wait_peak_ms = max(self.batch_collect_wait_peak_ms, float(collect_wait_ms))
                    self.batch_total_run_ms += float(run_ms)
                    self.batch_run_peak_ms = max(self.batch_run_peak_ms, float(run_ms))
                    self.batch_rows_peak = max(
                        self.batch_rows_peak, sum(int(task.model_input["char_ids"].shape[0]) for task in batch_tasks)
                    )
                    self.batch_requests_peak = max(self.batch_requests_peak, len(batch_tasks))

    def _submit_batched(self, model_input: Dict[str, np.ndarray]) -> tuple[np.ndarray, Dict[str, float]]:
        task = G2PWBatchTask(model_input=model_input)
        with self.batch_condition:
            if self.closed:
                raise G2PWCudaError("g2pw-cu batch worker already closed")
            task.enqueued_at = time.perf_counter()
            self.pending_tasks.append(task)
            self.batch_pending_peak = max(self.batch_pending_peak, len(self.pending_tasks))
            self.batch_condition.notify_all()
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        assert task.output is not None
        return task.output, dict(task.profile)

    def snapshot(self) -> Dict[str, float | int | bool]:
        with self.batch_condition:
            average_tasks_per_batch = (
                float(self.batch_total_tasks) / float(self.batch_total_batches) if self.batch_total_batches > 0 else 0.0
            )
            average_rows_per_batch = (
                float(self.batch_total_rows) / float(self.batch_total_batches) if self.batch_total_batches > 0 else 0.0
            )
            average_queue_wait_ms = (
                float(self.batch_total_queue_wait_ms) / float(self.batch_total_tasks) if self.batch_total_tasks > 0 else 0.0
            )
            average_collect_wait_ms = (
                float(self.batch_total_collect_wait_ms) / float(self.batch_total_tasks)
                if self.batch_total_tasks > 0
                else 0.0
            )
            return {
                "shard_index": int(self.shard_index),
                "enabled": bool(self.batch_enabled),
                "enable_cuda_graph": bool(self.enable_cuda_graph),
                "enable_profiling": bool(self.enable_profiling),
                "full_graph_cache_limit": int(self.full_graph_cache_limit),
                "tail_graph_cache_limit": int(self.tail_graph_cache_limit),
                "window_ms": float(self.batch_window_s * 1000.0),
                "max_requests": int(self.batch_max_requests),
                "max_rows": int(self.batch_max_rows),
                "max_tokens": int(self.batch_max_tokens),
                "pending": int(len(self.pending_tasks)),
                "pending_peak": int(self.batch_pending_peak),
                "total_batches": int(self.batch_total_batches),
                "total_tasks": int(self.batch_total_tasks),
                "total_rows": int(self.batch_total_rows),
                "avg_tasks_per_batch": float(average_tasks_per_batch),
                "avg_rows_per_batch": float(average_rows_per_batch),
                "avg_queue_wait_ms": float(average_queue_wait_ms),
                "queue_wait_peak_ms": float(self.batch_queue_wait_peak_ms),
                "avg_collect_wait_ms": float(average_collect_wait_ms),
                "collect_wait_peak_ms": float(self.batch_collect_wait_peak_ms),
                "run_total_ms": float(self.batch_total_run_ms),
                "run_peak_ms": float(self.batch_run_peak_ms),
                "batch_rows_peak": int(self.batch_rows_peak),
                "batch_requests_peak": int(self.batch_requests_peak),
            }

    def pending_rows(self) -> int:
        with self.batch_condition:
            return int(sum(int(task.model_input["char_ids"].shape[0]) for task in self.pending_tasks))

    def pending_count(self) -> int:
        with self.batch_condition:
            return int(len(self.pending_tasks))

    def run_with_profile(self, model_input: Dict[str, np.ndarray]) -> tuple[np.ndarray, Dict[str, float]]:
        direct_chunks = self._split_model_input_for_direct_run(model_input)
        if len(direct_chunks) > 1:
            return self._run_chunked_direct_with_profile(direct_chunks)
        if not self.batch_enabled:
            started = time.perf_counter()
            output = self._run_direct(model_input)
            return output, {
                "g2pw_runtime_queue_wait_ms": 0.0,
                "g2pw_runtime_collect_wait_ms": 0.0,
                "g2pw_runtime_run_ms": float((time.perf_counter() - started) * 1000.0),
                "g2pw_runtime_batch_rows": float(model_input["char_ids"].shape[0]),
                "g2pw_runtime_batch_requests": 1.0,
                "g2pw_runtime_task_rows": float(model_input["char_ids"].shape[0]),
                "g2pw_runtime_task_seq_len": float(model_input["input_ids"].shape[1]),
                "g2pw_runtime_shard_index": float(self.shard_index),
            }
        return self._submit_batched(model_input)

    def run(self, model_input: Dict[str, np.ndarray]) -> np.ndarray:
        output, _profile = self.run_with_profile(model_input)
        return output


class G2PWRuntimePool:
    def __init__(self) -> None:
        self.worker_count = max(1, _env_int("GPTSOVITS_G2PW_CUDA_WORKERS", 2))
        self.shards = [G2PWRuntimeWrapper(shard_index=index) for index in range(self.worker_count)]
        self.lock = threading.Lock()

    def _pick_shard(self) -> G2PWRuntimeWrapper:
        with self.lock:
            return min(
                self.shards,
                key=lambda shard: (
                    shard.pending_rows(),
                    shard.pending_count(),
                    shard.snapshot().get("avg_queue_wait_ms", 0.0),
                ),
            )

    def run_with_profile(self, model_input: Dict[str, np.ndarray]) -> tuple[np.ndarray, Dict[str, float]]:
        shard = self._pick_shard()
        output, profile = shard.run_with_profile(model_input)
        profile["g2pw_runtime_pool_workers"] = float(self.worker_count)
        return output, profile

    def run(self, model_input: Dict[str, np.ndarray]) -> np.ndarray:
        output, _profile = self.run_with_profile(model_input)
        return output

    def snapshot(self) -> Dict[str, float | int | bool | List[Dict[str, float | int | bool]]]:
        shard_snapshots = [dict(shard.snapshot()) for shard in self.shards]
        avg_queue_wait_ms = 0.0
        total_tasks = 0.0
        pending = 0
        pending_peak = 0
        total_batches = 0
        total_rows = 0
        batch_rows_peak = 0
        batch_requests_peak = 0
        for snapshot in shard_snapshots:
            tasks = float(snapshot.get("total_tasks", 0.0))
            avg_queue_wait_ms += float(snapshot.get("avg_queue_wait_ms", 0.0)) * tasks
            total_tasks += tasks
            pending += int(snapshot.get("pending", 0))
            pending_peak = max(pending_peak, int(snapshot.get("pending_peak", 0)))
            total_batches += int(snapshot.get("total_batches", 0))
            total_rows += int(snapshot.get("total_rows", 0))
            batch_rows_peak = max(batch_rows_peak, int(snapshot.get("batch_rows_peak", 0)))
            batch_requests_peak = max(batch_requests_peak, int(snapshot.get("batch_requests_peak", 0)))
        return {
            "worker_count": int(self.worker_count),
            "pending": int(pending),
            "pending_peak": int(pending_peak),
            "total_batches": int(total_batches),
            "total_tasks": int(total_tasks),
            "total_rows": int(total_rows),
            "avg_queue_wait_ms": float(avg_queue_wait_ms / total_tasks) if total_tasks > 0 else 0.0,
            "batch_rows_peak": int(batch_rows_peak),
            "batch_requests_peak": int(batch_requests_peak),
            "shards": shard_snapshots,
        }


class G2PWCudaConverter(_G2PWBaseOnnxConverter):
    def __init__(
        self,
        model_dir: str = "G2PWModel/",
        style: str = "bopomofo",
        model_source: str = None,
        enable_non_tradional_chinese: bool = False,
    ):
        super().__init__(
            model_dir=model_dir,
            style=style,
            model_source=model_source,
            enable_non_tradional_chinese=enable_non_tradional_chinese,
        )
        self.runtime = G2PWRuntimePool()
        self.backend = "cuda"
        primary_runtime = self.runtime.shards[0]
        self.device = f"cuda:{primary_runtime.device_ordinal}"
        self.checkpoint_path = str(primary_runtime.weights_path)
        self.providers = ["g2pw-cu"]
        self._prewarm_lock = threading.Lock()
        self._prewarmed = False

    def _predict(self, model_input: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        probs = self.runtime.run(model_input)
        preds = np.argmax(probs, axis=1).tolist()
        confidences = probs[np.arange(len(preds)), preds].astype(np.float32, copy=False).tolist()
        return [self.labels[pred] for pred in preds], confidences

    def _predict_with_profile(self, model_input: Dict[str, Any]) -> Tuple[List[str], List[float], Dict[str, float]]:
        started = time.perf_counter()
        probs, runtime_profile = self.runtime.run_with_profile(model_input)
        preds = np.argmax(probs, axis=1).tolist()
        confidences = probs[np.arange(len(preds)), preds].astype(np.float32, copy=False).tolist()
        profile = dict(runtime_profile)
        profile["g2pw_runtime_total_ms"] = float((time.perf_counter() - started) * 1000.0)
        profile["g2pw_predict_ms"] = float(profile["g2pw_runtime_total_ms"])
        return [self.labels[pred] for pred in preds], confidences, profile

    def snapshot(self) -> Dict[str, float | int | bool]:
        return dict(self.runtime.snapshot())

    def prewarm(self, sentences: List[str] | None = None, rounds: int = 1) -> bool:
        with self._prewarm_lock:
            if self._prewarmed:
                return False
            warm_sentences = list(sentences or [
                "重庆银行的行长在长安见到了重要的人。",
                "音乐老师重新调整了长句里的重音和节奏。",
                "我们准备把参考文本和目标文本一起处理。",
                "这个系统需要在高并发下稳定完成预处理。",
            ])
            warm_sentences = [str(item).strip() for item in warm_sentences if str(item).strip()]
            if not warm_sentences:
                warm_sentences = ["重庆银行的行长在长安见到了重要的人。"]
            multiplier_values = []
            for raw in str(os.environ.get("GPTSOVITS_G2PW_CUDA_PREWARM_BATCH_MULTIPLIERS", "1,2,4")).split(","):
                raw = raw.strip()
                if not raw:
                    continue
                multiplier_values.append(max(1, int(raw)))
            if not multiplier_values:
                multiplier_values = [1]
            warm_rounds = max(1, int(rounds))
            for _ in range(warm_rounds):
                for multiplier in sorted(set(multiplier_values)):
                    batch = list(warm_sentences) * int(multiplier)
                    if not batch:
                        continue
                    self.predict_sentences_with_profile(batch)
                    if len(batch) > 1:
                        self.predict_sentences_with_profile(list(reversed(batch)))
            self._prewarmed = True
            return True
