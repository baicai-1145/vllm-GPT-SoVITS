from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from AR.models.utils import logits_to_probs, make_pad_mask_left, multinomial_sample_one_no_sync, sample


def _sync_device(device: Any) -> None:
    try:
        device_str = str(device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception:
        pass


@dataclass
class SchedulerRequestSpec:
    request_id: str
    ref_audio_path: Path
    prompt_text: str
    prompt_lang: str
    text: str
    text_lang: str
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    early_stop_num: int
    aux_ref_audio_paths: List[str] = field(default_factory=list)
    ready_step: int = 0


@dataclass
class T2SRequestState:
    request_id: str
    ref_audio_path: Path
    prompt_text: str
    prompt_lang: str
    text: str
    text_lang: str
    norm_prompt_text: str
    norm_text: str
    phones: torch.LongTensor
    prompt_phones: torch.LongTensor
    all_phones: torch.LongTensor
    all_bert_features: torch.Tensor
    prompt_semantic: torch.LongTensor
    refer_spec: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]
    aux_refer_specs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
    raw_audio: torch.Tensor
    raw_sr: int
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    early_stop_num: int
    ready_step: int
    prepare_profile: Dict[str, float]


@dataclass
class T2SRunningRequest:
    state: T2SRequestState
    y_sequence: torch.LongTensor
    prefix_len: int
    decode_attn_mask: Optional[torch.Tensor]
    k_cache: List[torch.Tensor]
    v_cache: List[torch.Tensor]
    step_idx: int


@dataclass
class T2SFinishedItem:
    request_id: str
    semantic_tokens: torch.LongTensor
    finish_idx: int
    finish_reason: str


@dataclass
class T2SActiveBatch:
    request_ids: List[str]
    states: List[T2SRequestState]
    x: Optional[torch.Tensor]
    x_lens: Optional[torch.LongTensor]
    y_sequences: List[torch.LongTensor]
    prefix_lens: torch.LongTensor
    xy_pos: torch.Tensor
    key_padding_mask: Optional[torch.Tensor]
    prefill_attn_mask: Optional[torch.Tensor]
    decode_attn_mask: Optional[torch.Tensor]
    k_cache: Optional[List[torch.Tensor]]
    v_cache: Optional[List[torch.Tensor]]
    kv_lens: Optional[torch.LongTensor]
    step_indices: torch.LongTensor
    prefill_done: bool
    kv_cache_pooled: bool = False
    kv_cache_capacity: int = 0
    kv_cache_batch_capacity: int = 0


@dataclass
class PreparedTextFeatures:
    phones: List[int]
    bert_features: torch.Tensor
    norm_text: str
    profile: Dict[str, float]
    total_ms: float
    cpu_preprocess_ms: float


def build_empty_text_features(
    *,
    feature_dim: int = 1024,
    dtype: torch.dtype = torch.float32,
) -> PreparedTextFeatures:
    return PreparedTextFeatures(
        phones=[],
        bert_features=torch.empty((int(feature_dim), 0), dtype=dtype),
        norm_text="",
        profile={"cpu_preprocess_ms": 0.0, "bert_total_ms": 0.0},
        total_ms=0.0,
        cpu_preprocess_ms=0.0,
    )


def normalize_sentence(text: str, language: str) -> str:
    text = text.strip("\n").strip()
    if not text:
        return text
    if text[-1] not in {",", ".", "?", "!", "，", "。", "？", "！", "…", "；", ";", ":"}:
        text += "。" if language != "en" else "."
    return text


@torch.inference_mode()
def prepare_text_features(
    tts: Any,
    text: str,
    language: str,
) -> PreparedTextFeatures:
    device = tts.configs.device
    profile: Dict[str, float] = {}
    branch_start = time.perf_counter()
    profile["_branch_start_ts"] = float(branch_start)
    _sync_device(device)
    cpu_start = time.perf_counter()
    prepared_segments = tts.prepare_text_segments(text, language)
    _sync_device(device)
    cpu_preprocess_ms = (time.perf_counter() - cpu_start) * 1000.0
    profile["cpu_preprocess_ms"] = float(cpu_preprocess_ms)
    bert_start = time.perf_counter()
    phones, bert_features, norm_text = tts.build_text_features_from_segments(prepared_segments, profile=profile)
    _sync_device(device)
    profile["bert_total_ms"] = (time.perf_counter() - bert_start) * 1000.0
    total_ms = (time.perf_counter() - branch_start) * 1000.0
    profile.pop("_branch_start_ts", None)
    return PreparedTextFeatures(
        phones=phones,
        bert_features=bert_features,
        norm_text=norm_text,
        profile=profile,
        total_ms=float(total_ms),
        cpu_preprocess_ms=float(cpu_preprocess_ms),
    )


@torch.inference_mode()
def build_request_state_from_parts(
    tts: Any,
    spec: SchedulerRequestSpec,
    prompt_text: str,
    text: str,
    prompt_result: PreparedTextFeatures,
    target_result: PreparedTextFeatures,
    ref_audio_bundle: Dict[str, Any],
    prepare_start: float,
    prepare_sync_start: float,
    profile_overrides: Optional[Dict[str, float]] = None,
) -> T2SRequestState:
    device = tts.configs.device
    _sync_device(device)
    ref_audio_bundle_ms = float(ref_audio_bundle.get("profile", {}).get("bundle_total_ms", 0.0))
    bundle_profile = ref_audio_bundle.get("profile", {})
    prompt_semantic = ref_audio_bundle["prompt_semantic"].long()
    refer_spec_value = ref_audio_bundle.get("refer_spec")
    if refer_spec_value in [None, ()]:
        spec_audio, audio_16k = None, None
    else:
        spec_audio, audio_16k = refer_spec_value
    aux_refer_specs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    for aux_ref_audio_path in list(getattr(spec, "aux_ref_audio_paths", []) or []):
        if aux_ref_audio_path in [None, ""]:
            continue
        if not os.path.exists(str(aux_ref_audio_path)):
            continue
        aux_spec_audio, aux_audio_16k, _, _ = tts.extract_ref_spec(str(aux_ref_audio_path))
        aux_refer_specs.append((aux_spec_audio, aux_audio_16k))
    raw_audio = ref_audio_bundle["raw_audio"]
    raw_sr = int(ref_audio_bundle["raw_sr"])
    prompt_semantic_ms = float(bundle_profile.get("prompt_semantic_ms", ref_audio_bundle_ms))
    ref_spec_ms = float(bundle_profile.get("ref_spec_ms", 0.0))
    audio_load_ms = float(bundle_profile.get("audio_load_ms", 0.0))

    _sync_device(device)
    tensorize_start = time.perf_counter()
    phones_tensor = torch.LongTensor(target_result.phones).to(tts.configs.device)
    prompt_phones_tensor = torch.LongTensor(prompt_result.phones).to(tts.configs.device)
    all_phones = torch.LongTensor(prompt_result.phones + target_result.phones).to(tts.configs.device)
    prompt_bert_features = prompt_result.bert_features.to(dtype=tts.precision, device=tts.configs.device)
    target_bert_features = target_result.bert_features.to(dtype=tts.precision, device=tts.configs.device)
    all_bert_features = torch.cat([prompt_bert_features, target_bert_features], dim=1)
    _sync_device(device)
    tensorize_ms = (time.perf_counter() - tensorize_start) * 1000.0

    prepare_profile = {
        "prompt_text_features_ms": float(prompt_result.total_ms),
        "text_features_ms": float(target_result.total_ms),
        "prompt_text_cpu_preprocess_ms": float(prompt_result.cpu_preprocess_ms),
        "text_cpu_preprocess_ms": float(target_result.cpu_preprocess_ms),
        "prompt_text_bert_wait_ms": float(prompt_result.profile.get("bert_wait_ms", 0.0)),
        "prompt_text_bert_admission_wait_ms": float(prompt_result.profile.get("bert_admission_wait_ms", 0.0)),
        "prompt_text_bert_queue_wait_ms": float(prompt_result.profile.get("bert_queue_wait_ms", 0.0)),
        "prompt_text_bert_worker_queue_wait_ms": float(
            prompt_result.profile.get("bert_worker_queue_wait_ms", 0.0)
        ),
        "prompt_text_bert_submit_offset_first_ms": float(
            prompt_result.profile.get("bert_submit_offset_first_ms", 0.0)
        ),
        "prompt_text_bert_submit_offset_last_ms": float(
            prompt_result.profile.get("bert_submit_offset_last_ms", 0.0)
        ),
        "prompt_text_bert_batch_collect_wait_ms": float(prompt_result.profile.get("bert_batch_collect_wait_ms", 0.0)),
        "prompt_text_bert_batch_dispatch_delay_ms": float(
            prompt_result.profile.get("bert_batch_dispatch_delay_ms", 0.0)
        ),
        "prompt_text_bert_forward_ms": float(prompt_result.profile.get("bert_forward_ms", 0.0)),
        "prompt_text_bert_tokenize_ms": float(prompt_result.profile.get("bert_tokenize_ms", 0.0)),
        "prompt_text_bert_scatter_ms": float(prompt_result.profile.get("bert_scatter_ms", 0.0)),
        "prompt_text_bert_calls": float(prompt_result.profile.get("bert_calls", 0.0)),
        "prompt_text_bert_stage_slots": float(prompt_result.profile.get("bert_stage_slots", 0.0)),
        "prompt_text_bert_stage_inflight_peak": float(prompt_result.profile.get("bert_stage_inflight_peak", 0.0)),
        "prompt_text_bert_batch_size_peak": float(prompt_result.profile.get("bert_batch_size_peak", 0.0)),
        "prompt_text_bert_batch_tokens_peak": float(prompt_result.profile.get("bert_batch_tokens_peak", 0.0)),
        "prompt_text_bert_pending_depth_on_enqueue_peak": float(
            prompt_result.profile.get("bert_pending_depth_on_enqueue_peak", 0.0)
        ),
        "prompt_text_bert_pending_depth_on_collect_peak": float(
            prompt_result.profile.get("bert_pending_depth_on_collect_peak", 0.0)
        ),
        "prompt_text_bert_high_pressure_mode_peak": float(
            prompt_result.profile.get("bert_high_pressure_mode_peak", 0.0)
        ),
        "prompt_text_bert_batch_window_ms": float(prompt_result.profile.get("bert_batch_window_ms", 0.0)),
        "prompt_text_g2pw_total_ms": float(prompt_result.profile.get("g2pw_total_ms", 0.0)),
        "prompt_text_g2pw_prepare_ms": float(prompt_result.profile.get("g2pw_prepare_ms", 0.0)),
        "prompt_text_g2pw_predict_ms": float(prompt_result.profile.get("g2pw_predict_ms", 0.0)),
        "prompt_text_g2pw_post_ms": float(prompt_result.profile.get("g2pw_post_ms", 0.0)),
        "prompt_text_g2pw_runtime_total_ms": float(prompt_result.profile.get("g2pw_runtime_total_ms", 0.0)),
        "prompt_text_g2pw_runtime_queue_wait_ms": float(
            prompt_result.profile.get("g2pw_runtime_queue_wait_ms", 0.0)
        ),
        "prompt_text_g2pw_runtime_collect_wait_ms": float(
            prompt_result.profile.get("g2pw_runtime_collect_wait_ms", 0.0)
        ),
        "prompt_text_g2pw_runtime_run_ms": float(prompt_result.profile.get("g2pw_runtime_run_ms", 0.0)),
        "prompt_text_g2pw_runtime_batch_rows_peak": float(
            prompt_result.profile.get("g2pw_runtime_batch_rows_peak", 0.0)
        ),
        "prompt_text_g2pw_runtime_batch_requests_peak": float(
            prompt_result.profile.get("g2pw_runtime_batch_requests_peak", 0.0)
        ),
        "prompt_text_parallel_future_wait_ms": 0.0,
        "prompt_text_parallel_future_executor_queue_ms": 0.0,
        "prompt_text_parallel_future_run_ms": float(prompt_result.total_ms),
        "prompt_text_parallel_future_finish_after_submit_ms": float(prompt_result.total_ms),
        "prompt_text_parallel_future_queue_tail_after_target_ms": 0.0,
        "prompt_text_parallel_future_run_tail_after_target_ms": 0.0,
        "text_bert_wait_ms": float(target_result.profile.get("bert_wait_ms", 0.0)),
        "text_bert_admission_wait_ms": float(target_result.profile.get("bert_admission_wait_ms", 0.0)),
        "text_bert_queue_wait_ms": float(target_result.profile.get("bert_queue_wait_ms", 0.0)),
        "text_bert_worker_queue_wait_ms": float(target_result.profile.get("bert_worker_queue_wait_ms", 0.0)),
        "text_bert_submit_offset_first_ms": float(target_result.profile.get("bert_submit_offset_first_ms", 0.0)),
        "text_bert_submit_offset_last_ms": float(target_result.profile.get("bert_submit_offset_last_ms", 0.0)),
        "text_bert_batch_collect_wait_ms": float(target_result.profile.get("bert_batch_collect_wait_ms", 0.0)),
        "text_bert_batch_dispatch_delay_ms": float(target_result.profile.get("bert_batch_dispatch_delay_ms", 0.0)),
        "text_bert_forward_ms": float(target_result.profile.get("bert_forward_ms", 0.0)),
        "text_bert_tokenize_ms": float(target_result.profile.get("bert_tokenize_ms", 0.0)),
        "text_bert_scatter_ms": float(target_result.profile.get("bert_scatter_ms", 0.0)),
        "text_bert_calls": float(target_result.profile.get("bert_calls", 0.0)),
        "text_bert_stage_slots": float(target_result.profile.get("bert_stage_slots", 0.0)),
        "text_bert_stage_inflight_peak": float(target_result.profile.get("bert_stage_inflight_peak", 0.0)),
        "text_bert_batch_size_peak": float(target_result.profile.get("bert_batch_size_peak", 0.0)),
        "text_bert_batch_tokens_peak": float(target_result.profile.get("bert_batch_tokens_peak", 0.0)),
        "text_bert_pending_depth_on_enqueue_peak": float(
            target_result.profile.get("bert_pending_depth_on_enqueue_peak", 0.0)
        ),
        "text_bert_pending_depth_on_collect_peak": float(
            target_result.profile.get("bert_pending_depth_on_collect_peak", 0.0)
        ),
        "text_bert_high_pressure_mode_peak": float(target_result.profile.get("bert_high_pressure_mode_peak", 0.0)),
        "text_bert_batch_window_ms": float(target_result.profile.get("bert_batch_window_ms", 0.0)),
        "text_g2pw_total_ms": float(target_result.profile.get("g2pw_total_ms", 0.0)),
        "text_g2pw_prepare_ms": float(target_result.profile.get("g2pw_prepare_ms", 0.0)),
        "text_g2pw_predict_ms": float(target_result.profile.get("g2pw_predict_ms", 0.0)),
        "text_g2pw_post_ms": float(target_result.profile.get("g2pw_post_ms", 0.0)),
        "text_g2pw_runtime_total_ms": float(target_result.profile.get("g2pw_runtime_total_ms", 0.0)),
        "text_g2pw_runtime_queue_wait_ms": float(target_result.profile.get("g2pw_runtime_queue_wait_ms", 0.0)),
        "text_g2pw_runtime_collect_wait_ms": float(target_result.profile.get("g2pw_runtime_collect_wait_ms", 0.0)),
        "text_g2pw_runtime_run_ms": float(target_result.profile.get("g2pw_runtime_run_ms", 0.0)),
        "text_g2pw_runtime_batch_rows_peak": float(target_result.profile.get("g2pw_runtime_batch_rows_peak", 0.0)),
        "text_g2pw_runtime_batch_requests_peak": float(
            target_result.profile.get("g2pw_runtime_batch_requests_peak", 0.0)
        ),
        "text_feature_pair_ms": float(max(prompt_result.total_ms, target_result.total_ms)),
        "text_cpu_parallel_workers": float(getattr(tts, "prepare_text_cpu_workers", 0)),
        "audio_load_ms": audio_load_ms,
        "audio_stage_wait_ms": float(bundle_profile.get("audio_stage_wait_ms", 0.0)),
        "audio_stage_slots": float(bundle_profile.get("audio_stage_slots", 0.0)),
        "audio_stage_inflight_peak": float(bundle_profile.get("audio_stage_inflight_peak", 0.0)),
        "prompt_semantic_ms": prompt_semantic_ms,
        "prompt_semantic_wait_ms": float(bundle_profile.get("prompt_semantic_wait_ms", 0.0)),
        "prompt_semantic_submit_offset_ms": float(bundle_profile.get("prompt_semantic_submit_offset_ms", 0.0)),
        "prompt_semantic_submit_after_load_ms": float(bundle_profile.get("prompt_semantic_submit_after_load_ms", 0.0)),
        "prompt_semantic_cpu_prepare_wait_ms": float(bundle_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0)),
        "prompt_semantic_cpu_prepare_slots": float(bundle_profile.get("prompt_semantic_cpu_prepare_slots", 0.0)),
        "prompt_semantic_cpu_prepare_inflight_peak": float(
            bundle_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0)
        ),
        "prompt_semantic_worker_queue_wait_ms": float(
            bundle_profile.get("prompt_semantic_worker_queue_wait_ms", 0.0)
        ),
        "prompt_semantic_batch_collect_wait_ms": float(
            bundle_profile.get("prompt_semantic_batch_collect_wait_ms", 0.0)
        ),
        "prompt_semantic_stage_limiter_wait_ms": float(
            bundle_profile.get("prompt_semantic_stage_limiter_wait_ms", 0.0)
        ),
        "prompt_semantic_batch_dispatch_delay_ms": float(
            bundle_profile.get("prompt_semantic_batch_dispatch_delay_ms", 0.0)
        ),
        "prompt_semantic_cpu_prepare_ms": float(bundle_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)),
        "prompt_semantic_pack_ms": float(bundle_profile.get("prompt_semantic_pack_ms", 0.0)),
        "prompt_semantic_h2d_ms": float(bundle_profile.get("prompt_semantic_h2d_ms", 0.0)),
        "prompt_semantic_ssl_forward_ms": float(bundle_profile.get("prompt_semantic_ssl_forward_ms", 0.0)),
        "prompt_semantic_hidden_length_ms": float(bundle_profile.get("prompt_semantic_hidden_length_ms", 0.0)),
        "prompt_semantic_extract_latent_ms": float(bundle_profile.get("prompt_semantic_extract_latent_ms", 0.0)),
        "prompt_semantic_forward_ms": float(bundle_profile.get("prompt_semantic_forward_ms", 0.0)),
        "prompt_semantic_scatter_ms": float(bundle_profile.get("prompt_semantic_scatter_ms", 0.0)),
        "prompt_semantic_stage_slots": float(bundle_profile.get("prompt_semantic_stage_slots", 0.0)),
        "prompt_semantic_stage_inflight_peak": float(bundle_profile.get("prompt_semantic_stage_inflight_peak", 0.0)),
        "prompt_semantic_batch_size": float(bundle_profile.get("prompt_semantic_batch_size", 0.0)),
        "prompt_semantic_batch_samples": float(bundle_profile.get("prompt_semantic_batch_samples", 0.0)),
        "prompt_semantic_padded_batch_samples": float(bundle_profile.get("prompt_semantic_padded_batch_samples", 0.0)),
        "prompt_semantic_batch_pad_ratio": float(bundle_profile.get("prompt_semantic_batch_pad_ratio", 0.0)),
        "prompt_semantic_pool_bucket_index": float(bundle_profile.get("prompt_semantic_pool_bucket_index", 0.0)),
        "prompt_semantic_pool_workers": float(bundle_profile.get("prompt_semantic_pool_workers", 0.0)),
        "prompt_semantic_bucket_first_hit_serialized": float(
            bundle_profile.get("prompt_semantic_bucket_first_hit_serialized", 0.0)
        ),
        "prompt_semantic_runtime_exact_prewarm_applied": float(
            bundle_profile.get("prompt_semantic_runtime_exact_prewarm_applied", 0.0)
        ),
        "prompt_semantic_runtime_exact_prewarm_ms": float(
            bundle_profile.get("prompt_semantic_runtime_exact_prewarm_ms", 0.0)
        ),
        "prompt_semantic_runtime_exact_prewarm_target_samples": float(
            bundle_profile.get("prompt_semantic_runtime_exact_prewarm_target_samples", 0.0)
        ),
        "prompt_semantic_runtime_exact_prewarm_batch_sizes": float(
            bundle_profile.get("prompt_semantic_runtime_exact_prewarm_batch_sizes", 0.0)
        ),
        "prompt_semantic_runtime_exact_prewarm_skipped_capacity": float(
            bundle_profile.get("prompt_semantic_runtime_exact_prewarm_skipped_capacity", 0.0)
        ),
        "prompt_semantic_shard_index": float(bundle_profile.get("prompt_semantic_shard_index", 0.0)),
        "ref_spec_wait_ms": float(bundle_profile.get("ref_spec_wait_ms", 0.0)),
        "ref_spec_ms": ref_spec_ms,
        "ref_spec_to_device_ms": float(bundle_profile.get("ref_spec_to_device_ms", 0.0)),
        "ref_spec_main_resample_ms": float(bundle_profile.get("ref_spec_main_resample_ms", 0.0)),
        "ref_spec_norm_ms": float(bundle_profile.get("ref_spec_norm_ms", 0.0)),
        "ref_spec_spectrogram_ms": float(bundle_profile.get("ref_spec_spectrogram_ms", 0.0)),
        "ref_spec_post_resample_ms": float(bundle_profile.get("ref_spec_post_resample_ms", 0.0)),
        "ref_audio_bundle_ms": ref_audio_bundle_ms,
        "tensorize_ms": tensorize_ms,
        "total_ms": (time.perf_counter() - prepare_sync_start) * 1000.0,
        "wall_total_ms": (time.perf_counter() - prepare_start) * 1000.0,
    }
    if profile_overrides:
        prepare_profile.update({key: float(value) for key, value in profile_overrides.items()})
    return T2SRequestState(
        request_id=spec.request_id,
        ref_audio_path=spec.ref_audio_path,
        prompt_text=prompt_text,
        prompt_lang=spec.prompt_lang,
        text=text,
        text_lang=spec.text_lang,
        norm_prompt_text=prompt_result.norm_text,
        norm_text=target_result.norm_text,
        phones=phones_tensor,
        prompt_phones=prompt_phones_tensor,
        all_phones=all_phones,
        all_bert_features=all_bert_features,
        prompt_semantic=prompt_semantic,
        refer_spec=(None if spec_audio is None else (spec_audio, audio_16k)),
        aux_refer_specs=aux_refer_specs,
        raw_audio=raw_audio,
        raw_sr=raw_sr,
        top_k=spec.top_k,
        top_p=spec.top_p,
        temperature=spec.temperature,
        repetition_penalty=spec.repetition_penalty,
        early_stop_num=spec.early_stop_num,
        ready_step=spec.ready_step,
        prepare_profile=prepare_profile,
    )


_SCHEDULER_PREPARE_COORDINATOR_ATTR = "_scheduler_prepare_coordinator"
_SCHEDULER_PREPARE_COORDINATOR_LOCK_ATTR = "_scheduler_prepare_coordinator_lock"
_SCHEDULER_PROMPT_EXECUTOR_ATTR = "_scheduler_prompt_text_executor"
_SCHEDULER_PROMPT_EXECUTOR_LOCK_ATTR = "_scheduler_prompt_text_executor_lock"


def _get_scheduler_prepare_coordinator(tts: Any):
    coordinator = getattr(tts, _SCHEDULER_PREPARE_COORDINATOR_ATTR, None)
    generation = int(getattr(tts, "_vllm_runtime_prepare_generation", 0) or 0)
    cached_generation = int(getattr(tts, "_scheduler_prepare_coordinator_generation", -1) or -1)
    if coordinator is not None and cached_generation == generation:
        return coordinator
    coordinator = None
    if coordinator is not None:
        return coordinator
    lock = getattr(tts, _SCHEDULER_PREPARE_COORDINATOR_LOCK_ATTR, None)
    if lock is None:
        lock = threading.Lock()
        setattr(tts, _SCHEDULER_PREPARE_COORDINATOR_LOCK_ATTR, lock)
    with lock:
        coordinator = getattr(tts, _SCHEDULER_PREPARE_COORDINATOR_ATTR, None)
        cached_generation = int(getattr(tts, "_scheduler_prepare_coordinator_generation", -1) or -1)
        if coordinator is None or cached_generation != generation:
            from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import build_prepare_coordinator

            coordinator = build_prepare_coordinator(tts)
            setattr(tts, _SCHEDULER_PREPARE_COORDINATOR_ATTR, coordinator)
            setattr(tts, "_scheduler_prepare_coordinator_generation", generation)
        return coordinator


def _should_use_scheduler_prepare_coordinator(tts: Any) -> bool:
    if callable(getattr(tts, "_vllm_runtime_prepare_coordinator_factory", None)):
        return True
    return os.environ.get("GPTSOVITS_PREPARE_SCHEDULER_USE_COORDINATOR", "0") != "0"


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["value"] = exc

    thread = threading.Thread(target=_runner, name="prepare-request-state-sync", daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    return result.get("value")


def _get_scheduler_prompt_text_executor(tts: Any):
    executor = getattr(tts, _SCHEDULER_PROMPT_EXECUTOR_ATTR, None)
    if executor is not None:
        return executor
    lock = getattr(tts, _SCHEDULER_PROMPT_EXECUTOR_LOCK_ATTR, None)
    if lock is None:
        lock = threading.Lock()
        setattr(tts, _SCHEDULER_PROMPT_EXECUTOR_LOCK_ATTR, lock)
    with lock:
        executor = getattr(tts, _SCHEDULER_PROMPT_EXECUTOR_ATTR, None)
        if executor is None:
            worker_count = max(1, int(os.environ.get("GPTSOVITS_PREPARE_PROMPT_TEXT_WORKERS", "64")))
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="prepare-prompt-text",
            )
            setattr(tts, _SCHEDULER_PROMPT_EXECUTOR_ATTR, executor)
        return executor


def _run_profiled_prepare_text_features(
    tts: Any,
    text: str,
    language: str,
    submit_at: float,
) -> tuple[PreparedTextFeatures, float, float, float]:
    started_at = time.perf_counter()
    result = prepare_text_features(tts, text, language)
    finished_at = time.perf_counter()
    return result, float(submit_at), float(started_at), float(finished_at)


@torch.inference_mode()
def _prepare_request_state_legacy(
    tts: Any,
    spec: SchedulerRequestSpec,
) -> T2SRequestState:
    prepare_start = time.perf_counter()
    prepare_sync_start = time.perf_counter()
    prompt_text = normalize_sentence(spec.prompt_text, spec.prompt_lang)
    text = spec.text.strip("\n")
    prompt_result = None
    profile_overrides: Dict[str, float] = {}
    prompt_future = None
    prompt_async_enabled = (
        prompt_text not in [None, ""]
        and os.environ.get("GPTSOVITS_PREPARE_PROMPT_TEXT_ASYNC", "0") != "0"
    )
    if prompt_async_enabled:
        prompt_submit_at = time.perf_counter()
        prompt_future = _get_scheduler_prompt_text_executor(tts).submit(
            _run_profiled_prepare_text_features,
            tts,
            prompt_text,
            spec.prompt_lang,
            float(prompt_submit_at),
        )
    target_result = prepare_text_features(tts, text, spec.text_lang)
    target_done_at = time.perf_counter()
    if target_result.phones is None:
        raise ValueError(f"{spec.request_id} text preprocessing returned no phones")
    if prompt_text in [None, ""]:
        prompt_result = build_empty_text_features(
            feature_dim=int(target_result.bert_features.shape[0]),
            dtype=target_result.bert_features.dtype,
        )
    elif prompt_future is None:
        prompt_result = prepare_text_features(tts, prompt_text, spec.prompt_lang)
    ref_audio_bundle = tts.extract_ref_audio_bundle(str(spec.ref_audio_path))
    if prompt_future is not None:
        future_wait_start = time.perf_counter()
        prompt_result, prompt_submit_at, prompt_started_at, prompt_finished_at = prompt_future.result()
        future_wait_end = time.perf_counter()
        profile_overrides = {
            "prompt_text_parallel_future_wait_ms": max(0.0, (future_wait_end - future_wait_start) * 1000.0),
            "prompt_text_parallel_future_executor_queue_ms": max(
                0.0, (float(prompt_started_at) - float(prompt_submit_at)) * 1000.0
            ),
            "prompt_text_parallel_future_run_ms": max(
                0.0, (float(prompt_finished_at) - float(prompt_started_at)) * 1000.0
            ),
            "prompt_text_parallel_future_finish_after_submit_ms": max(
                0.0, (float(prompt_finished_at) - float(prompt_submit_at)) * 1000.0
            ),
            "prompt_text_parallel_future_queue_tail_after_target_ms": max(
                0.0, (float(prompt_started_at) - float(target_done_at)) * 1000.0
            ),
            "prompt_text_parallel_future_run_tail_after_target_ms": max(
                0.0, (float(prompt_finished_at) - max(float(prompt_started_at), float(target_done_at))) * 1000.0
            ),
        }
    assert prompt_result is not None
    return build_request_state_from_parts(
        tts=tts,
        spec=spec,
        prompt_text=prompt_text,
        text=text,
        prompt_result=prompt_result,
        target_result=target_result,
        ref_audio_bundle=ref_audio_bundle,
        prepare_start=prepare_start,
        prepare_sync_start=prepare_sync_start,
        profile_overrides=profile_overrides,
    )


@torch.inference_mode()
def prepare_request_state(
    tts: Any,
    spec: SchedulerRequestSpec,
) -> T2SRequestState:
    if not _should_use_scheduler_prepare_coordinator(tts):
        return _prepare_request_state_legacy(tts, spec)
    coordinator = _get_scheduler_prepare_coordinator(tts)
    state, _, _ = _run_coro_sync(coordinator.prepare_state_profiled_async(spec, time.perf_counter()))
    return state


def _left_pad_hidden(hidden: torch.Tensor, target_len: int) -> torch.Tensor:
    if hidden.shape[0] >= target_len:
        return hidden
    return F.pad(hidden, (0, 0, target_len - hidden.shape[0], 0), value=0)


def _ensure_audio_pe(model: Any, max_position: int, dtype: torch.dtype, device: torch.device) -> None:
    required_len = max_position + 1
    if model.ar_audio_position.pe is not None and model.ar_audio_position.pe.size(1) >= required_len:
        if model.ar_audio_position.pe.dtype != dtype or model.ar_audio_position.pe.device != device:
            model.ar_audio_position.pe = model.ar_audio_position.pe.to(dtype=dtype, device=device)
        return
    model.ar_audio_position.extend_pe(
        torch.zeros(1, required_len, model.ar_audio_position.embedding_dim, device=device, dtype=dtype)
    )


def _pad_token_sequences(
    token_sequences: Sequence[torch.LongTensor],
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    if not token_sequences:
        raise ValueError("token_sequences 不能为空")
    device = token_sequences[0].device
    max_len = max(int(sequence.shape[0]) for sequence in token_sequences)
    padded = torch.zeros((len(token_sequences), max_len), dtype=token_sequences[0].dtype, device=device)
    mask = torch.zeros((len(token_sequences), max_len), dtype=torch.bool, device=device)
    for row_index, sequence in enumerate(token_sequences):
        seq_len = int(sequence.shape[0])
        padded[row_index, :seq_len] = sequence
        mask[row_index, :seq_len] = True
    return padded, mask


def _stack_token_sequences_if_same_length(
    token_sequences: Sequence[torch.LongTensor],
) -> Optional[torch.LongTensor]:
    if not token_sequences:
        raise ValueError("token_sequences 不能为空")
    target_len = int(token_sequences[0].shape[0])
    for sequence in token_sequences[1:]:
        if int(sequence.shape[0]) != target_len:
            return None
    return torch.stack(list(token_sequences), dim=0)


def _sampling_group_key(
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    trim_eos: bool,
) -> Tuple[int, float, float, float, bool]:
    return (
        int(top_k),
        float(top_p),
        float(temperature),
        float(repetition_penalty),
        bool(trim_eos),
    )


def _iter_contiguous_sampling_groups(
    sampling_keys: Sequence[Tuple[int, float, float, float, bool]],
) -> List[Tuple[Tuple[int, float, float, float, bool], List[int]]]:
    groups: List[Tuple[Tuple[int, float, float, float, bool], List[int]]] = []
    if not sampling_keys:
        return groups
    current_key = sampling_keys[0]
    current_indices: List[int] = [0]
    for index in range(1, len(sampling_keys)):
        key = sampling_keys[index]
        if key == current_key:
            current_indices.append(index)
            continue
        groups.append((current_key, current_indices))
        current_key = key
        current_indices = [index]
    groups.append((current_key, current_indices))
    return groups


def _uniform_sampling_group_key(active_batch: T2SActiveBatch) -> Optional[Tuple[int, float, float, float, bool]]:
    if not active_batch.states:
        return None
    if active_batch.step_indices.numel() <= 0:
        return None
    first_step_index = int(active_batch.step_indices[0].item())
    if bool((active_batch.step_indices != first_step_index).any().item()):
        return None
    first_state = active_batch.states[0]
    first_key = _sampling_group_key(
        top_k=first_state.top_k,
        top_p=first_state.top_p,
        temperature=first_state.temperature,
        repetition_penalty=first_state.repetition_penalty,
        trim_eos=first_step_index < 11,
    )
    for state in active_batch.states[1:]:
        if (
            state.top_k != first_state.top_k
            or state.top_p != first_state.top_p
            or state.temperature != first_state.temperature
            or state.repetition_penalty != first_state.repetition_penalty
        ):
            return None
    return first_key


def _batched_sample_uniform(
    logits: torch.Tensor,
    histories: Sequence[torch.LongTensor],
    sampling_key: Tuple[int, float, float, float, bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    top_k, top_p, temperature, repetition_penalty, trim_eos = sampling_key
    sample_logits = logits[:, :-1] if trim_eos else logits
    padded_histories = _stack_token_sequences_if_same_length(histories)
    history_mask = None
    if padded_histories is None:
        padded_histories, history_mask = _pad_token_sequences(histories)
    probs = logits_to_probs(
        logits=sample_logits,
        previous_tokens=padded_histories,
        previous_token_mask=history_mask,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
    )
    sampled = multinomial_sample_one_no_sync(probs)
    argmax_tokens = torch.argmax(sample_logits, dim=-1)
    return sampled, argmax_tokens


def _batched_sample_by_group(
    logits: torch.Tensor,
    histories: Sequence[torch.LongTensor],
    sampling_keys: Sequence[Tuple[int, float, float, float, bool]],
) -> Tuple[List[torch.Tensor], List[int]]:
    sampled_list: List[Optional[torch.Tensor]] = [None] * len(histories)
    argmax_list: List[Optional[int]] = [None] * len(histories)
    for group_key, group_indices in _iter_contiguous_sampling_groups(sampling_keys):
        top_k, top_p, temperature, repetition_penalty, trim_eos = group_key
        index_tensor = torch.tensor(group_indices, dtype=torch.long, device=logits.device)
        group_logits = torch.index_select(logits, dim=0, index=index_tensor)
        if trim_eos:
            group_logits = group_logits[:, :-1]
        group_histories = [histories[index] for index in group_indices]
        padded_histories = _stack_token_sequences_if_same_length(group_histories)
        history_mask = None
        if padded_histories is None:
            padded_histories, history_mask = _pad_token_sequences(group_histories)
        probs = logits_to_probs(
            logits=group_logits,
            previous_tokens=padded_histories,
            previous_token_mask=history_mask,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )
        argmax_tokens = torch.argmax(group_logits, dim=-1)
        for local_index, global_index in enumerate(group_indices):
            sampled_list[global_index] = multinomial_sample_one_no_sync(probs[local_index : local_index + 1])
            argmax_list[global_index] = int(argmax_tokens[local_index].item())

    return [item for item in sampled_list if item is not None], [int(item) for item in argmax_list if item is not None]


@torch.inference_mode()
def build_prefill_batch(model: Any, states: Sequence[T2SRequestState]) -> T2SActiveBatch:
    x_items: List[torch.Tensor] = []
    y_pos_items: List[torch.Tensor] = []
    x_lens: List[int] = []
    prefix_lens: List[int] = []
    y_sequences: List[torch.LongTensor] = []

    for state in states:
        text_emb = model.ar_text_embedding(state.all_phones.unsqueeze(0))
        bert_proj = model.bert_proj(state.all_bert_features.transpose(0, 1).unsqueeze(0))
        x_pos = model.ar_text_position(text_emb + bert_proj).squeeze(0)
        y_emb = model.ar_audio_embedding(state.prompt_semantic.unsqueeze(0))
        y_pos = model.ar_audio_position(y_emb).squeeze(0)
        x_items.append(x_pos)
        y_pos_items.append(y_pos)
        x_lens.append(x_pos.shape[0])
        prefix_lens.append(y_pos.shape[0])
        y_sequences.append(state.prompt_semantic.clone())

    max_x_len = max(x_lens)
    max_prefix_len = max(prefix_lens)
    x_batch = torch.stack([_left_pad_hidden(item, max_x_len) for item in x_items], dim=0)
    y_pos_batch = torch.stack([_left_pad_hidden(item, max_prefix_len) for item in y_pos_items], dim=0)
    xy_pos = torch.cat([x_batch, y_pos_batch], dim=1)

    device = x_batch.device
    x_lens_tensor = torch.LongTensor(x_lens).to(device)
    prefix_lens_tensor = torch.LongTensor(prefix_lens).to(device)
    src_len = max_x_len + max_prefix_len

    x_padding_mask = make_pad_mask_left(x_lens_tensor, max_x_len)
    y_padding_mask = make_pad_mask_left(prefix_lens_tensor, max_prefix_len)
    key_padding_mask = torch.cat([x_padding_mask, y_padding_mask], dim=1).bool()
    x_mask = F.pad(torch.zeros(max_x_len, max_x_len, dtype=torch.bool, device=device), (0, max_prefix_len), value=True)
    y_mask = F.pad(
        torch.triu(torch.ones(max_prefix_len, max_prefix_len, dtype=torch.bool, device=device), diagonal=1),
        (max_x_len, 0),
        value=False,
    )
    causal_mask = torch.cat([x_mask, y_mask], dim=0).unsqueeze(0)
    attn_mask = causal_mask.logical_or(key_padding_mask.unsqueeze(1)).unsqueeze(1)

    return T2SActiveBatch(
        request_ids=[state.request_id for state in states],
        states=list(states),
        x=x_batch,
        x_lens=x_lens_tensor,
        y_sequences=y_sequences,
        prefix_lens=prefix_lens_tensor,
        xy_pos=xy_pos,
        key_padding_mask=key_padding_mask,
        prefill_attn_mask=attn_mask,
        decode_attn_mask=None,
        k_cache=None,
        v_cache=None,
        kv_lens=None,
        step_indices=torch.zeros((len(states),), dtype=torch.long, device=device),
        prefill_done=False,
        kv_cache_pooled=False,
        kv_cache_capacity=0,
        kv_cache_batch_capacity=0,
    )


def build_next_xy_pos(model: Any, y_sequences: Sequence[torch.LongTensor]) -> torch.Tensor:
    last_tokens = torch.stack([seq[-1:] for seq in y_sequences], dim=0)
    y_emb = model.ar_audio_embedding(last_tokens)
    position_ids = torch.LongTensor([int(seq.shape[0] - 1) for seq in y_sequences]).to(y_emb.device)
    _ensure_audio_pe(model, int(position_ids.max().item()), y_emb.dtype, y_emb.device)
    pos_emb = model.ar_audio_position.pe[0].index_select(0, position_ids).unsqueeze(1)
    return y_emb * model.ar_audio_position.x_scale + model.ar_audio_position.alpha * pos_emb.to(
        dtype=y_emb.dtype, device=y_emb.device
    )


def _get_kv_pool(model: Any) -> Any | None:
    pool = getattr(model, "kv_cache_pool", None)
    if pool is None:
        return None
    if not getattr(getattr(pool, "state", None), "enabled", False):
        return None
    return pool


def _set_kv_pool_active_rows(model: Any, active_rows: int) -> None:
    pool = _get_kv_pool(model)
    if pool is None:
        return
    try:
        pool.set_active_rows(active_rows)
    except Exception:
        pass


def _pack_active_batch_into_pool(model: Any, active_batch: T2SActiveBatch) -> bool:
    pool = _get_kv_pool(model)
    if (
        pool is None
        or active_batch.k_cache is None
        or active_batch.v_cache is None
        or active_batch.kv_lens is None
        or active_batch.kv_lens.numel() <= 0
    ):
        return False
    batch_size = int(active_batch.kv_lens.shape[0])
    max_kv_len = int(active_batch.kv_lens.max().item())
    if max_kv_len + 1 > int(pool.max_seq_len):
        try:
            pool.record_fallback(f"pack_decode_headroom_overflow(batch={batch_size},seq={max_kv_len},next={max_kv_len + 1})")
        except Exception:
            pass
        active_batch.kv_cache_pooled = False
        active_batch.kv_cache_capacity = 0
        active_batch.kv_cache_batch_capacity = 0
        _set_kv_pool_active_rows(model, 0)
        return False
    pooled_views = pool.pack_dynamic_cache_layers(
        k_layers=active_batch.k_cache,
        v_layers=active_batch.v_cache,
        kv_lens=active_batch.kv_lens,
    )
    if pooled_views is None:
        active_batch.kv_cache_pooled = False
        active_batch.kv_cache_capacity = 0
        active_batch.kv_cache_batch_capacity = 0
        _set_kv_pool_active_rows(model, 0)
        return False
    active_batch.k_cache, active_batch.v_cache = pooled_views
    active_batch.decode_attn_mask = None
    active_batch.kv_cache_pooled = True
    active_batch.kv_cache_capacity = int(pool.max_seq_len)
    active_batch.kv_cache_batch_capacity = int(pool.max_batch_size)
    _set_kv_pool_active_rows(model, len(active_batch.request_ids))
    return True


def _build_decode_mask_from_kv_lens(
    kv_lens: torch.LongTensor,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if kv_lens.numel() <= 0:
        return None
    target_len = int(kv_lens.max().item()) + 1
    mask = torch.ones((int(kv_lens.shape[0]), 1, 1, target_len), dtype=torch.bool, device=device)
    for batch_index, kv_len in enumerate(kv_lens.tolist()):
        current_len = kv_len + 1
        mask[batch_index, :, :, -current_len:] = False
    if not mask.any().item():
        return None
    return mask


def _compact_pooled_active_batch(model: Any, active_batch: T2SActiveBatch, keep_indices: Sequence[int]) -> bool:
    pool = _get_kv_pool(model)
    if pool is None or active_batch.kv_lens is None:
        return False
    pooled_views = pool.compact_rows(keep_indices=keep_indices, kv_lens=active_batch.kv_lens)
    if pooled_views is None:
        active_batch.kv_cache_pooled = False
        active_batch.kv_cache_capacity = 0
        active_batch.kv_cache_batch_capacity = 0
        _set_kv_pool_active_rows(model, 0)
        return False
    active_batch.k_cache, active_batch.v_cache = pooled_views
    active_batch.decode_attn_mask = None
    active_batch.kv_cache_pooled = True
    active_batch.kv_cache_capacity = int(pool.max_seq_len)
    active_batch.kv_cache_batch_capacity = int(pool.max_batch_size)
    _set_kv_pool_active_rows(model, len(active_batch.request_ids))
    return True


def _fallback_pooled_active_batch_to_dynamic_cache(
    model: Any,
    active_batch: T2SActiveBatch,
    *,
    reason: str,
) -> None:
    if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
        raise ValueError("pooled KV fallback 缺少 KV cache 或 kv_lens")
    pool = _get_kv_pool(model)
    if pool is not None:
        try:
            pool.record_fallback(reason)
        except Exception:
            pass
    target_len = int(active_batch.kv_lens.max().item())
    kv_lens_list = [int(item) for item in active_batch.kv_lens.tolist()]
    unpacked_k_cache: List[torch.Tensor] = []
    unpacked_v_cache: List[torch.Tensor] = []
    for pooled_k, pooled_v in zip(active_batch.k_cache, active_batch.v_cache):
        unpacked_k = pooled_k.new_zeros((pooled_k.shape[0], target_len, pooled_k.shape[2]))
        unpacked_v = pooled_v.new_zeros((pooled_v.shape[0], target_len, pooled_v.shape[2]))
        for batch_index, kv_len in enumerate(kv_lens_list):
            if kv_len <= 0:
                continue
            unpacked_k[batch_index, -kv_len:, :] = pooled_k[batch_index, :kv_len, :]
            unpacked_v[batch_index, -kv_len:, :] = pooled_v[batch_index, :kv_len, :]
        unpacked_k_cache.append(unpacked_k)
        unpacked_v_cache.append(unpacked_v)
    active_batch.k_cache = unpacked_k_cache
    active_batch.v_cache = unpacked_v_cache
    active_batch.decode_attn_mask = _build_decode_mask_from_kv_lens(
        active_batch.kv_lens,
        device=active_batch.k_cache[0].device,
    )
    active_batch.kv_cache_pooled = False
    active_batch.kv_cache_capacity = 0
    active_batch.kv_cache_batch_capacity = 0
    _set_kv_pool_active_rows(model, 0)


def _compact_cache_to_kv_lens(
    cache: torch.Tensor,
    kv_lens: torch.LongTensor,
) -> torch.Tensor:
    target_len = int(kv_lens.max().item())
    if cache.shape[1] == target_len and torch.all(kv_lens == target_len).item():
        return cache
    compacted = cache.new_zeros((cache.shape[0], target_len, cache.shape[2]))
    for batch_index, kv_len in enumerate(kv_lens.tolist()):
        if kv_len <= 0:
            continue
        compacted[batch_index, -kv_len:, :] = cache[batch_index, -kv_len:, :]
    return compacted


def _compact_decode_mask_to_kv_lens(
    decode_attn_mask: Optional[torch.Tensor],
    kv_lens: torch.LongTensor,
) -> Optional[torch.Tensor]:
    target_len = int(kv_lens.max().item()) + 1
    if decode_attn_mask is None:
        return None
    if decode_attn_mask.shape[-1] == target_len and torch.all(kv_lens + 1 == target_len).item():
        return decode_attn_mask
    compacted = torch.ones(
        (decode_attn_mask.shape[0], 1, 1, target_len),
        dtype=decode_attn_mask.dtype,
        device=decode_attn_mask.device,
    )
    for batch_index, kv_len in enumerate(kv_lens.tolist()):
        current_len = kv_len + 1
        compacted[batch_index, :, :, -current_len:] = decode_attn_mask[batch_index, :, :, -current_len:]
    if not compacted.any().item():
        return None
    return compacted


def _advance_decode_mask(
    decode_attn_mask: Optional[torch.Tensor],
    kv_lens: torch.LongTensor,
) -> Optional[torch.Tensor]:
    if decode_attn_mask is None:
        return None
    target_len = int(kv_lens.max().item()) + 2
    advanced = torch.zeros(
        (decode_attn_mask.shape[0], 1, 1, target_len),
        dtype=decode_attn_mask.dtype,
        device=decode_attn_mask.device,
    )
    for batch_index, kv_len in enumerate(kv_lens.tolist()):
        current_len = kv_len + 1
        next_mask = F.pad(decode_attn_mask[batch_index : batch_index + 1, :, :, -current_len:], (0, 1), value=False)
        advanced[batch_index : batch_index + 1, :, :, -next_mask.shape[-1] :] = next_mask
    if not advanced.any().item():
        return None
    return advanced


def _sample_per_request(
    model: Any,
    active_batch: T2SActiveBatch,
    logits: torch.Tensor,
    max_steps: int,
) -> Tuple[List[T2SFinishedItem], List[int], List[torch.LongTensor]]:
    finished_items: List[T2SFinishedItem] = []
    keep_indices: List[int] = []
    updated_sequences: List[torch.LongTensor] = []

    uniform_sampling_key = _uniform_sampling_group_key(active_batch)
    sampled_items: List[torch.Tensor]
    argmax_tokens: List[int]
    sampled_token_tensor: Optional[torch.Tensor] = None
    argmax_token_tensor: Optional[torch.Tensor] = None
    if uniform_sampling_key is not None:
        sampled_tensor, argmax_tensor = _batched_sample_uniform(
            logits=logits,
            histories=active_batch.y_sequences,
            sampling_key=uniform_sampling_key,
        )
        sampled_token_tensor = sampled_tensor.view(-1)
        argmax_token_tensor = argmax_tensor.view(-1)
        stacked_histories = _stack_token_sequences_if_same_length(active_batch.y_sequences)
        if (
            all(state.early_stop_num == -1 for state in active_batch.states)
            and int(active_batch.step_indices[0].item()) + 1 < max_steps
            and not bool(sampled_token_tensor.eq(model.EOS).any().item())
            and not bool(argmax_token_tensor.eq(model.EOS).any().item())
        ):
            return (
                [],
                list(range(len(active_batch.states))),
                list(torch.cat([stacked_histories, sampled_token_tensor.view(-1, 1)], dim=1).unbind(0))
                if stacked_histories is not None
                else [torch.cat([history, sampled_token_tensor[index : index + 1]], dim=0) for index, history in enumerate(active_batch.y_sequences)],
            )
        sampled_items = [sampled_tensor[index : index + 1] for index in range(sampled_tensor.shape[0])]
        argmax_tokens = [int(item) for item in argmax_tensor.tolist()]
    else:
        sampling_keys = [
            _sampling_group_key(
                top_k=state.top_k,
                top_p=state.top_p,
                temperature=state.temperature,
                repetition_penalty=state.repetition_penalty,
                trim_eos=int(active_batch.step_indices[batch_index].item()) < 11,
            )
            for batch_index, state in enumerate(active_batch.states)
        ]
        sampled_items, argmax_tokens = _batched_sample_by_group(
            logits=logits,
            histories=active_batch.y_sequences,
            sampling_keys=sampling_keys,
        )
    for batch_index, state in enumerate(active_batch.states):
        step_index = int(active_batch.step_indices[batch_index].item())
        current_history = active_batch.y_sequences[batch_index]
        if sampled_token_tensor is not None and argmax_token_tensor is not None:
            sampled = sampled_token_tensor[batch_index : batch_index + 1]
            sampled_token = int(sampled_token_tensor[batch_index].item())
            argmax_token = int(argmax_token_tensor[batch_index].item())
        else:
            sampled = sampled_items[batch_index]
            sampled_token = int(sampled[0, 0].item())
            argmax_token = argmax_tokens[batch_index]
        new_history = torch.cat([current_history, sampled.view(-1)], dim=0)

        finish_reason: Optional[str] = None
        if state.early_stop_num != -1 and (new_history.shape[0] - int(active_batch.prefix_lens[batch_index].item())) > state.early_stop_num:
            finish_reason = "early_stop"
        elif step_index + 1 >= max_steps:
            finish_reason = "max_step"
        elif sampled_token == model.EOS:
            finish_reason = "eos_sample"
        elif argmax_token == model.EOS:
            finish_reason = "eos_argmax"

        if finish_reason is not None:
            prefix_len = int(active_batch.prefix_lens[batch_index].item())
            finished_items.append(
                T2SFinishedItem(
                    request_id=state.request_id,
                    semantic_tokens=new_history[prefix_len:-1].clone(),
                    finish_idx=step_index,
                    finish_reason=finish_reason,
                )
            )
        else:
            keep_indices.append(batch_index)
            updated_sequences.append(new_history)
    return finished_items, keep_indices, updated_sequences


@torch.inference_mode()
def decode_one_step(
    model: Any,
    active_batch: T2SActiveBatch,
    max_steps: int,
) -> Tuple[Optional[T2SActiveBatch], List[T2SFinishedItem]]:
    was_prefill = not active_batch.prefill_done
    if was_prefill:
        if active_batch.prefill_attn_mask is None or active_batch.key_padding_mask is None:
            raise ValueError("prefill 阶段缺少必要 mask")
        xy_dec, active_batch.k_cache, active_batch.v_cache = model.t2s_transformer.process_prompt(
            active_batch.xy_pos, active_batch.prefill_attn_mask, None
        )
        active_batch.kv_lens = active_batch.x_lens + active_batch.prefix_lens
        if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
            raise ValueError("prefill 阶段未生成完整 KV cache")
        if not _pack_active_batch_into_pool(model, active_batch):
            active_batch.decode_attn_mask = F.pad(
                active_batch.key_padding_mask.unsqueeze(1).unsqueeze(1),
                (0, 1),
                value=False,
            )
            active_batch.k_cache = [_compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.k_cache]
            active_batch.v_cache = [_compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.v_cache]
            active_batch.decode_attn_mask = _compact_decode_mask_to_kv_lens(active_batch.decode_attn_mask, active_batch.kv_lens)
        active_batch.x = None
        active_batch.x_lens = None
        active_batch.key_padding_mask = None
        active_batch.prefill_attn_mask = None
        active_batch.prefill_done = True
    else:
        if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
            raise ValueError("decode 阶段缺少 KV cache")
        if active_batch.kv_cache_pooled:
            pool = _get_kv_pool(model)
            if pool is None:
                raise ValueError("pooled KV cache 已丢失")
            next_kv_lens = active_batch.kv_lens + 1
            if int(next_kv_lens.max().item()) > int(pool.max_seq_len):
                _fallback_pooled_active_batch_to_dynamic_cache(
                    model,
                    active_batch,
                    reason=(
                        "decode_headroom_overflow"
                        f"(batch={int(next_kv_lens.shape[0])},seq={int(active_batch.kv_lens.max().item())},"
                        f"next={int(next_kv_lens.max().item())})"
                    ),
                )
            else:
                batched_decode_attn_mask = pool.build_decode_mask(next_kv_lens)
                xy_dec, active_batch.k_cache, active_batch.v_cache = model.decode_next_token_prealloc_runtime(
                    active_batch.xy_pos,
                    active_batch.k_cache,
                    active_batch.v_cache,
                    active_batch.kv_lens,
                    batched_decode_attn_mask,
                )
                logits = model.ar_predict_layer(xy_dec[:, -1])
                finished_items, keep_indices, updated_sequences = _sample_per_request(
                    model,
                    active_batch,
                    logits,
                    max_steps=max_steps,
                )
                if len(keep_indices) == 0:
                    _set_kv_pool_active_rows(model, 0)
                    return None, finished_items
                if len(keep_indices) == len(active_batch.request_ids):
                    active_batch.y_sequences = updated_sequences
                    active_batch.step_indices = active_batch.step_indices + 1
                    active_batch.kv_lens = active_batch.kv_lens + 1
                    active_batch.xy_pos = build_next_xy_pos(model, active_batch.y_sequences)
                    return active_batch, finished_items
                device = logits.device
                keep_tensor = torch.LongTensor(keep_indices).to(device)
                active_batch.request_ids = [active_batch.request_ids[i] for i in keep_indices]
                active_batch.states = [active_batch.states[i] for i in keep_indices]
                active_batch.y_sequences = updated_sequences
                active_batch.prefix_lens = torch.index_select(active_batch.prefix_lens, dim=0, index=keep_tensor)
                next_step_indices = torch.index_select(active_batch.step_indices, dim=0, index=keep_tensor)
                next_kv_lens = torch.index_select(active_batch.kv_lens, dim=0, index=keep_tensor) + 1
                active_batch.step_indices = next_step_indices + 1
                active_batch.kv_lens = next_kv_lens
                pooled_compacted = _compact_pooled_active_batch(model, active_batch, keep_indices)
                if not pooled_compacted:
                    active_batch.kv_cache_pooled = False
                if (not active_batch.kv_cache_pooled) and active_batch.k_cache is not None and active_batch.v_cache is not None:
                    for cache_index in range(len(active_batch.k_cache)):
                        active_batch.k_cache[cache_index] = torch.index_select(
                            active_batch.k_cache[cache_index],
                            dim=0,
                            index=keep_tensor,
                        )
                        active_batch.v_cache[cache_index] = torch.index_select(
                            active_batch.v_cache[cache_index],
                            dim=0,
                            index=keep_tensor,
                        )
                    active_batch.k_cache = [
                        _compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.k_cache
                    ]
                    active_batch.v_cache = [
                        _compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.v_cache
                    ]
                    active_batch.decode_attn_mask = _build_decode_mask_from_kv_lens(
                        active_batch.kv_lens,
                        device=active_batch.k_cache[0].device,
                    )
                active_batch.xy_pos = build_next_xy_pos(model, active_batch.y_sequences)
                return active_batch, finished_items
        batched_decode_attn_mask = None
        if active_batch.decode_attn_mask is not None:
            batched_decode_attn_mask = _materialize_decode_mask_for_active_batch(active_batch)
            if not batched_decode_attn_mask.any().item():
                batched_decode_attn_mask = None
        xy_dec, active_batch.k_cache, active_batch.v_cache = model.t2s_transformer.decode_next_token(
            active_batch.xy_pos,
            active_batch.k_cache,
            active_batch.v_cache,
            batched_decode_attn_mask,
        )
        active_batch.decode_attn_mask = _advance_decode_mask(active_batch.decode_attn_mask, active_batch.kv_lens)

    logits = model.ar_predict_layer(xy_dec[:, -1])

    finished_items, keep_indices, updated_sequences = _sample_per_request(
        model,
        active_batch,
        logits,
        max_steps=max_steps,
    )
    if len(keep_indices) == 0:
        if active_batch.kv_cache_pooled:
            _set_kv_pool_active_rows(model, 0)
        return None, finished_items
    if len(keep_indices) == len(active_batch.request_ids):
        active_batch.y_sequences = updated_sequences
        active_batch.step_indices = active_batch.step_indices + 1
        if not was_prefill and active_batch.kv_lens is not None:
            active_batch.kv_lens = active_batch.kv_lens + 1
        active_batch.xy_pos = build_next_xy_pos(model, active_batch.y_sequences)
        return active_batch, finished_items

    device = logits.device
    keep_tensor = torch.LongTensor(keep_indices).to(device)
    active_batch.request_ids = [active_batch.request_ids[i] for i in keep_indices]
    active_batch.states = [active_batch.states[i] for i in keep_indices]
    active_batch.y_sequences = updated_sequences
    active_batch.prefix_lens = torch.index_select(active_batch.prefix_lens, dim=0, index=keep_tensor)
    next_step_indices = torch.index_select(active_batch.step_indices, dim=0, index=keep_tensor)
    next_kv_lens = None if active_batch.kv_lens is None else torch.index_select(active_batch.kv_lens, dim=0, index=keep_tensor)
    active_batch.step_indices = next_step_indices + 1
    if not was_prefill:
        if next_kv_lens is not None:
            active_batch.kv_lens = next_kv_lens + 1
    else:
        active_batch.kv_lens = next_kv_lens

    if active_batch.decode_attn_mask is not None:
        active_batch.decode_attn_mask = torch.index_select(active_batch.decode_attn_mask, dim=0, index=keep_tensor)
        if not active_batch.decode_attn_mask.any().item():
            active_batch.decode_attn_mask = None
    if active_batch.kv_cache_pooled:
        pooled_compacted = _compact_pooled_active_batch(model, active_batch, keep_indices)
        if not pooled_compacted:
            active_batch.kv_cache_pooled = False
    if (not active_batch.kv_cache_pooled) and active_batch.k_cache is not None and active_batch.v_cache is not None:
        for cache_index in range(len(active_batch.k_cache)):
            active_batch.k_cache[cache_index] = torch.index_select(active_batch.k_cache[cache_index], dim=0, index=keep_tensor)
            active_batch.v_cache[cache_index] = torch.index_select(active_batch.v_cache[cache_index], dim=0, index=keep_tensor)
        if active_batch.kv_lens is not None:
            active_batch.k_cache = [_compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.k_cache]
            active_batch.v_cache = [_compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.v_cache]
            active_batch.decode_attn_mask = _compact_decode_mask_to_kv_lens(
                active_batch.decode_attn_mask,
                active_batch.kv_lens,
            )

    active_batch.xy_pos = build_next_xy_pos(model, active_batch.y_sequences)
    return active_batch, finished_items


def run_scheduler_batch(
    model: Any,
    states: Sequence[T2SRequestState],
    max_steps: int,
) -> List[T2SFinishedItem]:
    return run_scheduler_continuous(model, states, max_steps=max_steps)


def _pad_cache_left(cache: torch.Tensor, target_len: int) -> torch.Tensor:
    pad_len = target_len - cache.shape[1]
    if pad_len <= 0:
        return cache
    return F.pad(cache, (0, 0, pad_len, 0), value=0)


def _extract_cache_row(
    cache: torch.Tensor,
    batch_index: int,
    kv_len: int,
    *,
    pooled: bool,
) -> torch.Tensor:
    if kv_len <= 0:
        return cache[batch_index : batch_index + 1, :0, :]
    if pooled:
        return cache[batch_index : batch_index + 1, :kv_len, :]
    return cache[batch_index : batch_index + 1, -kv_len:, :]


def _pad_decode_mask_left(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    pad_len = target_len - mask.shape[-1]
    if pad_len <= 0:
        return mask
    return F.pad(mask, (pad_len, 0), value=True)


def _fit_decode_mask_length(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if mask.shape[-1] > target_len:
        return mask[:, :, :, -target_len:]
    if mask.shape[-1] < target_len:
        return _pad_decode_mask_left(mask, target_len)
    return mask


def _materialize_decode_mask_for_request(running_request: T2SRunningRequest) -> torch.Tensor:
    expected_mask_len = running_request.k_cache[0].shape[1] + 1
    if running_request.decode_attn_mask is not None:
        return _fit_decode_mask_length(running_request.decode_attn_mask, expected_mask_len)
    current_mask_len = running_request.k_cache[0].shape[1] + 1
    return torch.zeros(
        (1, 1, 1, current_mask_len),
        dtype=torch.bool,
        device=running_request.k_cache[0].device,
    )


def _materialize_decode_mask_for_active_batch(
    active_batch: T2SActiveBatch,
    target_mask_len: Optional[int] = None,
) -> torch.Tensor:
    if active_batch.k_cache is None or active_batch.kv_lens is None:
        raise ValueError("active batch 缺少 KV cache 或 kv_lens")
    current_mask_len = active_batch.k_cache[0].shape[1] + 1
    if target_mask_len is None:
        target_mask_len = current_mask_len
    if active_batch.decode_attn_mask is None:
        mask = torch.zeros(
            (len(active_batch.request_ids), 1, 1, current_mask_len),
            dtype=torch.bool,
            device=active_batch.k_cache[0].device,
        )
    else:
        rows: List[torch.Tensor] = []
        for batch_index, kv_len in enumerate(active_batch.kv_lens.tolist()):
            row_len = kv_len + 1
            row_mask = _fit_decode_mask_length(
                active_batch.decode_attn_mask[batch_index : batch_index + 1],
                row_len,
            )
            rows.append(_pad_decode_mask_left(row_mask, target_mask_len))
        mask = torch.cat(rows, dim=0)
    if target_mask_len != current_mask_len and active_batch.decode_attn_mask is None:
        mask = _pad_decode_mask_left(mask, target_mask_len)
    return mask


@torch.inference_mode()
def run_prefill_active_batch(
    model: Any,
    states: Sequence[T2SRequestState],
    max_steps: int,
) -> Tuple[Optional[T2SActiveBatch], List[T2SFinishedItem]]:
    if not states:
        return None, []
    active_batch = build_prefill_batch(model, states)
    return decode_one_step(model, active_batch, max_steps=max_steps)


@torch.inference_mode()
def merge_active_batches(
    model: Any,
    left_batch: Optional[T2SActiveBatch],
    right_batch: Optional[T2SActiveBatch],
) -> Optional[T2SActiveBatch]:
    if left_batch is None:
        return right_batch
    if right_batch is None:
        return left_batch
    if not left_batch.prefill_done or not right_batch.prefill_done:
        raise ValueError("只有 prefill 完成后的 active batch 才能 merge")
    if left_batch.k_cache is None or left_batch.v_cache is None or right_batch.k_cache is None or right_batch.v_cache is None:
        raise ValueError("merge active batch 时缺少 KV cache")
    if left_batch.kv_lens is None or right_batch.kv_lens is None:
        raise ValueError("merge active batch 时缺少 kv_lens")
    merged_kv_lens = torch.cat([left_batch.kv_lens, right_batch.kv_lens], dim=0)
    merged_kv_len = int(merged_kv_lens.max().item())
    merged_mask_len = merged_kv_len + 1

    merged_k_cache: List[torch.Tensor] = []
    merged_v_cache: List[torch.Tensor] = []
    left_request_count = len(left_batch.request_ids)
    right_request_count = len(right_batch.request_ids)
    for layer_index in range(len(left_batch.k_cache)):
        layer_device = left_batch.k_cache[layer_index].device
        layer_dtype = left_batch.k_cache[layer_index].dtype
        layer_hidden = int(left_batch.k_cache[layer_index].shape[2])
        merged_layer_k = torch.zeros(
            (left_request_count + right_request_count, merged_kv_len, layer_hidden),
            dtype=layer_dtype,
            device=layer_device,
        )
        merged_layer_v = torch.zeros_like(merged_layer_k)
        for batch_index, kv_len in enumerate(left_batch.kv_lens.tolist()):
            if kv_len <= 0:
                continue
            merged_layer_k[batch_index : batch_index + 1, -kv_len:, :] = _extract_cache_row(
                left_batch.k_cache[layer_index],
                batch_index,
                int(kv_len),
                pooled=left_batch.kv_cache_pooled,
            )
            merged_layer_v[batch_index : batch_index + 1, -kv_len:, :] = _extract_cache_row(
                left_batch.v_cache[layer_index],
                batch_index,
                int(kv_len),
                pooled=left_batch.kv_cache_pooled,
            )
        for batch_index, kv_len in enumerate(right_batch.kv_lens.tolist()):
            if kv_len <= 0:
                continue
            target_index = left_request_count + batch_index
            merged_layer_k[target_index : target_index + 1, -kv_len:, :] = _extract_cache_row(
                right_batch.k_cache[layer_index],
                batch_index,
                int(kv_len),
                pooled=right_batch.kv_cache_pooled,
            )
            merged_layer_v[target_index : target_index + 1, -kv_len:, :] = _extract_cache_row(
                right_batch.v_cache[layer_index],
                batch_index,
                int(kv_len),
                pooled=right_batch.kv_cache_pooled,
            )
        merged_k_cache.append(merged_layer_k)
        merged_v_cache.append(merged_layer_v)
    merged_decode_attn_mask = torch.ones(
        (left_request_count + right_request_count, 1, 1, merged_mask_len),
        dtype=torch.bool,
        device=merged_k_cache[0].device,
    )
    for batch_index, kv_len in enumerate(merged_kv_lens.tolist()):
        merged_decode_attn_mask[batch_index : batch_index + 1, :, :, -(int(kv_len) + 1) :] = False
    if not merged_decode_attn_mask.any().item():
        merged_decode_attn_mask = None
    merged_request_ids = list(left_batch.request_ids) + list(right_batch.request_ids)
    merged_states = list(left_batch.states) + list(right_batch.states)
    merged_y_sequences = list(left_batch.y_sequences) + list(right_batch.y_sequences)
    merged_prefix_lens = torch.cat([left_batch.prefix_lens, right_batch.prefix_lens], dim=0)
    merged_step_indices = torch.cat([left_batch.step_indices, right_batch.step_indices], dim=0)

    merged_batch = T2SActiveBatch(
        request_ids=merged_request_ids,
        states=merged_states,
        x=None,
        x_lens=None,
        y_sequences=merged_y_sequences,
        prefix_lens=merged_prefix_lens,
        xy_pos=build_next_xy_pos(model, merged_y_sequences),
        key_padding_mask=None,
        prefill_attn_mask=None,
        decode_attn_mask=merged_decode_attn_mask,
        k_cache=merged_k_cache,
        v_cache=merged_v_cache,
        kv_lens=merged_kv_lens,
        step_indices=merged_step_indices,
        prefill_done=True,
        kv_cache_pooled=False,
        kv_cache_capacity=0,
        kv_cache_batch_capacity=0,
    )
    _pack_active_batch_into_pool(model, merged_batch)
    return merged_batch


@torch.inference_mode()
def run_prefill_step(
    model: Any,
    states: Sequence[T2SRequestState],
    max_steps: int,
) -> Tuple[List[T2SRunningRequest], List[T2SFinishedItem]]:
    if not states:
        return [], []

    active_batch = build_prefill_batch(model, states)
    xy_dec, k_cache, v_cache = model.t2s_transformer.process_prompt(active_batch.xy_pos, active_batch.prefill_attn_mask, None)
    decode_attn_mask = F.pad(active_batch.key_padding_mask.unsqueeze(1).unsqueeze(1), (0, 1), value=False)
    if len(states) == 1 and not decode_attn_mask.any().item():
        decode_attn_mask = None
    logits = model.ar_predict_layer(xy_dec[:, -1])
    sampling_keys = [
        _sampling_group_key(
            top_k=state.top_k,
            top_p=state.top_p,
            temperature=state.temperature,
            repetition_penalty=state.repetition_penalty,
            trim_eos=True,
        )
        for state in states
    ]
    sampled_items, argmax_tokens = _batched_sample_by_group(
        logits=logits,
        histories=active_batch.y_sequences,
        sampling_keys=sampling_keys,
    )

    running_requests: List[T2SRunningRequest] = []
    finished_items: List[T2SFinishedItem] = []

    for batch_index, state in enumerate(states):
        current_history = active_batch.y_sequences[batch_index]
        sampled = sampled_items[batch_index]
        sampled_token = int(sampled[0, 0].item())
        argmax_token = argmax_tokens[batch_index]
        new_history = torch.cat([current_history, sampled.view(-1)], dim=0)
        prefix_len = int(active_batch.prefix_lens[batch_index].item())

        finish_reason: Optional[str] = None
        if state.early_stop_num != -1 and (new_history.shape[0] - prefix_len) > state.early_stop_num:
            finish_reason = "early_stop"
        elif 1 >= max_steps:
            finish_reason = "max_step"
        elif sampled_token == model.EOS:
            finish_reason = "eos_sample"
        elif argmax_token == model.EOS:
            finish_reason = "eos_argmax"

        if finish_reason is not None:
            finished_items.append(
                T2SFinishedItem(
                    request_id=state.request_id,
                    semantic_tokens=new_history[prefix_len:-1].clone(),
                    finish_idx=0,
                    finish_reason=finish_reason,
                )
            )
            continue

        real_kv_len = int(active_batch.x_lens[batch_index].item()) + prefix_len
        request_k_cache = [layer[batch_index : batch_index + 1, -real_kv_len:, :].clone() for layer in k_cache]
        request_v_cache = [layer[batch_index : batch_index + 1, -real_kv_len:, :].clone() for layer in v_cache]
        request_decode_attn_mask = None
        if decode_attn_mask is not None:
            request_decode_attn_mask = decode_attn_mask[batch_index : batch_index + 1].clone()
            request_decode_attn_mask = _fit_decode_mask_length(request_decode_attn_mask, real_kv_len + 1)
            if not request_decode_attn_mask.any().item():
                request_decode_attn_mask = None

        running_requests.append(
            T2SRunningRequest(
                state=state,
                y_sequence=new_history,
                prefix_len=prefix_len,
                decode_attn_mask=request_decode_attn_mask,
                k_cache=request_k_cache,
                v_cache=request_v_cache,
                step_idx=1,
            )
        )

    return running_requests, finished_items


def _build_decode_batch_from_running(
    model: Any,
    running_requests: Sequence[T2SRunningRequest],
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
    xy_pos = build_next_xy_pos(model, [item.y_sequence for item in running_requests])
    max_kv_len = max(item.k_cache[0].shape[1] for item in running_requests)
    num_layers = len(running_requests[0].k_cache)

    batched_k_cache: List[torch.Tensor] = []
    batched_v_cache: List[torch.Tensor] = []
    for layer_index in range(num_layers):
        batched_k_cache.append(
            torch.cat([_pad_cache_left(item.k_cache[layer_index], max_kv_len) for item in running_requests], dim=0)
        )
        batched_v_cache.append(
            torch.cat([_pad_cache_left(item.v_cache[layer_index], max_kv_len) for item in running_requests], dim=0)
        )

    if all(item.decode_attn_mask is None for item in running_requests):
        batched_decode_attn_mask = None
    else:
        materialized_masks = [_materialize_decode_mask_for_request(item) for item in running_requests]
        max_mask_len = max(mask.shape[-1] for mask in materialized_masks)
        batched_decode_attn_mask = torch.cat(
            [_pad_decode_mask_left(mask, max_mask_len) for mask in materialized_masks],
            dim=0,
        )
    return xy_pos, batched_k_cache, batched_v_cache, batched_decode_attn_mask


@torch.inference_mode()
def run_decode_step_for_running(
    model: Any,
    running_requests: Sequence[T2SRunningRequest],
    max_steps: int,
) -> Tuple[List[T2SRunningRequest], List[T2SFinishedItem]]:
    if not running_requests:
        return [], []

    xy_pos, batched_k_cache, batched_v_cache, batched_decode_attn_mask = _build_decode_batch_from_running(
        model, running_requests
    )
    xy_dec, next_k_cache, next_v_cache = model.t2s_transformer.decode_next_token(
        xy_pos,
        batched_k_cache,
        batched_v_cache,
        batched_decode_attn_mask,
    )
    logits = model.ar_predict_layer(xy_dec[:, -1])
    sampling_keys = [
        _sampling_group_key(
            top_k=running_request.state.top_k,
            top_p=running_request.state.top_p,
            temperature=running_request.state.temperature,
            repetition_penalty=running_request.state.repetition_penalty,
            trim_eos=running_request.step_idx < 11,
        )
        for running_request in running_requests
    ]
    histories = [running_request.y_sequence for running_request in running_requests]
    sampled_items, argmax_tokens = _batched_sample_by_group(
        logits=logits,
        histories=histories,
        sampling_keys=sampling_keys,
    )

    next_running: List[T2SRunningRequest] = []
    finished_items: List[T2SFinishedItem] = []

    for batch_index, running_request in enumerate(running_requests):
        current_idx = running_request.step_idx
        sampled = sampled_items[batch_index]
        sampled_token = int(sampled[0, 0].item())
        argmax_token = argmax_tokens[batch_index]
        new_history = torch.cat([running_request.y_sequence, sampled.view(-1)], dim=0)

        finish_reason: Optional[str] = None
        if running_request.state.early_stop_num != -1 and (new_history.shape[0] - running_request.prefix_len) > running_request.state.early_stop_num:
            finish_reason = "early_stop"
        elif current_idx + 1 >= max_steps:
            finish_reason = "max_step"
        elif sampled_token == model.EOS:
            finish_reason = "eos_sample"
        elif argmax_token == model.EOS:
            finish_reason = "eos_argmax"

        if finish_reason is not None:
            finished_items.append(
                T2SFinishedItem(
                    request_id=running_request.state.request_id,
                    semantic_tokens=new_history[running_request.prefix_len:-1].clone(),
                    finish_idx=current_idx,
                    finish_reason=finish_reason,
                )
            )
            continue

        real_next_kv_len = running_request.k_cache[0].shape[1] + 1
        request_k_cache = [layer[batch_index : batch_index + 1, -real_next_kv_len:, :].clone() for layer in next_k_cache]
        request_v_cache = [layer[batch_index : batch_index + 1, -real_next_kv_len:, :].clone() for layer in next_v_cache]
        if batched_decode_attn_mask is None:
            next_decode_attn_mask = None
        else:
            current_decode_mask_len = running_request.k_cache[0].shape[1] + 1
            current_decode_attn_mask = batched_decode_attn_mask[
                batch_index : batch_index + 1, :, :, -current_decode_mask_len:
            ]
            next_decode_attn_mask = F.pad(current_decode_attn_mask, (0, 1), value=False)
            next_decode_attn_mask = _fit_decode_mask_length(next_decode_attn_mask, real_next_kv_len + 1)
            if not next_decode_attn_mask.any().item():
                next_decode_attn_mask = None
        next_running.append(
            T2SRunningRequest(
                state=running_request.state,
                y_sequence=new_history,
                prefix_len=running_request.prefix_len,
                decode_attn_mask=next_decode_attn_mask,
                k_cache=request_k_cache,
                v_cache=request_v_cache,
                step_idx=current_idx + 1,
            )
        )

    return next_running, finished_items


@torch.inference_mode()
def run_scheduler_continuous(
    model: Any,
    states: Sequence[T2SRequestState],
    max_steps: int,
) -> List[T2SFinishedItem]:
    pending = sorted(states, key=lambda item: (item.ready_step, item.request_id))
    active_batch: Optional[T2SActiveBatch] = None
    finished: List[T2SFinishedItem] = []
    current_tick = 0

    while pending or active_batch is not None:
        admitted: List[T2SRequestState] = []
        while pending and pending[0].ready_step <= current_tick:
            admitted.append(pending.pop(0))

        admitted_active_batch, admitted_finished = run_prefill_active_batch(model, admitted, max_steps=max_steps)
        finished.extend(admitted_finished)
        active_batch = merge_active_batches(model, active_batch, admitted_active_batch)

        if active_batch is not None:
            active_batch, step_finished = decode_one_step(model, active_batch, max_steps=max_steps)
            finished.extend(step_finished)

        if active_batch is None and pending:
            current_tick = max(current_tick + 1, pending[0].ready_step)
            continue

        current_tick += 1

    finished.sort(key=lambda item: item.request_id)
    return finished
