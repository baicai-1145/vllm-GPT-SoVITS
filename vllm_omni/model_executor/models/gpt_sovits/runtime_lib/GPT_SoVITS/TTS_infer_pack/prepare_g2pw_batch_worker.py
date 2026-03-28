import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Sequence, Tuple


def _segment_char_weight(segment: Any) -> int:
    if not bool(getattr(segment, "needs_g2pw", False)):
        return 0
    norm_text = str(getattr(segment, "norm_text", "") or "")
    return max(1, len(norm_text)) if norm_text else 0


@dataclass
class G2PWBatchTask:
    segment_batches: List[List[Any]]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)
    enqueued_at: float = 0.0
    admission_wait_ms: float = 0.0
    pending_depth_on_enqueue: int = 0
    batch_popped_at: float = 0.0
    done_event: threading.Event = field(default_factory=threading.Event)
    done_loop: asyncio.AbstractEventLoop | None = None
    done_future: asyncio.Future | None = None
    result_batches: List[List[Any]] | None = None
    batch_profiles: List[Dict[str, float]] | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    error: Exception | None = None
    worker_profile: Dict[str, float] = field(default_factory=dict)


class PrepareG2PWBatchWorker:
    def __init__(
        self,
        resolve_batch_fn: Callable[[List[List[Any]], List[Dict[str, float]] | None], List[List[Any]]],
        batch_window_ms: int = 2,
        max_batch_tasks: int = 64,
        max_batch_groups: int = 128,
        max_batch_chars: int = 4096,
        max_pending_tasks: int = 0,
        admission_poll_ms: int = 1,
        high_pressure_pending_threshold: int = 0,
        high_pressure_batch_window_ms: int | None = None,
        high_pressure_max_batch_tasks: int | None = None,
        high_pressure_max_batch_groups: int | None = None,
        high_pressure_max_batch_chars: int | None = None,
    ):
        self.resolve_batch_fn = resolve_batch_fn
        self.batch_window_ms = max(0, int(batch_window_ms))
        self.batch_window_s = float(self.batch_window_ms) / 1000.0
        self.max_batch_tasks = max(1, int(max_batch_tasks))
        self.max_batch_groups = max(1, int(max_batch_groups))
        self.max_batch_chars = max(1, int(max_batch_chars))
        self.max_pending_tasks = max(0, int(max_pending_tasks))
        self.admission_poll_s = max(0.0005, float(max(1, int(admission_poll_ms))) / 1000.0)
        self.high_pressure_pending_threshold = max(
            0,
            int(high_pressure_pending_threshold)
            if int(high_pressure_pending_threshold) > 0
            else max(self.max_batch_tasks * 2, 32),
        )
        hp_window_ms = self.batch_window_ms if high_pressure_batch_window_ms is None else int(high_pressure_batch_window_ms)
        hp_tasks = self.max_batch_tasks if high_pressure_max_batch_tasks is None else int(high_pressure_max_batch_tasks)
        hp_groups = self.max_batch_groups if high_pressure_max_batch_groups is None else int(high_pressure_max_batch_groups)
        hp_chars = self.max_batch_chars if high_pressure_max_batch_chars is None else int(high_pressure_max_batch_chars)
        self.high_pressure_batch_window_ms = max(0, hp_window_ms)
        self.high_pressure_batch_window_s = float(self.high_pressure_batch_window_ms) / 1000.0
        self.high_pressure_max_batch_tasks = max(self.max_batch_tasks, hp_tasks)
        self.high_pressure_max_batch_groups = max(self.max_batch_groups, hp_groups)
        self.high_pressure_max_batch_chars = max(self.max_batch_chars, hp_chars)

        self.condition = threading.Condition()
        self.pending_tasks: Deque[G2PWBatchTask] = deque()
        self.pending_peak = 0
        self.total_submitted = 0
        self.total_finished = 0
        self.total_batches = 0
        self.active_batch_size = 0
        self.active_batch_peak = 0
        self.active_batch_groups = 0
        self.active_batch_groups_peak = 0
        self.active_batch_chars = 0
        self.active_batch_chars_peak = 0
        self.high_pressure_batches = 0
        self.admission_wait_total_ms = 0.0
        self.admission_wait_peak_ms = 0.0
        self.worker_thread = threading.Thread(
            target=self._run_loop,
            name="prepare-g2pw-batch-worker",
            daemon=True,
        )
        self.worker_thread.start()

    def _estimate_task_groups(self, task: G2PWBatchTask) -> int:
        return int(len(task.segment_batches))

    def _estimate_task_chars(self, task: G2PWBatchTask) -> int:
        total = 0
        for segment_batch in task.segment_batches:
            total += sum(_segment_char_weight(segment) for segment in segment_batch)
        return max(1, int(total))

    def _can_enqueue_locked(self) -> bool:
        if self.max_pending_tasks <= 0:
            return True
        return (len(self.pending_tasks) + self.active_batch_size) < self.max_pending_tasks

    def _record_enqueue_locked(self, task: G2PWBatchTask, admission_wait_ms: float) -> None:
        task.admission_wait_ms = float(max(0.0, admission_wait_ms))
        task.enqueued_at = time.perf_counter()
        task.pending_depth_on_enqueue = int(len(self.pending_tasks))
        self.pending_tasks.append(task)
        self.total_submitted += 1
        self.admission_wait_total_ms += task.admission_wait_ms
        self.admission_wait_peak_ms = max(self.admission_wait_peak_ms, task.admission_wait_ms)
        self.pending_peak = max(self.pending_peak, len(self.pending_tasks))
        self.condition.notify_all()

    async def _enqueue_task_async(self, task: G2PWBatchTask) -> None:
        admission_started = time.perf_counter()
        while True:
            with self.condition:
                if self._can_enqueue_locked():
                    self._record_enqueue_locked(task, (time.perf_counter() - admission_started) * 1000.0)
                    return
            await asyncio.sleep(self.admission_poll_s)

    async def submit_async(
        self,
        segment_batches: Sequence[Sequence[Any]],
    ) -> Tuple[List[List[Any]], List[Dict[str, float]], Dict[str, float], float, float, float]:
        loop = asyncio.get_running_loop()
        task = G2PWBatchTask(
            segment_batches=[list(segment_batch) for segment_batch in segment_batches],
            done_loop=loop,
            done_future=loop.create_future(),
        )
        await self._enqueue_task_async(task)
        return await task.done_future

    def snapshot(self) -> Dict[str, int | float]:
        with self.condition:
            pending_groups = sum(self._estimate_task_groups(task) for task in self.pending_tasks)
            pending_chars = sum(self._estimate_task_chars(task) for task in self.pending_tasks)
            return {
                "pending": int(len(self.pending_tasks)),
                "pending_peak": int(self.pending_peak),
                "pending_groups": int(pending_groups),
                "pending_chars": int(pending_chars),
                "outstanding": int(len(self.pending_tasks) + self.active_batch_size),
                "total_submitted": int(self.total_submitted),
                "total_finished": int(self.total_finished),
                "total_batches": int(self.total_batches),
                "active_batch_size": int(self.active_batch_size),
                "active_batch_peak": int(self.active_batch_peak),
                "active_batch_groups": int(self.active_batch_groups),
                "active_batch_groups_peak": int(self.active_batch_groups_peak),
                "active_batch_chars": int(self.active_batch_chars),
                "active_batch_chars_peak": int(self.active_batch_chars_peak),
                "batch_window_ms": int(self.batch_window_ms),
                "max_batch_tasks": int(self.max_batch_tasks),
                "max_batch_groups": int(self.max_batch_groups),
                "max_batch_chars": int(self.max_batch_chars),
                "max_pending_tasks": int(self.max_pending_tasks),
                "high_pressure_pending_threshold": int(self.high_pressure_pending_threshold),
                "high_pressure_batch_window_ms": int(self.high_pressure_batch_window_ms),
                "high_pressure_max_batch_tasks": int(self.high_pressure_max_batch_tasks),
                "high_pressure_max_batch_groups": int(self.high_pressure_max_batch_groups),
                "high_pressure_max_batch_chars": int(self.high_pressure_max_batch_chars),
                "high_pressure_batches": int(self.high_pressure_batches),
                "admission_wait_total_ms": float(self.admission_wait_total_ms),
                "admission_wait_peak_ms": float(self.admission_wait_peak_ms),
            }

    def _select_batch_policy_locked(self) -> Tuple[float, int, int, int, bool, int]:
        pending_depth = len(self.pending_tasks)
        use_high_pressure = (
            self.high_pressure_pending_threshold > 0 and pending_depth >= self.high_pressure_pending_threshold
        )
        if use_high_pressure:
            return (
                self.high_pressure_batch_window_s,
                self.high_pressure_max_batch_tasks,
                self.high_pressure_max_batch_groups,
                self.high_pressure_max_batch_chars,
                True,
                pending_depth,
            )
        return (
            self.batch_window_s,
            self.max_batch_tasks,
            self.max_batch_groups,
            self.max_batch_chars,
            False,
            pending_depth,
        )

    def _collect_batch(self) -> Tuple[List[G2PWBatchTask], Dict[str, float]]:
        with self.condition:
            while not self.pending_tasks:
                self.condition.wait()
            collect_started = time.perf_counter()
            batch_window_s, max_batch_tasks, max_batch_groups, max_batch_chars, use_high_pressure, pending_depth = (
                self._select_batch_policy_locked()
            )
            first_task = self.pending_tasks.popleft()
            first_task.batch_popped_at = time.perf_counter()
            batch = [first_task]
            batch_groups = self._estimate_task_groups(first_task)
            batch_chars = self._estimate_task_chars(first_task)
            deadline = time.perf_counter() + batch_window_s
            while len(batch) < max_batch_tasks:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                if not self.pending_tasks:
                    self.condition.wait(timeout=remaining)
                    continue
                next_task = self.pending_tasks[0]
                next_groups = self._estimate_task_groups(next_task)
                next_chars = self._estimate_task_chars(next_task)
                if (batch_groups + next_groups) > max_batch_groups or (batch_chars + next_chars) > max_batch_chars:
                    break
                popped_task = self.pending_tasks.popleft()
                popped_task.batch_popped_at = time.perf_counter()
                batch.append(popped_task)
                batch_groups += next_groups
                batch_chars += next_chars
            self.active_batch_size = len(batch)
            self.active_batch_groups = int(batch_groups)
            self.active_batch_chars = int(batch_chars)
            self.active_batch_peak = max(self.active_batch_peak, self.active_batch_size)
            self.active_batch_groups_peak = max(self.active_batch_groups_peak, self.active_batch_groups)
            self.active_batch_chars_peak = max(self.active_batch_chars_peak, self.active_batch_chars)
            if use_high_pressure:
                self.high_pressure_batches += 1
            collected_at = time.perf_counter()
            return batch, {
                "collect_wait_ms": float((time.perf_counter() - collect_started) * 1000.0),
                "collected_at_ts": float(collected_at),
                "batch_groups": float(batch_groups),
                "batch_chars": float(batch_chars),
                "pending_depth_on_collect": float(pending_depth),
                "high_pressure_mode": 1.0 if use_high_pressure else 0.0,
                "batch_window_ms": float(self.high_pressure_batch_window_ms if use_high_pressure else self.batch_window_ms),
            }

    def _finalize_batch(self, batch: List[G2PWBatchTask]) -> None:
        with self.condition:
            self.active_batch_size = 0
            self.active_batch_groups = 0
            self.active_batch_chars = 0
            self.total_batches += 1
            self.total_finished += len(batch)
            self.condition.notify_all()

    @staticmethod
    def _resolve_done_future(task: G2PWBatchTask) -> None:
        if task.done_future is None or task.done_future.done():
            return
        if task.error is not None:
            task.done_future.set_exception(task.error)
            return
        assert task.result_batches is not None
        assert task.batch_profiles is not None
        task.done_future.set_result(
            (
                task.result_batches,
                [dict(profile) for profile in task.batch_profiles],
                dict(task.worker_profile),
                float(task.enqueued_at),
                float(task.started_at),
                float(task.finished_at),
            )
        )

    def _notify_done_future(self, task: G2PWBatchTask) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_done_future, task)
        except RuntimeError:
            pass

    def _run_batch(self, batch: List[G2PWBatchTask], batch_meta: Dict[str, float]) -> None:
        batch_started = time.perf_counter()
        batch_collected_ts = float(batch_meta.get("collected_at_ts", batch_started))
        flat_batches: List[List[Any]] = []
        flat_profiles: List[Dict[str, float]] = []
        task_slices: List[Tuple[int, int]] = []
        for task in batch:
            start = len(flat_batches)
            task.result_batches = None
            task.batch_profiles = None
            for segment_batch in task.segment_batches:
                flat_batches.append(list(segment_batch))
                flat_profiles.append({})
            task_slices.append((start, len(flat_batches)))
        resolved_batches = self.resolve_batch_fn(flat_batches, profiles=flat_profiles)
        batch_finished = time.perf_counter()
        for task, (start, end) in zip(batch, task_slices):
            task.started_at = float(batch_started)
            task.finished_at = float(batch_finished)
            task.result_batches = [list(batch_result) for batch_result in resolved_batches[start:end]]
            task.batch_profiles = [dict(profile) for profile in flat_profiles[start:end]]
            worker_queue_wait_ms = max(0.0, (float(task.batch_popped_at) - float(task.enqueued_at)) * 1000.0)
            batch_collect_wait_ms = max(0.0, (float(batch_collected_ts) - float(task.batch_popped_at)) * 1000.0)
            batch_dispatch_delay_ms = max(0.0, (float(batch_started) - float(batch_collected_ts)) * 1000.0)
            task.worker_profile = {
                "g2pw_wait_ms": float((batch_started - task.created_at) * 1000.0),
                "g2pw_admission_wait_ms": float(task.admission_wait_ms),
                "g2pw_worker_queue_wait_ms": float(worker_queue_wait_ms),
                "g2pw_batch_collect_wait_ms": float(batch_collect_wait_ms),
                "g2pw_batch_dispatch_delay_ms": float(batch_dispatch_delay_ms),
                "g2pw_batch_size": float(len(batch)),
                "g2pw_batch_groups": float(batch_meta["batch_groups"]),
                "g2pw_batch_chars": float(batch_meta["batch_chars"]),
                "g2pw_pending_depth_on_enqueue": float(task.pending_depth_on_enqueue),
                "g2pw_pending_depth_on_collect": float(batch_meta["pending_depth_on_collect"]),
                "g2pw_high_pressure_mode": float(batch_meta["high_pressure_mode"]),
                "g2pw_batch_window_ms": float(batch_meta["batch_window_ms"]),
            }
        for task in batch:
            task.done_event.set()
            self._notify_done_future(task)

    def _run_loop(self) -> None:
        while True:
            batch, batch_meta = self._collect_batch()
            try:
                self._run_batch(batch, batch_meta)
            except Exception as exc:  # noqa: PERF203
                for task in batch:
                    task.error = exc
                    task.done_event.set()
                    self._notify_done_future(task)
            finally:
                self._finalize_batch(batch)


class PrepareG2PWBatchWorkerPool:
    def __init__(
        self,
        resolve_batch_fn: Callable[[List[List[Any]], List[Dict[str, float]] | None], List[List[Any]]],
        batch_window_ms: int = 2,
        max_batch_tasks: int = 64,
        max_batch_groups: int = 128,
        max_batch_chars: int = 4096,
        max_pending_tasks: int = 0,
        admission_poll_ms: int = 1,
        high_pressure_pending_threshold: int = 0,
        high_pressure_batch_window_ms: int | None = None,
        high_pressure_max_batch_tasks: int | None = None,
        high_pressure_max_batch_groups: int | None = None,
        high_pressure_max_batch_chars: int | None = None,
        worker_count: int = 1,
    ):
        self.worker_count = max(1, int(worker_count))
        self.lock = threading.Lock()
        self.shards = [
            PrepareG2PWBatchWorker(
                resolve_batch_fn=resolve_batch_fn,
                batch_window_ms=batch_window_ms,
                max_batch_tasks=max_batch_tasks,
                max_batch_groups=max_batch_groups,
                max_batch_chars=max_batch_chars,
                max_pending_tasks=max_pending_tasks,
                admission_poll_ms=admission_poll_ms,
                high_pressure_pending_threshold=high_pressure_pending_threshold,
                high_pressure_batch_window_ms=high_pressure_batch_window_ms,
                high_pressure_max_batch_tasks=high_pressure_max_batch_tasks,
                high_pressure_max_batch_groups=high_pressure_max_batch_groups,
                high_pressure_max_batch_chars=high_pressure_max_batch_chars,
            )
            for _ in range(self.worker_count)
        ]

    def _pick_shard(self) -> PrepareG2PWBatchWorker:
        with self.lock:
            return min(
                self.shards,
                key=lambda shard: (
                    int(shard.snapshot().get("outstanding", 0)),
                    int(shard.snapshot().get("pending_chars", 0)),
                    int(shard.snapshot().get("active_batch_size", 0)),
                ),
            )

    async def submit_async(
        self,
        segment_batches: Sequence[Sequence[Any]],
    ) -> Tuple[List[List[Any]], List[Dict[str, float]], Dict[str, float], float, float, float]:
        shard = self._pick_shard()
        resolved_batches, batch_profiles, worker_profile, submit_at, started_at, finished_at = await shard.submit_async(
            segment_batches
        )
        merged_profile = dict(worker_profile)
        merged_profile["g2pw_pool_workers"] = float(self.worker_count)
        return resolved_batches, batch_profiles, merged_profile, submit_at, started_at, finished_at

    def snapshot(self) -> Dict[str, int | float | List[Dict[str, int | float]]]:
        shard_snapshots = [dict(shard.snapshot()) for shard in self.shards]
        return {
            "worker_count": int(self.worker_count),
            "pending": int(sum(int(snapshot.get("pending", 0)) for snapshot in shard_snapshots)),
            "pending_peak": int(max((int(snapshot.get("pending_peak", 0)) for snapshot in shard_snapshots), default=0)),
            "pending_groups": int(sum(int(snapshot.get("pending_groups", 0)) for snapshot in shard_snapshots)),
            "pending_chars": int(sum(int(snapshot.get("pending_chars", 0)) for snapshot in shard_snapshots)),
            "outstanding": int(sum(int(snapshot.get("outstanding", 0)) for snapshot in shard_snapshots)),
            "total_submitted": int(sum(int(snapshot.get("total_submitted", 0)) for snapshot in shard_snapshots)),
            "total_finished": int(sum(int(snapshot.get("total_finished", 0)) for snapshot in shard_snapshots)),
            "total_batches": int(sum(int(snapshot.get("total_batches", 0)) for snapshot in shard_snapshots)),
            "active_batch_size": int(sum(int(snapshot.get("active_batch_size", 0)) for snapshot in shard_snapshots)),
            "active_batch_peak": int(max((int(snapshot.get("active_batch_peak", 0)) for snapshot in shard_snapshots), default=0)),
            "active_batch_groups": int(sum(int(snapshot.get("active_batch_groups", 0)) for snapshot in shard_snapshots)),
            "active_batch_groups_peak": int(
                max((int(snapshot.get("active_batch_groups_peak", 0)) for snapshot in shard_snapshots), default=0)
            ),
            "active_batch_chars": int(sum(int(snapshot.get("active_batch_chars", 0)) for snapshot in shard_snapshots)),
            "active_batch_chars_peak": int(
                max((int(snapshot.get("active_batch_chars_peak", 0)) for snapshot in shard_snapshots), default=0)
            ),
            "batch_window_ms": int(shard_snapshots[0].get("batch_window_ms", 0)) if shard_snapshots else 0,
            "max_batch_tasks": int(shard_snapshots[0].get("max_batch_tasks", 0)) if shard_snapshots else 0,
            "max_batch_groups": int(shard_snapshots[0].get("max_batch_groups", 0)) if shard_snapshots else 0,
            "max_batch_chars": int(shard_snapshots[0].get("max_batch_chars", 0)) if shard_snapshots else 0,
            "max_pending_tasks": int(shard_snapshots[0].get("max_pending_tasks", 0)) if shard_snapshots else 0,
            "high_pressure_pending_threshold": (
                int(shard_snapshots[0].get("high_pressure_pending_threshold", 0)) if shard_snapshots else 0
            ),
            "high_pressure_batch_window_ms": (
                int(shard_snapshots[0].get("high_pressure_batch_window_ms", 0)) if shard_snapshots else 0
            ),
            "high_pressure_max_batch_tasks": (
                int(shard_snapshots[0].get("high_pressure_max_batch_tasks", 0)) if shard_snapshots else 0
            ),
            "high_pressure_max_batch_groups": (
                int(shard_snapshots[0].get("high_pressure_max_batch_groups", 0)) if shard_snapshots else 0
            ),
            "high_pressure_max_batch_chars": (
                int(shard_snapshots[0].get("high_pressure_max_batch_chars", 0)) if shard_snapshots else 0
            ),
            "high_pressure_batches": int(
                sum(int(snapshot.get("high_pressure_batches", 0)) for snapshot in shard_snapshots)
            ),
            "admission_wait_total_ms": float(
                sum(float(snapshot.get("admission_wait_total_ms", 0.0)) for snapshot in shard_snapshots)
            ),
            "admission_wait_peak_ms": float(
                max((float(snapshot.get("admission_wait_peak_ms", 0.0)) for snapshot in shard_snapshots), default=0.0)
            ),
            "shards": shard_snapshots,
        }
