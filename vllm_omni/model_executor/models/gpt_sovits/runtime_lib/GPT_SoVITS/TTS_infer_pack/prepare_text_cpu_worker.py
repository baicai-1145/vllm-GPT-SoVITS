import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Sequence, Tuple


@dataclass
class TextCpuTask:
    text: str
    language: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)
    enqueued_at: float = 0.0
    admission_wait_ms: float = 0.0
    backpressure_wait_ms: float = 0.0
    capacity_wait_ms: float = 0.0
    pending_depth_on_enqueue: int = 0
    batch_popped_at: float = 0.0
    done_event: threading.Event = field(default_factory=threading.Event)
    done_loop: asyncio.AbstractEventLoop | None = None
    done_future: asyncio.Future | None = None
    result: Any = None
    error: Exception | None = None
    profile: Dict[str, float] = field(default_factory=dict)


class PrepareTextCpuWorker:
    def __init__(
        self,
        process_fn: Callable[[str, str], Any] | None = None,
        batch_process_fn: Callable[[Sequence[Tuple[str, str]]], Sequence[Any]] | None = None,
        worker_count: int = 1,
        batch_window_ms: int = 0,
        max_batch_items: int = 1,
        high_pressure_pending_threshold: int = 0,
        high_pressure_batch_window_ms: int | None = None,
        high_pressure_max_batch_items: int | None = None,
        max_pending_tasks: int = 0,
        admission_poll_ms: int = 1,
        admission_controller: Callable[[], Dict[str, float | int | bool]] | None = None,
        large_batch_priority: bool = True,
        min_batch_window_ms: int = 2,
        min_high_pressure_batch_window_ms: int | None = None,
    ) -> None:
        if process_fn is None and batch_process_fn is None:
            raise ValueError("PrepareTextCpuWorker 需要 process_fn 或 batch_process_fn")
        self.process_fn = process_fn
        self.batch_process_fn = batch_process_fn
        self.worker_count = max(1, int(worker_count))
        self.max_batch_items = max(1, int(max_batch_items))
        self.large_batch_priority = bool(large_batch_priority)
        self.min_batch_window_ms = max(0, int(min_batch_window_ms))
        hp_min_window_ms = self.min_batch_window_ms if min_high_pressure_batch_window_ms is None else int(
            min_high_pressure_batch_window_ms
        )
        self.min_high_pressure_batch_window_ms = max(0, hp_min_window_ms)
        self.batch_window_ms = self._resolve_effective_batch_window_ms(
            int(batch_window_ms),
            self.max_batch_items,
            self.min_batch_window_ms,
        )
        self.batch_window_s = float(self.batch_window_ms) / 1000.0
        self.high_pressure_pending_threshold = max(
            0,
            int(high_pressure_pending_threshold)
            if int(high_pressure_pending_threshold) > 0
            else max(self.max_batch_items * 2, 64),
        )
        hp_items = self.max_batch_items if high_pressure_max_batch_items is None else int(high_pressure_max_batch_items)
        hp_window_ms = self.batch_window_ms if high_pressure_batch_window_ms is None else int(high_pressure_batch_window_ms)
        self.high_pressure_max_batch_items = max(self.max_batch_items, hp_items)
        self.high_pressure_batch_window_ms = self._resolve_effective_batch_window_ms(
            hp_window_ms,
            self.high_pressure_max_batch_items,
            self.min_high_pressure_batch_window_ms,
        )
        self.high_pressure_batch_window_s = float(self.high_pressure_batch_window_ms) / 1000.0
        self.max_pending_tasks = max(0, int(max_pending_tasks))
        self.admission_poll_s = max(0.0005, float(max(1, int(admission_poll_ms))) / 1000.0)
        self.admission_controller = admission_controller

        self.condition = threading.Condition()
        self.pending_tasks: Deque[TextCpuTask] = deque()
        self.pending_peak = 0
        self.total_submitted = 0
        self.total_finished = 0
        self.total_batches = 0
        self.active_workers = 0
        self.active_workers_peak = 0
        self.active_tasks = 0
        self.active_tasks_peak = 0
        self.active_batch_peak = 0
        self.high_pressure_batches = 0
        self.admission_wait_total_ms = 0.0
        self.admission_wait_peak_ms = 0.0
        self.backpressure_wait_total_ms = 0.0
        self.backpressure_wait_peak_ms = 0.0
        self.capacity_wait_total_ms = 0.0
        self.capacity_wait_peak_ms = 0.0
        self.backpressure_blocked_total = 0

        self.worker_threads = [
            threading.Thread(target=self._run_loop, name=f"prepare-text-cpu-worker-{index}", daemon=True)
            for index in range(self.worker_count)
        ]
        for thread in self.worker_threads:
            thread.start()

    def _resolve_effective_batch_window_ms(
        self,
        batch_window_ms: int,
        max_batch_items: int,
        min_batch_window_ms: int,
    ) -> int:
        effective_window_ms = max(0, int(batch_window_ms))
        if not self.large_batch_priority or int(max_batch_items) <= 1:
            return effective_window_ms
        return max(effective_window_ms, max(0, int(min_batch_window_ms)))

    def _can_enqueue_locked(self, task_count: int = 1) -> bool:
        if self.max_pending_tasks <= 0:
            return True
        return (len(self.pending_tasks) + self.active_tasks + max(0, int(task_count)) - 1) < self.max_pending_tasks

    def _get_admission_state(self) -> Dict[str, float | int | bool]:
        if self.admission_controller is None:
            return {"blocked": False}
        try:
            state = dict(self.admission_controller() or {})
        except Exception:
            return {"blocked": False}
        state["blocked"] = bool(state.get("blocked", False))
        return state

    def _record_enqueue_locked(
        self,
        task: TextCpuTask,
        *,
        admission_wait_ms: float,
        backpressure_wait_ms: float,
        capacity_wait_ms: float,
    ) -> None:
        task.admission_wait_ms = float(max(0.0, admission_wait_ms))
        task.backpressure_wait_ms = float(max(0.0, backpressure_wait_ms))
        task.capacity_wait_ms = float(max(0.0, capacity_wait_ms))
        task.enqueued_at = time.perf_counter()
        task.pending_depth_on_enqueue = int(len(self.pending_tasks))
        self.pending_tasks.append(task)
        self.total_submitted += 1
        self.admission_wait_total_ms += task.admission_wait_ms
        self.admission_wait_peak_ms = max(self.admission_wait_peak_ms, task.admission_wait_ms)
        self.backpressure_wait_total_ms += task.backpressure_wait_ms
        self.backpressure_wait_peak_ms = max(self.backpressure_wait_peak_ms, task.backpressure_wait_ms)
        self.capacity_wait_total_ms += task.capacity_wait_ms
        self.capacity_wait_peak_ms = max(self.capacity_wait_peak_ms, task.capacity_wait_ms)
        if task.backpressure_wait_ms > 0.0:
            self.backpressure_blocked_total += 1
        if len(self.pending_tasks) > self.pending_peak:
            self.pending_peak = len(self.pending_tasks)
        self.condition.notify_all()

    async def _enqueue_tasks_async(self, tasks: Sequence[TextCpuTask]) -> None:
        if not tasks:
            return
        admission_started = time.perf_counter()
        backpressure_wait_ms = 0.0
        capacity_wait_ms = 0.0
        while True:
            loop_start = time.perf_counter()
            admission_state = self._get_admission_state()
            blocked = bool(admission_state.get("blocked", False))
            with self.condition:
                if not blocked and self._can_enqueue_locked(len(tasks)):
                    waited_ms = (time.perf_counter() - admission_started) * 1000.0
                    for task in tasks:
                        self._record_enqueue_locked(
                            task,
                            admission_wait_ms=waited_ms,
                            backpressure_wait_ms=backpressure_wait_ms,
                            capacity_wait_ms=capacity_wait_ms,
                        )
                    return
            await asyncio.sleep(self.admission_poll_s)
            waited_ms = (time.perf_counter() - loop_start) * 1000.0
            if blocked:
                backpressure_wait_ms += waited_ms
            else:
                capacity_wait_ms += waited_ms

    async def _enqueue_task_async(self, task: TextCpuTask) -> None:
        await self._enqueue_tasks_async([task])

    def submit(self, text: str, language: str) -> Tuple[Any, Dict[str, float]]:
        task = TextCpuTask(text=str(text), language=str(language))
        asyncio.run(self._enqueue_task_async(task))
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        return task.result, dict(task.profile)

    async def submit_async(self, text: str, language: str) -> Tuple[Any, Dict[str, float]]:
        loop = asyncio.get_running_loop()
        task = TextCpuTask(
            text=str(text),
            language=str(language),
            done_loop=loop,
            done_future=loop.create_future(),
        )
        await self._enqueue_task_async(task)
        return await task.done_future

    async def submit_many_async(self, items: Sequence[Tuple[str, str]]) -> List[Tuple[Any, Dict[str, float]]]:
        if not items:
            return []
        loop = asyncio.get_running_loop()
        tasks = [
            TextCpuTask(
                text=str(text),
                language=str(language),
                done_loop=loop,
                done_future=loop.create_future(),
            )
            for text, language in items
        ]
        await self._enqueue_tasks_async(tasks)
        return list(await asyncio.gather(*[task.done_future for task in tasks]))

    @staticmethod
    def _resolve_done_future(task: TextCpuTask) -> None:
        if task.done_future is None or task.done_future.done():
            return
        if task.error is not None:
            task.done_future.set_exception(task.error)
            return
        task.done_future.set_result((task.result, dict(task.profile)))

    def _notify_task_done(self, task: TextCpuTask) -> None:
        task.done_event.set()
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_done_future, task)
        except RuntimeError:
            pass

    def snapshot(self) -> Dict[str, int | float]:
        with self.condition:
            return {
                "worker_count": int(self.worker_count),
                "pending": int(len(self.pending_tasks)),
                "pending_peak": int(self.pending_peak),
                "active_workers": int(self.active_workers),
                "active_workers_peak": int(self.active_workers_peak),
                "active_tasks": int(self.active_tasks),
                "active_tasks_peak": int(self.active_tasks_peak),
                "active_batch_peak": int(self.active_batch_peak),
                "total_submitted": int(self.total_submitted),
                "total_finished": int(self.total_finished),
                "total_batches": int(self.total_batches),
                "batch_window_ms": int(self.batch_window_ms),
                "max_batch_items": int(self.max_batch_items),
                "high_pressure_pending_threshold": int(self.high_pressure_pending_threshold),
                "high_pressure_batch_window_ms": int(self.high_pressure_batch_window_ms),
                "high_pressure_max_batch_items": int(self.high_pressure_max_batch_items),
                "large_batch_priority": bool(self.large_batch_priority),
                "min_batch_window_ms": int(self.min_batch_window_ms),
                "min_high_pressure_batch_window_ms": int(self.min_high_pressure_batch_window_ms),
                "high_pressure_batches": int(self.high_pressure_batches),
                "max_pending_tasks": int(self.max_pending_tasks),
                "admission_wait_total_ms": float(self.admission_wait_total_ms),
                "admission_wait_peak_ms": float(self.admission_wait_peak_ms),
                "backpressure_wait_total_ms": float(self.backpressure_wait_total_ms),
                "backpressure_wait_peak_ms": float(self.backpressure_wait_peak_ms),
                "capacity_wait_total_ms": float(self.capacity_wait_total_ms),
                "capacity_wait_peak_ms": float(self.capacity_wait_peak_ms),
                "backpressure_blocked_total": int(self.backpressure_blocked_total),
            }

    def _select_batch_policy_locked(self) -> Tuple[float, int, bool]:
        pending_depth = len(self.pending_tasks)
        use_high_pressure = (
            self.high_pressure_pending_threshold > 0
            and pending_depth >= self.high_pressure_pending_threshold
        )
        if use_high_pressure:
            return self.high_pressure_batch_window_s, self.high_pressure_max_batch_items, True
        return self.batch_window_s, self.max_batch_items, False

    def _collect_batch(self) -> Tuple[List[TextCpuTask], bool]:
        with self.condition:
            while not self.pending_tasks:
                self.condition.wait()

            batch_window_s, max_batch_items, use_high_pressure = self._select_batch_policy_locked()
            first_task = self.pending_tasks.popleft()
            first_task.batch_popped_at = time.perf_counter()
            batch: List[TextCpuTask] = [first_task]
            deadline = time.perf_counter() + batch_window_s

            while len(batch) < max_batch_items:
                while self.pending_tasks and len(batch) < max_batch_items:
                    next_task = self.pending_tasks.popleft()
                    next_task.batch_popped_at = time.perf_counter()
                    batch.append(next_task)
                if len(batch) >= max_batch_items:
                    break
                next_batch_window_s, next_max_batch_items, next_use_high_pressure = self._select_batch_policy_locked()
                if next_use_high_pressure:
                    if not use_high_pressure:
                        deadline = max(deadline, time.perf_counter() + next_batch_window_s)
                    max_batch_items = max(max_batch_items, next_max_batch_items)
                    use_high_pressure = True
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                if not self.pending_tasks:
                    self.condition.wait(timeout=remaining)
                    continue
                next_task = self.pending_tasks.popleft()
                next_task.batch_popped_at = time.perf_counter()
                batch.append(next_task)

            self.active_workers += 1
            self.active_workers_peak = max(self.active_workers_peak, self.active_workers)
            self.active_tasks += len(batch)
            self.active_tasks_peak = max(self.active_tasks_peak, self.active_tasks)
            self.active_batch_peak = max(self.active_batch_peak, len(batch))
            if use_high_pressure:
                self.high_pressure_batches += 1
            return batch, use_high_pressure

    def _finalize_batch(self, batch: Sequence[TextCpuTask]) -> None:
        with self.condition:
            self.active_workers = max(0, self.active_workers - 1)
            self.active_tasks = max(0, self.active_tasks - len(batch))
            self.total_batches += 1
            self.total_finished += len(batch)
            self.condition.notify_all()

    def _run_batch(self, batch: List[TextCpuTask], use_high_pressure: bool) -> None:
        batch_collected_ts = time.perf_counter()
        batch_started = time.perf_counter()
        try:
            if self.batch_process_fn is not None:
                results = list(self.batch_process_fn([(task.text, task.language) for task in batch]))
            else:
                assert self.process_fn is not None
                results = [self.process_fn(task.text, task.language) for task in batch]
            if len(results) != len(batch):
                raise RuntimeError(
                    f"text cpu batch 结果数量不匹配: expected={len(batch)} actual={len(results)}"
                )
            batch_finished = time.perf_counter()
            batch_run_ms = max(0.0, (batch_finished - batch_started) * 1000.0)
            for task, result in zip(batch, results):
                task.result = result
                worker_queue_wait_ms = max(0.0, (float(task.batch_popped_at) - float(task.enqueued_at)) * 1000.0)
                batch_collect_wait_ms = max(0.0, (float(batch_collected_ts) - float(task.batch_popped_at)) * 1000.0)
                batch_dispatch_delay_ms = max(0.0, (float(batch_started) - float(batch_collected_ts)) * 1000.0)
                task.profile = {
                    "text_cpu_admission_wait_ms": float(task.admission_wait_ms),
                    "text_cpu_backpressure_wait_ms": float(task.backpressure_wait_ms),
                    "text_cpu_capacity_wait_ms": float(task.capacity_wait_ms),
                    "text_cpu_queue_wait_ms": max(0.0, (batch_started - task.enqueued_at) * 1000.0),
                    "text_cpu_worker_queue_wait_ms": worker_queue_wait_ms,
                    "text_cpu_batch_collect_wait_ms": batch_collect_wait_ms,
                    "text_cpu_batch_dispatch_delay_ms": batch_dispatch_delay_ms,
                    "text_cpu_pending_depth_on_enqueue": float(task.pending_depth_on_enqueue),
                    "text_cpu_run_ms": batch_run_ms,
                    "text_cpu_batch_size": float(len(batch)),
                    "text_cpu_high_pressure_mode": 1.0 if use_high_pressure else 0.0,
                }
        except Exception as exc:  # noqa: PERF203
            for task in batch:
                task.error = exc
        finally:
            self._finalize_batch(batch)
            for task in batch:
                self._notify_task_done(task)

    def _run_loop(self) -> None:
        while True:
            batch, use_high_pressure = self._collect_batch()
            self._run_batch(batch, use_high_pressure)
