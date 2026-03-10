from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass


@dataclass
class JobState:
    job_id: str
    product_id: str
    generation_id: str
    status: str = "pending"  # pending | running | completed | failed
    progress: float = 0.0
    message: str = ""
    error: str = ""


class JobManager:
    """Manages background job state and progress."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._events: dict[str, list[threading.Event]] = {}

    def create(self, product_id: str, generation_id: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = JobState(
                job_id=job_id,
                product_id=product_id,
                generation_id=generation_id,
            )
        return job_id

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(
        self, job_id: str, progress: float, message: str = ""
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "running"
                job.progress = progress
                job.message = message
        self._notify(job_id)

    def complete(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "completed"
                job.progress = 1.0
                job.message = "Done"
        self._notify(job_id)

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = error
        self._notify(job_id)

    def subscribe(self, job_id: str) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._events.setdefault(job_id, []).append(event)
        return event

    def unsubscribe(self, job_id: str, event: threading.Event) -> None:
        with self._lock:
            listeners = self._events.get(job_id, [])
            if event in listeners:
                listeners.remove(event)

    def list_active(self) -> list[JobState]:
        """Return all jobs that are still pending or running."""
        with self._lock:
            return [
                j for j in self._jobs.values()
                if j.status in ("pending", "running")
            ]

    def _notify(self, job_id: str) -> None:
        with self._lock:
            for event in self._events.get(job_id, []):
                event.set()
