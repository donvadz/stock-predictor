"""
Job Manager for persistent background jobs.

Jobs run in background threads and survive page refresh/tab close.
Only explicit cancellation stops a job.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a background job."""
    id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    progress_message: str = ""
    result: Any = None
    error: Optional[str] = None
    cancelled: bool = False
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert job to dictionary for API response."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "result": self.result,
            "error": self.error,
            "cancelled": self.cancelled,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self._get_elapsed_seconds(),
        }

    def _get_elapsed_seconds(self) -> int:
        """Get elapsed time in seconds."""
        if self.started_at is None:
            return 0
        end_time = self.completed_at if self.completed_at else time.time()
        return int(end_time - self.started_at)


class JobManager:
    """
    Thread-safe manager for background jobs.

    Jobs persist in memory until:
    - Explicitly deleted
    - TTL expires (1 hour after completion)
    """

    JOB_TTL = 3600  # 1 hour

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background thread to clean up old jobs."""
        def cleanup_loop():
            while not self._shutdown:
                time.sleep(300)  # Check every 5 minutes
                self._cleanup_old_jobs()

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_old_jobs(self):
        """Remove jobs that have been completed for more than TTL."""
        now = time.time()
        with self._lock:
            expired_ids = []
            for job_id, job in self._jobs.items():
                if job.completed_at and (now - job.completed_at) > self.JOB_TTL:
                    expired_ids.append(job_id)
            for job_id in expired_ids:
                del self._jobs[job_id]

    def create_job(self, job_type: str) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())[:8]  # Short ID for convenience
        job = Job(id=job_id, job_type=job_type)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Request cancellation of a job.

        Sets the cancelled flag - the job worker must check this flag
        and stop processing accordingly.

        Returns True if job was found and cancellation was requested.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False  # Already finished
            job.cancelled = True
            return True

    def start_job(
        self,
        job: Job,
        worker_fn: Callable[[Job], Any],
    ) -> None:
        """
        Start a job in a background thread.

        Args:
            job: The job to run
            worker_fn: Function that takes the job and performs work.
                       Should check job.cancelled periodically and return early if True.
                       Should update job.progress and job.progress_message.
                       Should return the result or raise an exception.
        """
        def run_job():
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            try:
                result = worker_fn(job)
                if job.cancelled:
                    job.status = JobStatus.CANCELLED
                    job.progress_message = "Cancelled by user"
                else:
                    job.status = JobStatus.COMPLETED
                    job.result = result
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
            finally:
                job.completed_at = time.time()

        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

    def list_jobs(self, job_type: Optional[str] = None) -> list:
        """List all jobs, optionally filtered by type."""
        with self._lock:
            jobs = list(self._jobs.values())
            if job_type:
                jobs = [j for j in jobs if j.job_type == job_type]
            return [j.to_dict() for j in jobs]

    def shutdown(self):
        """Clean shutdown of the job manager."""
        self._shutdown = True


# Global instance
job_manager = JobManager()
