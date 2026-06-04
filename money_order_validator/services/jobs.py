from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from money_order_validator.settings import settings


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, meta: Optional[Dict[str, Any]] = None) -> str:
        self.cleanup()
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "processing",
            "job_id": job_id,
            "progress": meta or {},
            "created_at": time.time(),
            "result": None,
            "error": None,
        }
        return job_id

    def update_progress(self, job_id: str, progress: Dict[str, Any]) -> None:
        job = self._jobs.get(job_id)
        if job:
            job["progress"] = {**(job.get("progress") or {}), **progress}

    def complete(self, job_id: str, result: Dict[str, Any]) -> None:
        job = self._jobs.get(job_id)
        if job:
            job["status"] = "done"
            job["result"] = result
            job["progress"] = {**(job.get("progress") or {}), "done": True}

    def fail(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job["status"] = "failed"
            job["error"] = error

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        self.cleanup()
        return self._jobs.get(job_id)

    def cleanup(self) -> None:
        cutoff = time.time() - settings.result_retention_minutes * 60
        for job_id in list(self._jobs):
            if self._jobs[job_id].get("created_at", 0) < cutoff:
                self._jobs.pop(job_id, None)


job_store = JobStore()
