"""Database-backed background job queue with retry + dead letter."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.config import Config
from app.models import BackgroundJob, db
from app.services import metrics as metrics_svc

logger = logging.getLogger(__name__)

KIND_SCORE_RESUME = "score_resume"
KIND_GENERATE_INTERVIEW = "generate_interview_questions"
KIND_PII_RETENTION = "pii_retention_sweep"


def enqueue(kind: str, payload: dict | None = None) -> BackgroundJob:
    job = BackgroundJob(
        kind=kind,
        status="pending",
        progress=0,
        attempts=0,
        max_attempts=Config.JOB_MAX_ATTEMPTS,
        next_run_at=datetime.now(timezone.utc),
    )
    job.set_payload(payload or {})
    db.session.add(job)
    db.session.commit()
    metrics_svc.incr("jobs_enqueued")
    logger.info("Enqueued job %s kind=%s", job.id, kind)

    if Config.INLINE_JOBS:
        from app.services.job_handlers import run_job

        run_job(job.id)
        db.session.refresh(job)
    return job


def claim_next() -> BackgroundJob | None:
    """Claim one pending job whose next_run_at has arrived."""
    now = datetime.now(timezone.utc)
    job = (
        BackgroundJob.query.filter(
            BackgroundJob.status == "pending",
            (BackgroundJob.next_run_at.is_(None)) | (BackgroundJob.next_run_at <= now),
        )
        .order_by(BackgroundJob.id.asc())
        .first()
    )
    if not job:
        return None
    updated = BackgroundJob.query.filter_by(id=job.id, status="pending").update(
        {"status": "running", "progress": 5}
    )
    db.session.commit()
    if not updated:
        return None
    return db.session.get(BackgroundJob, job.id)


def mark_succeeded(job: BackgroundJob) -> None:
    job.status = "succeeded"
    job.progress = 100
    job.finished_at = datetime.now(timezone.utc)
    db.session.commit()
    metrics_svc.incr("jobs_succeeded")


def mark_failed(job: BackgroundJob, error: str) -> None:
    """Retry with backoff or move to dead letter."""
    job.attempts = int(job.attempts or 0) + 1
    job.error = (error or "")[:2000]
    max_a = int(job.max_attempts or Config.JOB_MAX_ATTEMPTS)
    if job.attempts < max_a:
        delay = Config.JOB_RETRY_BASE_SECONDS * (2 ** (job.attempts - 1))
        job.status = "pending"
        job.progress = 0
        job.next_run_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
        job.finished_at = None
        db.session.commit()
        metrics_svc.incr("jobs_retried")
        logger.warning(
            "Job %s failed attempt %s/%s; retry in %ss: %s",
            job.id,
            job.attempts,
            max_a,
            delay,
            error,
        )
        return

    job.status = "dead"
    job.finished_at = datetime.now(timezone.utc)
    db.session.commit()
    metrics_svc.incr("jobs_dead")
    logger.error("Job %s moved to dead letter after %s attempts: %s", job.id, job.attempts, error)


def dead_letter_count() -> int:
    return BackgroundJob.query.filter_by(status="dead").count()


def pending_count() -> int:
    return BackgroundJob.query.filter_by(status="pending").count()
