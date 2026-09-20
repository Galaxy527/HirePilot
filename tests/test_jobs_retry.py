"""Job retry / dead-letter basics."""


def test_mark_failed_retries_then_dead(app):
    with app.app_context():
        from datetime import datetime, timezone

        from app.config import Config
        from app.models import BackgroundJob, db
        from app.services import jobs as jobq

        Config.JOB_MAX_ATTEMPTS = 2
        Config.JOB_RETRY_BASE_SECONDS = 0.01
        job = BackgroundJob(
            kind="noop",
            status="running",
            attempts=0,
            max_attempts=2,
            next_run_at=datetime.now(timezone.utc),
        )
        job.set_payload({})
        db.session.add(job)
        db.session.commit()

        jobq.mark_failed(job, "boom1")
        db.session.refresh(job)
        assert job.status == "pending"
        assert job.attempts == 1

        job.status = "running"
        db.session.commit()
        jobq.mark_failed(job, "boom2")
        db.session.refresh(job)
        assert job.status == "dead"
        assert job.attempts == 2
