"""Background worker: python -m app.worker"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app import create_app
from app.config import Config
from app.services import jobs as jobq
from app.services.job_handlers import run_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hirepilot.worker")


def main() -> None:
    app = create_app()
    last_sweep_day: str | None = None
    logger.info("Worker started (poll=%ss inline=%s)", Config.WORKER_POLL_SECONDS, Config.INLINE_JOBS)
    with app.app_context():
        while True:
            # Daily PII retention enqueue once per calendar day
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if day != last_sweep_day:
                jobq.enqueue(jobq.KIND_PII_RETENTION, {})
                last_sweep_day = day

            job = jobq.claim_next()
            if job:
                logger.info("Running job %s kind=%s", job.id, job.kind)
                run_job(job.id)
            else:
                time.sleep(Config.WORKER_POLL_SECONDS)


if __name__ == "__main__":
    main()
