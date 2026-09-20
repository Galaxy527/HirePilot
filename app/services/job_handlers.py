"""Job handler implementations executed by worker / inline mode."""
from __future__ import annotations

import logging

from app.agents.graph_runtime import invoke_agent, new_thread_id
from app.models import BackgroundJob, InterviewSession, Resume, db
from app.services import jobs as jobq
from app.services.resume_service import apply_screening_result, index_resume_rag

logger = logging.getLogger(__name__)


def run_job(job_id: int) -> None:
    job = db.session.get(BackgroundJob, job_id)
    if not job:
        return
    if job.status == "pending":
        job.status = "running"
        job.progress = 5
        db.session.commit()
    try:
        if job.kind == jobq.KIND_SCORE_RESUME:
            _handle_score_resume(job)
        elif job.kind == jobq.KIND_GENERATE_INTERVIEW:
            _handle_generate_interview(job)
        elif job.kind == jobq.KIND_PII_RETENTION:
            from app.services.pii import sweep_expired_resumes

            n = sweep_expired_resumes()
            job.progress = 100
            jobq.mark_succeeded(job)
            logger.info("PII sweep deleted %s resumes", n)
            return
        else:
            raise ValueError(f"Unknown job kind: {job.kind}")
        jobq.mark_succeeded(job)
    except Exception as exc:
        logger.exception("Job %s crashed", job_id)
        payload = job.payload()
        before = job.attempts or 0
        jobq.mark_failed(job, str(exc))
        db.session.refresh(job)
        if job.status == "dead" and job.kind == jobq.KIND_SCORE_RESUME:
            resume_id = payload.get("resume_id")
            if resume_id:
                resume = db.session.get(Resume, int(resume_id))
                if resume:
                    resume.processing_status = "failed"
                    db.session.commit()


def _handle_score_resume(job: BackgroundJob) -> None:
    from app.services.notify import notify_hrs

    payload = job.payload()
    resume_id = int(payload["resume_id"])
    candidate_name = payload.get("candidate_name") or ""
    resume = db.session.get(Resume, resume_id)
    if not resume:
        raise ValueError(f"resume {resume_id} not found")

    resume.processing_status = "scoring"
    job.progress = 20
    db.session.commit()

    thread_id = new_thread_id()
    result = invoke_agent(
        {
            "intent": "resume_screen",
            "role": "candidate",
            "user_id": resume.candidate_id,
            "candidate_id": resume.candidate_id,
            "candidate_name": candidate_name,
            "resume_id": resume.id,
            "org_id": resume.org_id,
            "raw_text": resume.raw_text,
            "job_title": resume.target_job_title or "",
            "job_requirements": resume.target_job_requirements or "",
            "query": f"请对照岗位「{resume.target_job_title or '未命名'}」筛选这份简历",
        },
        thread_id=thread_id,
    )
    job.progress = 70
    db.session.commit()

    apply_screening_result(resume, result)
    resume.processing_status = "ready"
    if not resume.pipeline_stage or resume.pipeline_stage == "new":
        resume.pipeline_stage = "screening"
    db.session.commit()

    index_resume_rag(resume, candidate_name)

    if resume.meets_threshold:
        notify_hrs(
            title="新达标候选人",
            body=(
                f"{candidate_name or '候选人'} · 简历 #{resume.id} · "
                f"{resume.target_job_title or ''} · 分 {resume.total_score}"
            ),
            link=f"/hr/resume/{resume.id}",
            org_id=resume.org_id,
        )
    job.progress = 95
    db.session.commit()


def _handle_generate_interview(job: BackgroundJob) -> None:
    payload = job.payload()
    session_id = int(payload["session_id"])
    session = db.session.get(InterviewSession, session_id)
    if not session:
        raise ValueError(f"session {session_id} not found")
    resume = session.resume
    thread_id = session.checkpoint_thread_id or new_thread_id()
    result = invoke_agent(
        {
            "intent": "interview_generate",
            "role": "candidate",
            "user_id": resume.candidate_id,
            "candidate_id": resume.candidate_id,
            "resume_summary": resume.raw_text[:3000],
            "structured": resume.structured(),
            "job_title": resume.target_job_title or "",
            "job_requirements": resume.target_job_requirements or "",
            "query": "开始模拟面试",
        },
        thread_id=thread_id,
    )
    qs = result.get("questions") or []
    session.set_questions(qs)
    session.checkpoint_thread_id = thread_id
    db.session.commit()
    logger.info(
        "Interview session %s questions ready count=%s from_llm=%s",
        session.id,
        len(qs),
        result.get("questions_from_llm"),
    )
