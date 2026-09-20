"""Shared resume processing helpers used by routes and workers."""
from __future__ import annotations

import logging

from app.config import Config
from app.models import Resume, db

logger = logging.getLogger(__name__)


def create_resume_draft(
    *,
    candidate_id: int,
    raw_text: str,
    filename: str | None = None,
    job_id: int | None = None,
    job_title: str | None = None,
    job_requirements: str | None = None,
    org_id: int | None = None,
) -> Resume:
    """Persist resume row in queued state (scoring happens async)."""
    resume = Resume(
        candidate_id=candidate_id,
        org_id=org_id,
        filename=filename,
        raw_text=raw_text,
        job_id=job_id,
        target_job_title=job_title,
        target_job_requirements=job_requirements,
        processing_status="queued",
        pipeline_stage="new",
    )
    db.session.add(resume)
    db.session.commit()
    return resume


def apply_screening_result(resume: Resume, result: dict) -> Resume:
    scores = result.get("scores") or {}
    structured = result.get("structured") or {}
    resume.set_structured(structured)
    resume.total_score = scores.get("total_score")
    resume.skill_score = scores.get("skill_score")
    resume.project_score = scores.get("project_score")
    resume.education_score = scores.get("education_score")
    resume.fit_score = scores.get("fit_score")
    resume.recommendation = scores.get("recommendation")
    resume.recommendation_reason = scores.get("recommendation_reason")
    if scores.get("details"):
        resume.set_score_details(scores["details"])
    resume.meets_threshold = bool(
        scores.get("meets_threshold")
        if "meets_threshold" in scores
        else (resume.total_score or 0) >= Config.RESUME_SCORE_THRESHOLD
    )
    return resume


def index_resume_rag(resume: Resume, candidate_name: str = "") -> None:
    try:
        from app.services.rag import get_rag

        # Index resume body only — never prepend full job JD (pollutes retrieval).
        get_rag().index_resume(
            resume_id=resume.id,
            candidate_id=resume.candidate_id,
            raw_text=resume.raw_text or "",
            structured=resume.structured(),
            candidate_name=candidate_name,
            org_id=resume.org_id,
            job_title=resume.target_job_title,
        )
    except Exception:
        logger.exception("RAG index failed for resume %s", resume.id)


def process_resume_text(
    *,
    candidate_id: int,
    candidate_name: str,
    raw_text: str,
    filename: str | None = None,
    resume: Resume | None = None,
    job_id: int | None = None,
    job_title: str | None = None,
    job_requirements: str | None = None,
) -> Resume:
    """Legacy sync path: create + score in-request (used by tests / INLINE)."""
    from app.agents.graph_runtime import invoke_agent, new_thread_id

    if resume is None:
        resume = create_resume_draft(
            candidate_id=candidate_id,
            raw_text=raw_text,
            filename=filename,
            job_id=job_id,
            job_title=job_title,
            job_requirements=job_requirements,
        )
    else:
        resume.raw_text = raw_text
        if filename:
            resume.filename = filename
        resume.job_id = job_id
        resume.target_job_title = job_title
        resume.target_job_requirements = job_requirements
        resume.processing_status = "scoring"
        db.session.commit()

    resume.processing_status = "scoring"
    db.session.commit()

    thread_id = new_thread_id()
    result = invoke_agent(
        {
            "intent": "resume_screen",
            "role": "candidate",
            "user_id": candidate_id,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "resume_id": resume.id,
            "raw_text": raw_text,
            "job_title": job_title or "",
            "job_requirements": job_requirements or "",
            "query": f"请对照岗位「{job_title or '未命名'}」筛选这份简历",
        },
        thread_id=thread_id,
    )
    apply_screening_result(resume, result)
    resume.processing_status = "ready"
    if not resume.pipeline_stage or resume.pipeline_stage == "new":
        resume.pipeline_stage = "screening"
    db.session.commit()
    index_resume_rag(resume, candidate_name)
    logger.info(
        "Resume %s scored sync for job=%s total=%s meets=%s",
        resume.id,
        job_title,
        resume.total_score,
        resume.meets_threshold,
    )
    return resume
