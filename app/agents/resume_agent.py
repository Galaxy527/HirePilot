"""简历筛选子 Agent — parse + score + index（对照企业岗位需求）。"""
from __future__ import annotations

import logging
from typing import TypedDict

from app.config import Config
from app.services.rag import get_rag
from app.services.scoring import score_resume, structure_resume
from app.services.weights import get_dim_weights

logger = logging.getLogger(__name__)


class ResumeAgentState(TypedDict, total=False):
    raw_text: str
    candidate_id: int
    resume_id: int
    org_id: int
    candidate_name: str
    job_title: str
    job_requirements: str
    structured: dict
    scores: dict
    error: str


def run_resume_screening(state: ResumeAgentState) -> ResumeAgentState:
    """Checkpointable resume screening step."""
    raw = state.get("raw_text") or ""
    if not raw.strip():
        return {**state, "error": "简历文本为空"}

    job_title = state.get("job_title") or ""
    job_requirements = state.get("job_requirements") or ""

    org_id = state.get("org_id")
    if not org_id and state.get("resume_id"):
        try:
            from app.models import Resume, db

            resume = db.session.get(Resume, int(state["resume_id"]))
            if resume:
                org_id = resume.org_id
        except Exception:
            logger.exception("resolve org_id for scoring failed")

    structured = structure_resume(raw)
    result = score_resume(
        raw,
        structured,
        job_title=job_title,
        job_requirements=job_requirements,
        dim_weights=get_dim_weights(org_id),
    )
    threshold = Config.RESUME_SCORE_THRESHOLD
    scores = {
        "total_score": result.total_score,
        "skill_score": result.skill_score,
        "project_score": result.project_score,
        "education_score": result.education_score,
        "fit_score": result.fit_score,
        "recommendation": result.recommendation,
        "recommendation_reason": result.recommendation_reason,
        "meets_threshold": result.total_score >= threshold,
        "details": result.details,
    }

    resume_id = state.get("resume_id")
    candidate_id = state.get("candidate_id")
    if resume_id and candidate_id:
        try:
            get_rag().index_resume(
                resume_id=resume_id,
                candidate_id=candidate_id,
                raw_text=raw,
                structured=result.structured,
                candidate_name=state.get("candidate_name") or result.structured.get("name", ""),
                org_id=org_id,
                job_title=job_title,
            )
        except Exception as exc:
            logger.exception("RAG index failed: %s", exc)

    return {
        **state,
        "structured": result.structured,
        "scores": scores,
        "error": "",
    }
