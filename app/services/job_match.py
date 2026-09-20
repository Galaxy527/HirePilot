"""Match candidate job title to HR-published JobPosting."""
from __future__ import annotations

import re

from app.models import JobPosting


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\s\-_/·•,，.。()（）\[\]【】]+", "", text)
    return text


def resolve_job_for_candidate(
    *,
    job_id: int | None = None,
    job_title: str | None = None,
    org_id: int | None = None,
) -> tuple[JobPosting | None, str, str, str]:
    """
    Resolve HR job posting for scoring.

    Returns: (job, title, requirements, match_note)
    """
    title = (job_title or "").strip()

    def _scope(q):
        if org_id is not None:
            return q.filter_by(org_id=org_id)
        return q

    if job_id:
        job = _scope(JobPosting.query.filter_by(id=job_id, is_active=True)).first()
        if job:
            return job, job.title, job.requirements, f"已匹配 HR 岗位「{job.title}」"

    if not title:
        return None, "", "", "未填写求职岗位"

    jobs = _scope(JobPosting.query.filter_by(is_active=True)).all()
    norm = _normalize(title)
    exact = [j for j in jobs if _normalize(j.title) == norm]
    if exact:
        job = exact[0]
        return job, job.title, job.requirements, f"已精确匹配 HR 岗位「{job.title}」"

    scored: list[tuple[float, JobPosting]] = []
    for j in jobs:
        jt = _normalize(j.title)
        if not jt:
            continue
        score = 0.0
        if norm in jt or jt in norm:
            score = max(score, min(len(norm), len(jt)) / max(len(norm), len(jt)))
        inter = len(set(norm) & set(jt))
        union = len(set(norm) | set(jt)) or 1
        score = max(score, inter / union)
        if score >= 0.45:
            scored.append((score, j))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, job = scored[0]
        return (
            job,
            job.title,
            job.requirements,
            f"已模糊匹配 HR 岗位「{job.title}」（相似度 {best_score:.0%}）",
        )

    return None, title, "", f"未找到与「{title}」对应的 HR 岗位，请从列表选择或联系 HR 发布该岗位"
