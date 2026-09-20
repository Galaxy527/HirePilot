"""HR visibility / RBAC helpers (org + admin/recruiter)."""
from __future__ import annotations

from flask import abort
from sqlalchemy import or_

from app.models import Resume, User


def hr_resume_query(user: User, *, include_all_assigned_scope: bool = True):
    """Base resume query scoped to HR org + role."""
    if not user.is_hr or not user.org_id:
        return Resume.query.filter_by(id=-1)  # empty
    q = Resume.query.filter(Resume.org_id == user.org_id)
    if user.is_hr_admin:
        return q
    # recruiter: assigned to me OR unassigned pool
    if include_all_assigned_scope:
        return q.filter(
            or_(Resume.assigned_hr_id == user.id, Resume.assigned_hr_id.is_(None))
        )
    return q.filter(Resume.assigned_hr_id == user.id)


def can_hr_view_resume(user: User, resume: Resume) -> bool:
    if not user.is_hr:
        return False
    if resume.org_id and user.org_id and resume.org_id != user.org_id:
        return False
    if user.is_hr_admin:
        return True
    return resume.assigned_hr_id in (None, user.id)


def require_hr_resume(user: User, resume: Resume) -> Resume:
    if not can_hr_view_resume(user, resume):
        abort(403)
    return resume


def can_force_search(user: User) -> bool:
    """Force search across org (still org-scoped) — admin only."""
    return user.is_hr_admin
