"""PII retention and candidate data deletion."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import Config
from app.models import ChatSession, InterviewSession, Message, Notification, Resume, ResumeNote, db
from app.services.audit import audit

logger = logging.getLogger(__name__)


def _purge_resume_rag(resume_id: int) -> None:
    try:
        from app.services.rag import get_rag

        rag = get_rag()
        rag.chunks = [c for c in rag.chunks if c.metadata.get("resume_id") != resume_id]
        if rag.chunks:
            rag._vectors = rag._embed_texts([c.text for c in rag.chunks])
        else:
            rag._vectors = None
        rag._rebuild_faiss()
        rag._save()
    except Exception:
        logger.exception("RAG cleanup failed for resume %s", resume_id)


def delete_one_resume(resume_id: int, *, actor_id: int, candidate_id: int) -> bool:
    """Delete a single resume owned by candidate_id (RAG + related rows)."""
    resume = db.session.get(Resume, resume_id)
    if not resume or resume.candidate_id != candidate_id:
        return False
    return _delete_resume_record(resume, actor_id=actor_id, by_hr=False)


def hr_delete_resume(resume_id: int, *, actor_id: int, org_id: int | None) -> bool:
    """HR deletes a resume in their org (RAG + interviews + notes + upload file)."""
    resume = db.session.get(Resume, resume_id)
    if not resume:
        return False
    if org_id is not None and resume.org_id not in (None, org_id):
        return False
    return _delete_resume_record(resume, actor_id=actor_id, by_hr=True)


def _delete_resume_record(resume: Resume, *, actor_id: int, by_hr: bool = False) -> bool:
    from app.models import LiveInterview

    resume_id = resume.id
    candidate_id = resume.candidate_id
    _purge_resume_rag(resume_id)
    InterviewSession.query.filter_by(resume_id=resume_id).delete()
    ResumeNote.query.filter_by(resume_id=resume_id).delete()
    LiveInterview.query.filter_by(resume_id=resume_id).delete()

    filename = resume.filename
    db.session.delete(resume)
    db.session.commit()

    if filename:
        upload_dir = Path(Config.UPLOAD_FOLDER)
        if upload_dir.exists():
            for p in upload_dir.iterdir():
                if p.is_file() and filename in p.name:
                    try:
                        p.unlink()
                    except OSError:
                        pass

    audit(
        "resume_deleted",
        "resume",
        resume_id,
        detail={"candidate_id": candidate_id, "by_hr": by_hr},
        actor_id=actor_id,
    )
    return True


def delete_candidate_data(user_id: int, *, actor_id: int | None = None) -> dict:
    """Cascade-delete resumes, interviews, chats, notes, local uploads, RAG chunks."""
    resumes = Resume.query.filter_by(candidate_id=user_id).all()
    resume_ids = [r.id for r in resumes]
    filenames = [r.filename for r in resumes if r.filename]
    files_removed = 0

    for r in resumes:
        _purge_resume_rag(r.id)
        InterviewSession.query.filter_by(resume_id=r.id).delete()
        ResumeNote.query.filter_by(resume_id=r.id).delete()
        db.session.delete(r)

    upload_dir = Path(Config.UPLOAD_FOLDER)
    if upload_dir.exists() and filenames:
        for p in upload_dir.iterdir():
            if p.is_file() and any(fn in p.name for fn in filenames):
                try:
                    p.unlink()
                    files_removed += 1
                except OSError:
                    pass

    sessions = ChatSession.query.filter_by(user_id=user_id).all()
    for s in sessions:
        Message.query.filter_by(session_id=s.id).delete()
        db.session.delete(s)

    Notification.query.filter_by(user_id=user_id).delete()
    db.session.commit()

    audit(
        "candidate_data_deleted",
        "user",
        user_id,
        detail={"resume_ids": resume_ids, "files_removed": files_removed},
        actor_id=actor_id or user_id,
    )
    return {"resumes": len(resume_ids), "files_removed": files_removed}


def sweep_expired_resumes() -> int:
    """Hard-delete resumes older than retention window."""
    days = max(1, Config.PII_RETENTION_DAYS)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    old = Resume.query.filter(Resume.created_at < cutoff).all()
    count = 0
    for r in old:
        cid = r.candidate_id
        rid = r.id
        InterviewSession.query.filter_by(resume_id=rid).delete()
        ResumeNote.query.filter_by(resume_id=rid).delete()
        _purge_resume_rag(rid)
        db.session.delete(r)
        count += 1
        audit(
            "pii_retention_delete",
            "resume",
            rid,
            detail={"candidate_id": cid, "retention_days": days},
            actor_id=None,
        )
    db.session.commit()
    return count
